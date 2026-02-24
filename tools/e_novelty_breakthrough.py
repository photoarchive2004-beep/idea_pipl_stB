#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module E — Cross-Domain & Breakthrough Scoring (Universal)

Purpose
- Provide *measurable* (best-effort) proxies for:
  - Conventionality (foundation + coherence of the literature base)
  - Novel combination (rarity of topic combinations inside the collected corpus)
  - Bridge feasibility (is there already a bridge literature, and how strong/clean is it?)
  - Breakthrough potential (novelty x feasibility, adjusted by evidence quality flags)

Inputs (per idea folder)
- REQUIRED:
  - out/corpus.csv  (from Module B)
- OPTIONAL (improves diagnostics):
  - out/structured_idea.json (from Module A)
  - out/evidence_table.csv   (from Module C)
  - out/gap_map.json         (from Module D)

Outputs (per idea folder)
- out/novelty_breakthrough.json
- out/bridge_ideas.md
- out/llm_prompt_E.txt  (only when --ask-llm)

Design constraints
- Robust: never crash; always writes output stubs when inputs missing/invalid.
- Universal: no idea-specific assumptions.
- Minimal deps: stdlib only.

Note on metrics
- Uzzi-style novelty uses a *global* reference set (millions of papers). We cannot ship that.
  Here we compute a within-corpus proxy (topic-cooccurrence PMI ranks) that is still useful
  for *relative* comparison across your ideas when Stage B settings are consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------- utilities -------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_text(p: Path, encodings: Sequence[str] = ("utf-8", "utf-8-sig", "cp1251")) -> str:
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    try:
        return p.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return ""


def safe_read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(safe_read_text(p))
    except Exception:
        return {}


# ----------------- encoding repair (Windows mojibake) -----------------
def _cp1252_reverse_map() -> Dict[str, int]:
    """Return mapping unicode-char -> byte for Windows-1252, but keep undefined bytes as control chars.
    This helps reverse typical mojibake like 'ÐŸÑ...' back to Cyrillic.
    """
    mp: Dict[str, int] = {}
    for b in range(256):
        try:
            ch = bytes([b]).decode("cp1252")
        except UnicodeDecodeError:
            ch = chr(b)  # keep as control char (U+0081 etc.)
        mp[ch] = b
    return mp


_CP1252_REV = _cp1252_reverse_map()


def _looks_like_utf8_mojibake(s: str) -> bool:
    # Typical symptoms: lots of Ð/Ñ/â sequences and/or C1 controls (U+0080..U+009F)
    if not s:
        return False
    if not ("Ð" in s or "Ñ" in s or "â" in s or "Ã" in s):
        return False
    return any((0x80 <= ord(c) <= 0x9F) for c in s) or ("Ð" in s or "Ñ" in s)


def _try_repair_mojibake(s: str) -> str:
    """Best-effort repair for strings where UTF-8 bytes were mis-decoded as Windows-1252/Latin1."""
    if not _looks_like_utf8_mojibake(s):
        return s
    out = bytearray()
    for ch in s:
        o = ord(ch)
        if o <= 0xFF:
            out.append(o)
        elif ch in _CP1252_REV:
            out.append(_CP1252_REV[ch])
        else:
            # last resort: keep the character as UTF-8 bytes
            out.extend(ch.encode("utf-8", errors="ignore"))
    try:
        repaired = out.decode("utf-8")
    except Exception:
        return s
    # accept only if it looks "better" (drops typical mojibake markers)
    if ("Ð" in repaired) or ("Ñ" in repaired) or ("â" in repaired) or ("Ã" in repaired):
        # could still be valid text in some domains; keep original
        return repaired if _has_cyrillic(repaired) and not _has_cyrillic(s) else s
    return repaired


def _has_cyrillic(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF or 0x0500 <= o <= 0x052F:
            return True
    return False


def deep_repair(obj: Any) -> Any:
    """Recursively repair mojibake in dict/list/str objects."""
    if isinstance(obj, str):
        return _try_repair_mojibake(obj)
    if isinstance(obj, list):
        return [deep_repair(x) for x in obj]
    if isinstance(obj, dict):
        return {deep_repair(k) if isinstance(k, str) else k: deep_repair(v) for k, v in obj.items()}
    return obj
# ----------------------------------------------------------------------

def sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    except Exception:
        class D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return D()


def read_csv_best_effort(path: Path) -> List[Dict[str, str]]:
    raw = None
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except Exception:
            continue
    if raw is None:
        try:
            raw = path.read_bytes().decode("utf-8", errors="replace")
        except Exception:
            return []

    sample = raw[:65536]
    dialect = sniff_dialect(sample)

    try:
        reader = csv.DictReader(raw.splitlines(), dialect=dialect)
        rows: List[Dict[str, str]] = []
        for r in reader:
            if not r:
                continue
            rr = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            if any(rr.values()):
                rows.append(rr)
        return rows
    except Exception:
        try:
            reader = csv.DictReader(raw.splitlines(), delimiter=",")
            return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in reader if r]
        except Exception:
            return []


def clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def to_score01(x: float) -> int:
    return int(round(100.0 * clamp01(x)))


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float((x or "").strip()))
    except Exception:
        return default


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float((x or "").strip())
    except Exception:
        return default


def split_list_field(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    # OpenAlex output uses '; ' for concepts/topics, but be defensive
    parts = re.split(r"\s*[;|,]\s*", s)
    return [p.strip() for p in parts if p.strip()]


def percentile_rank(sorted_vals: Sequence[float], x: float) -> float:
    """Return fraction of values <= x (0..1). sorted_vals must be sorted."""
    if not sorted_vals:
        return 0.0
    lo, hi = 0, len(sorted_vals)
    # bisect_right
    while lo < hi:
        mid = (lo + hi) // 2
        if x < sorted_vals[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo / len(sorted_vals)


def log_line(log_fp: Optional[Any], msg: str) -> None:
    if log_fp is None:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_fp.write(f"[{ts}] {msg}\n")
    log_fp.flush()


# ------------------------- core computation -------------------------

def build_topic_sets(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Counter]:
    """Extract per-paper topics (list), cited_by, doi/title/year."""
    papers: List[Dict[str, Any]] = []
    topic_freq: Counter = Counter()

    for r in rows:
        topics = split_list_field(r.get("top_topics", ""))
        if not topics:
            # allow fallback to concepts if topics missing
            topics = split_list_field(r.get("top_concepts", ""))
        topics = list(dict.fromkeys([t for t in topics if t]))  # unique, keep order

        cited_by = safe_int(r.get("cited_by", "0"), 0)
        year = safe_int(r.get("year", "0"), 0)
        doi = (r.get("doi") or "").strip()
        title = (r.get("title") or "").strip()
        venue = (r.get("venue") or "").strip()

        papers.append({
            "doi": doi,
            "title": title,
            "year": year,
            "venue": venue,
            "cited_by": cited_by,
            "topics": topics,
        })
        for t in topics:
            topic_freq[t] += 1

    return papers, topic_freq


def cooccurrence_counts(papers: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for p in papers:
        ts = p.get("topics") or []
        # limit to first 8 to avoid pathological huge lists
        ts = ts[:8]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1
    return pair_counts


def pmi(obs: float, fa: float, fb: float, n: float, smoothing: float = 0.5) -> float:
    """PMI proxy with small smoothing; returns log2((obs/n)/((fa/n)*(fb/n)))."""
    if n <= 0:
        return 0.0
    obs2 = obs if obs > 0 else smoothing
    pa = fa / n
    pb = fb / n
    pab = obs2 / n
    if pa <= 0 or pb <= 0 or pab <= 0:
        return 0.0
    return math.log(pab / (pa * pb), 2)


def pick_two_domains(topic_freq: Counter, pair_counts: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
    """Pick two "distinct" frequent topics to seed two domains."""
    top = [t for t, _ in topic_freq.most_common(20)]
    if not top:
        return "", ""
    t1 = top[0]
    if len(top) == 1:
        return t1, ""

    def co(a: str, b: str) -> int:
        if a > b:
            a, b = b, a
        return pair_counts.get((a, b), 0)

    # pick t2 among next topics with minimal co-occurrence with t1 (to maximize separation)
    cand = top[1:]
    cand_sorted = sorted(cand, key=lambda t: (co(t1, t), -topic_freq[t], t))
    t2 = cand_sorted[0] if cand_sorted else top[1]
    return t1, t2


def assign_domains(topic_freq: Counter, pair_counts: Dict[Tuple[str, str], int], t1: str, t2: str) -> Tuple[set, set]:
    if not t1:
        return set(), set()
    if not t2:
        return set(topic_freq.keys()), set()

    def co(a: str, b: str) -> int:
        if a > b:
            a, b = b, a
        return pair_counts.get((a, b), 0)

    d1, d2 = set([t1]), set([t2])
    for t in topic_freq.keys():
        if t in (t1, t2):
            continue
        c1, c2 = co(t, t1), co(t, t2)
        if c1 >= c2:
            d1.add(t)
        else:
            d2.add(t)
    return d1, d2


def compute_entropy(topic_freq: Counter, n_papers: int) -> float:
    """Normalized Shannon entropy of topic distribution across papers (0..1)."""
    if n_papers <= 0:
        return 0.0
    probs = [cnt / n_papers for _, cnt in topic_freq.items() if cnt > 0]
    if not probs:
        return 0.0
    h = -sum(p * math.log(p + 1e-12, 2) for p in probs)
    hmax = math.log(len(probs) + 1e-12, 2)
    if hmax <= 0:
        return 0.0
    return clamp01(h / hmax)


def compute_scores(
    papers: List[Dict[str, Any]],
    topic_freq: Counter,
    pair_counts: Dict[Tuple[str, str], int],
    gap_map: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(papers)
    years = [p["year"] for p in papers if p.get("year")]
    cited = [p["cited_by"] for p in papers]

    # topic entropy (breadth proxy)
    breadth = compute_entropy(topic_freq, n)

    # seed and assign two domains
    t1, t2 = pick_two_domains(topic_freq, pair_counts)
    d1, d2 = assign_domains(topic_freq, pair_counts, t1, t2)

    # bridge papers: contain topics from both domains
    bridge = []
    for p in papers:
        ts = set(p.get("topics") or [])
        if not ts:
            continue
        if (ts & d1) and (ts & d2):
            bridge.append(p)

    bridge_rate = (len(bridge) / n) if n else 0.0

    # PMI distribution across all observed pairs
    pmi_vals_all: List[float] = []
    for (a, b), obs in pair_counts.items():
        fa, fb = topic_freq.get(a, 0), topic_freq.get(b, 0)
        pmi_vals_all.append(pmi(obs, fa, fb, n))
    pmi_vals_all.sort()

    # Cross-domain PMI: for bridge papers compute median PMI across cross-domain topic pairs
    cross_pmis: List[float] = []
    for p in bridge:
        ts = p.get("topics") or []
        ts = ts[:8]
        left = [t for t in ts if t in d1]
        right = [t for t in ts if t in d2]
        if not left or not right:
            continue
        # all cross pairs for this paper
        vals = []
        for a in left:
            for b in right:
                aa, bb = (a, b) if a < b else (b, a)
                obs = pair_counts.get((aa, bb), 0)
                vals.append(pmi(obs, topic_freq.get(a, 0), topic_freq.get(b, 0), n))
        if vals:
            cross_pmis.append(sum(vals) / len(vals))

    cross_median_pmi = median(cross_pmis) if cross_pmis else 0.0
    cross_percentile = percentile_rank(pmi_vals_all, cross_median_pmi) if pmi_vals_all else 0.0

    # Novel combination score: lower PMI (rarer than expected) => higher score
    novel_score = to_score01(1.0 - cross_percentile)

    # Conventionality score: coherence (alignment with top topics) + foundation size + citation baseline
    top5 = [t for t, _ in topic_freq.most_common(5)]
    def core_cov(p: Dict[str, Any]) -> float:
        ts = p.get("topics") or []
        if not ts:
            return 0.0
        k = sum(1 for t in ts if t in top5)
        return k / len(ts)

    coherence = sum(core_cov(p) for p in papers) / n if n else 0.0
    foundation = clamp01(n / 300.0)  # Stage B defaults to ~300
    cited_med = median(cited) if cited else 0.0
    citation_norm = clamp01(math.log1p(cited_med) / math.log1p(2000.0))  # soft scale

    conventionality_score = to_score01(0.5 * coherence + 0.3 * foundation + 0.2 * citation_norm)

    # Evidence-quality penalty using gap_map topic mismatch (if present)
    topic_mismatch_fraction = None
    try:
        topic_mismatch_fraction = float((gap_map.get("overall") or {}).get("topic_mismatch_fraction"))
    except Exception:
        topic_mismatch_fraction = None

    evidence_quality = 1.0
    if topic_mismatch_fraction is not None:
        evidence_quality = clamp01(1.0 - topic_mismatch_fraction)

    # Bridge feasibility
    # - if bridge literature exists, and evidence_quality is good, feasibility is higher
    # - reward if any bridge paper is highly cited (top-10% within corpus)
    cited_sorted = sorted(cited)
    p90 = cited_sorted[int(0.9 * (len(cited_sorted) - 1))] if len(cited_sorted) >= 2 else (cited_sorted[0] if cited_sorted else 0)
    bridge_high = sum(1 for p in bridge if p.get("cited_by", 0) >= p90 and p90 > 0)
    bridge_high_bonus = clamp01(bridge_high / 5.0)  # saturate

    feasibility_score = to_score01(0.6 * bridge_rate + 0.3 * evidence_quality + 0.1 * bridge_high_bonus)

    # Gap opportunity: if many evidence gaps exist, there is a "hole" to fill (opportunity),
    # but only if mismatch is not extreme.
    gap_opportunity = 0.0
    try:
        ov = gap_map.get("overall") or {}
        n_claims = int(ov.get("n_claims") or 0)
        n_gaps = int(ov.get("n_evidence_gaps") or 0)
        if n_claims > 0:
            gap_opportunity = clamp01(n_gaps / n_claims)
    except Exception:
        gap_opportunity = 0.0

    gap_opportunity *= evidence_quality

    breakthrough_score = to_score01(0.5 * (novel_score / 100.0) + 0.3 * (feasibility_score / 100.0) + 0.2 * gap_opportunity)

    # Simple diagnostics
    diag = {
        "n_papers": n,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "cited_by_median": cited_med,
        "topic_entropy_norm": round(breadth, 4),
        "domain_seed_topics": [t1, t2],
        "bridge_papers": len(bridge),
        "bridge_rate": round(bridge_rate, 4),
        "cross_domain_median_pmi": round(cross_median_pmi, 4),
        "cross_domain_pmi_percentile": round(cross_percentile, 4),
        "topic_mismatch_fraction": topic_mismatch_fraction,
    }

    # Assemble top topics per domain
    d1_top = [t for t, _ in topic_freq.most_common(30) if t in d1][:12]
    d2_top = [t for t, _ in topic_freq.most_common(30) if t in d2][:12]

    # Bridge works (top by cited_by then recency)
    bridge_sorted = sorted(bridge, key=lambda p: (p.get("cited_by", 0), p.get("year", 0)), reverse=True)
    bridge_list = []
    for p in bridge_sorted[:15]:
        bridge_list.append({
            "doi": p.get("doi", ""),
            "title": p.get("title", ""),
            "year": p.get("year", None),
            "venue": p.get("venue", ""),
            "cited_by": p.get("cited_by", 0),
            "topics": (p.get("topics") or [])[:8],
        })

    return {
        "generated": _now(),
        "status": "ok",
        "scores": {
            "conventionality": conventionality_score,
            "novel_combination": novel_score,
            "bridge_feasibility": feasibility_score,
            "breakthrough": breakthrough_score,
        },
        "diagnostics": diag,
        "domains": {
            "domain_1_seed": t1,
            "domain_2_seed": t2,
            "domain_1_topics_top": d1_top,
            "domain_2_topics_top": d2_top,
        },
        "bridge_literature": bridge_list,
    }


def make_bridge_ideas_md(
    res: Dict[str, Any],
    structured: Dict[str, Any],
) -> str:
    scores = res.get("scores") or {}
    diag = res.get("diagnostics") or {}
    domains = res.get("domains") or {}
    bridges = res.get("bridge_literature") or []

    def s(x):
        return "?" if x is None else str(x)

    lines: List[str] = []
    lines.append("# Stage E — Новизна/прорывность и мосты в смежные области\n")
    lines.append(f"Generated: {res.get('generated')}\n")

    lines.append("\n## Оценки (0–100)\n")
    lines.append(f"- Конвенциональность: **{scores.get('conventionality','?')}**\n")
    lines.append(f"- Новая комбинация: **{scores.get('novel_combination','?')}**\n")
    lines.append(f"- Реализуемость моста: **{scores.get('bridge_feasibility','?')}**\n")
    lines.append(f"- Потенциал прорыва: **{scores.get('breakthrough','?')}**\n")

    lines.append("\n## Диагностика\n")
    lines.append(f"- Размер корпуса (публикаций): **{diag.get('n_papers','?')}**\n")
    lines.append(f"- Годы: {s(diag.get('year_min'))} – {s(diag.get('year_max'))}\n")
    lines.append(f"- Медиана цитирований (cited_by): **{diag.get('cited_by_median','?')}**\n")
    lines.append(f"- Ширина тематики (энтропия, 0–1): **{diag.get('topic_entropy_norm','?')}**\n")
    if diag.get("topic_mismatch_fraction") is not None:
        lines.append(f"- Доля off-topic (из Stage D): **{diag.get('topic_mismatch_fraction')}**\n")

    lines.append("\n## Разделение на 2 области (best-effort)\n")
    lines.append(f"Семенные темы: **{domains.get('domain_1_seed','')}**  vs  **{domains.get('domain_2_seed','')}**\n")
    lines.append("\n**Область 1 — топ-темы**\n")
    for t in (domains.get("domain_1_topics_top") or [])[:12]:
        lines.append(f"- {t}\n")
    lines.append("\n**Область 2 — топ-темы**\n")
    for t in (domains.get("domain_2_topics_top") or [])[:12]:
        lines.append(f"- {t}\n")

    lines.append("\n## Литература-мост (примеры)\n")
    if bridges:
        for b in bridges[:10]:
            doi = b.get("doi", "")
            year = b.get("year", "")
            cited = b.get("cited_by", 0)
            title = b.get("title", "(no title)")
            venue = b.get("venue", "")
            lines.append(f"- {title} ({year}) — {venue} — cited_by={cited} — {doi}\n")
    else:
        lines.append("No bridge papers detected in the current corpus (this may be OK, but lowers feasibility).\n")

    # Actionable ideas: ground them in the user's decisive tests if available
    decisive_tests = []
    try:
        decisive_tests = (structured.get("structured_idea") or {}).get("decisive_tests") or []
    except Exception:
        decisive_tests = []

    lines.append("\n## Bridge ideas (actionable, grounded)\n")
    if decisive_tests:
        lines.append("Below are *bridge-framed* versions of your decisive tests (so the bridge is explicit).\n")
        for i, t in enumerate(decisive_tests[:3], 1):
            name = (t or {}).get("test") or f"Test {i}"
            analysis = (t or {}).get("analysis") or ""
            data = (t or {}).get("data_needed") or ""
            lines.append(f"### Idea {i}: {name}\n")
            lines.append("**Bridge framing:** connect a *method-heavy* subfield to your *system-specific* subfield by making the validation cross-domain and cross-basin.\n")
            if data:
                lines.append(f"- Minimal data: {data}\n")
            if analysis:
                lines.append(f"- Analysis: {analysis}\n")
            lines.append("- Deliverable: 1 figure that explicitly shows *why* the bridge works (or fails) + 1 table of robustness checks.\n")
            lines.append("\n")
    else:
        lines.append("No decisive tests found in structured_idea.json; generating generic bridge actions.\n")

    # Generic, always safe: how to improve Stage B/C to get better novelty+bridge
    lines.append("### Practical upgrades that usually increase Stage E quality\n")
    lines.append("- If you see many off-topic sources in evidence (topic mismatch), re-run Stage B with a tighter query and/or add organism/system constraints.\n")
    lines.append("- Add 5–20 PDFs from Zotero to in\\pdf\\ (optional) and re-run Stage C: this improves evidence certainty and reduces indirectness.\n")
    lines.append("- Use Bridge literature list above to seed ResearchRabbit / Zotero collections for the two domains separately, then look for cross-links.\n")

    return "".join(lines)


# ------------------------- LLM prompt (optional) -------------------------

def build_llm_prompt(res_json: Dict[str, Any], structured: Dict[str, Any]) -> str:
    """A compact prompt to get higher-quality narrative + bridge ideas from ChatGPT/Gemini/etc."""
    # Keep it deterministic and short; user will paste.
    payload = {
        "scores": res_json.get("scores"),
        "domains": res_json.get("domains"),
        "bridge_literature": res_json.get("bridge_literature", [])[:12],
        "structured_idea": (structured.get("structured_idea") or {}),
        "notes": "Bridge ideas MUST be grounded in the bridge_literature items (cite DOI) and in decisive tests. Do not hallucinate references."
    }
    return (
        "You are a strict scientific assistant.\n"
        "Task: Improve the cross-domain (Stage E) output: write a short, high-quality rationale for novelty/breakthrough,\n"
        "and propose 8 bridge ideas.\n\n"
        "Rules:\n"
        "- You MUST ground each bridge idea in at least 1 item from bridge_literature (cite DOI).\n"
        "- If evidence quality is low (topic mismatch), explicitly say so and suggest query refinements.\n"
        "- Do NOT invent papers, DOIs, or quotes.\n"
        "- Output ONLY valid JSON (no markdown).\n\n"
        "JSON input:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "Return JSON schema:\n"
        "{\n"
        "  'narrative': {'why_conventional': str, 'why_novel': str, 'key_risks': [str], 'how_to_raise_score_fast': [str]},\n"
        "  'bridge_ideas': [\n"
        "    {'title': str, 'bridge': str, 'minimal_data': str, 'analysis': str, 'decisive_test_link': str, 'grounding_dois': [str]}\n"
        "  ]\n"
        "}\n"
    )


def merge_llm(res: Dict[str, Any], llm_json: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_json:
        return res
    res2 = dict(res)
    res2["llm"] = llm_json
    return res2


# ------------------------- stubs -------------------------

def write_stub(out_json: Path, out_md: Path, reason: str) -> None:
    ensure_dir(out_json.parent)
    stub = {
        "generated": _now(),
        "status": "not_ready",
        "reason": reason,
        "scores": {
            "conventionality": 0,
            "novel_combination": 0,
            "bridge_feasibility": 0,
            "breakthrough": 0,
        },
    }
    out_json.write_text(json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        "# Stage E — Новизна/прорывность и мосты в смежные области\n\n"
        f"Generated: {_now()}\n\n"
        "## НЕ ГОТОВО\n\n"
        f"Причина: {reason}\n\n"
        "Что делать:\n"
        "- Сначала запустите Stage B, чтобы появился out\\corpus.csv\n"
        "- (Опционально) Запустите Stage C и D, чтобы улучшить диагностику качества доказательств\n"
        "- Затем запустите RUN_E.bat ещё раз\n",
        encoding="utf-8",
    )


# ------------------------- main -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Idea folder (contains out/corpus.csv)")
    ap.add_argument("--ask-llm", action="store_true", help="Generate llm prompt and exit with code 2")
    args = ap.parse_args()

    idea = Path(args.idea)
    out_dir = idea / "out"
    in_dir = idea / "in"
    logs_dir = idea / "logs"
    ensure_dir(out_dir)
    ensure_dir(in_dir)
    ensure_dir(logs_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"moduleE_{stamp}.log"

    out_json = out_dir / "novelty_breakthrough.json"
    out_md = out_dir / "bridge_ideas.md"
    prompt_path = out_dir / "llm_prompt_E.txt"

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, "[INFO] Module E started")
        log_line(log_fp, f"[DIAG] idea={idea}")

        corpus_csv = out_dir / "corpus.csv"
        if not corpus_csv.exists():
            write_stub(out_json, out_md, "missing out/corpus.csv")
            log_line(log_fp, "[WARN] corpus.csv missing -> wrote NOT READY stub")
            return 0

        rows = read_csv_best_effort(corpus_csv)
        if not rows:
            write_stub(out_json, out_md, "could not parse corpus.csv (empty or unreadable)")
            log_line(log_fp, "[WARN] corpus.csv unreadable -> wrote NOT READY stub")
            return 0

        papers, topic_freq = build_topic_sets(rows)
        pair_counts = cooccurrence_counts(papers)

        structured = deep_repair(safe_read_json(out_dir / "structured_idea.json"))
        gap_map = deep_repair(safe_read_json(out_dir / "gap_map.json"))

        res = compute_scores(papers, topic_freq, pair_counts, gap_map)

        # LLM integration (optional)
        llm_path = in_dir / "llm_stageE.json"
        llm_raw = safe_read_text(llm_path).strip() if llm_path.exists() else ""
        llm_json = {}
        if llm_raw:
            llm_json = safe_read_json(llm_path)
            if not llm_json:
                log_line(log_fp, f"[WARN] llm_stageE.json exists but is empty/invalid JSON: {llm_path}")
                log_line(log_fp, f"[WARN] Please paste ONLY JSON (no extra text).")
        if llm_json:
            res = merge_llm(res, llm_json)
            log_line(log_fp, f"[OK] llm_stageE.json loaded and merged: {llm_path}")

        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text(make_bridge_ideas_md(res, structured), encoding="utf-8")
        log_line(log_fp, f"[OK] novelty_breakthrough.json -> {out_json}")
        log_line(log_fp, f"[OK] bridge_ideas.md -> {out_md}")

        # if user asked to run LLM refinement as a next step
        if args.ask_llm:
            # If LLM JSON already provided, we are done (do NOT loop forever).
            if llm_json:
                log_line(log_fp, "[OK] AskLLM mode: llm_stageE.json is present -> finished.")
                return 0

            prompt = build_llm_prompt(res, structured)
            prompt_path.write_text(prompt, encoding="utf-8-sig")
            log_line(log_fp, f"[NEXT] Wrote LLM prompt: {prompt_path}")

            # prepare target file (so Notepad has something to open)
            if not llm_path.exists():
                llm_path.write_text("", encoding="utf-8")

            # signal launcher to copy prompt to clipboard + open target file
            return 2

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
