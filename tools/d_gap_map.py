#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module D — Gap Map & Contradiction Finder

Goal (from spec):
- Find where evidence is truly unclear.
- Detect contradictions between sources.
- Surface "thin spots" in design and top missing decisive facts.

Design constraints:
- Robust: never crash; always write logs and an output stub when inputs missing.
- Universal: works for any idea folder with out/evidence_table.csv (from Module C).
- No extra dependencies (stdlib only).

Outputs (per idea folder):
- out/gap_map.md
- out/gap_map.json   (machine-readable)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_text(p: Path, encodings=("utf-8", "utf-8-sig", "cp1251")) -> str:
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    # last resort: binary decode with replacement
    try:
        return p.read_bytes().decode("utf-8", errors="replace")
    except Exception:
        return ""


def safe_read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(safe_read_text(p))
    except Exception:
        return {}


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
    """
    Read CSV with encoding/delimiter sniffing. Returns list of dict rows.
    Never raises (returns empty on failure).
    """
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

    # dialect sniff from first ~64kb
    sample = raw[:65536]
    dialect = sniff_dialect(sample)

    try:
        reader = csv.DictReader(raw.splitlines(), dialect=dialect)
        rows: List[Dict[str, str]] = []
        for r in reader:
            if r is None:
                continue
            # normalize keys
            rr = { (k or "").strip(): (v or "").strip() for k, v in r.items() }
            if any(rr.values()):
                rows.append(rr)
        return rows
    except Exception:
        # ultra-safe fallback: try comma
        try:
            reader = csv.DictReader(raw.splitlines(), delimiter=",")
            return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in reader if r]
        except Exception:
            return []


def norm_rel(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ("support", "supports", "supported"):
        return "supports"
    if x in ("contradict", "contradicts", "contradicted"):
        return "contradicts"
    if x in ("unclear", "unknown", "indirect", "mixed", ""):
        return "unclear"
    return x


def norm_cert(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ("high", "h"):
        return "High"
    if x in ("med", "medium", "moderate", "m"):
        return "Med"
    if x in ("low", "l", "very low", "very_low", "verylow"):
        return "Low"
    if not x:
        return ""
    # keep original-ish
    return x[:1].upper() + x[1:]


def is_topic_mismatch(reason: str) -> bool:
    r = (reason or "").lower()
    needles = [
        "topic mismatch", "mismatch", "off-topic", "off topic",
        "не ", "не ",  # harmless; we add more specific below
        "не ",         # keep as placeholder (no-op)
    ]
    # stricter russian hints:
    ru = ["не по теме", "несоответ", "другая система", "другой объект", "растение", "livestock", "plant", "bird"]
    if any(n in r for n in ["topic mismatch", "off-topic", "off topic", "mismatch"]):
        return True
    if any(n in r for n in ru):
        return True
    return False


def compress_reason(reason: str) -> str:
    """Short, stable label for recurring problems."""
    r = (reason or "").strip()
    if not r:
        return ""
    rl = r.lower()
    if "topic mismatch" in rl or "off-topic" in rl or "off topic" in rl or "несоответ" in rl or "не по теме" in rl:
        return "topic_mismatch"
    if "indirect" in rl or "indirectness" in rl or "косвен" in rl:
        return "indirectness"
    if "abstract" in rl or "только абстракт" in rl:
        return "abstract_only"
    if "inconsisten" in rl or "противореч" in rl:
        return "inconsistency"
    if "risk of bias" in rl or "bias" in rl or "системат" in rl:
        return "risk_of_bias"
    if "imprecision" in rl or "неточно" in rl or "широк" in rl:
        return "imprecision"
    if "publication bias" in rl:
        return "publication_bias"
    if "methods mismatch" in rl or "methods" in rl:
        return "methods_only"
    return "other"


def claim_keyword_hint(claim: str, keywords: List[str]) -> str:
    """
    Produce a non-hallucinatory hint for re-search:
    - pick up to 3 keywords that share tokens with claim.
    """
    if not keywords:
        return ""
    c = (claim or "").lower()
    scored: List[Tuple[int, str]] = []
    for kw in keywords:
        k = (kw or "").lower()
        # simple overlap score
        toks = [t for t in re_split_tokens(k) if len(t) >= 4]
        score = sum(1 for t in toks if t in c)
        if score > 0:
            scored.append((score, kw))
    scored.sort(reverse=True)
    best = [kw for _, kw in scored[:3]]
    if best:
        return "; ".join(best)
    return ""


def re_split_tokens(s: str) -> List[str]:
    import re
    return [t for t in re.split(r"[^0-9A-Za-zА-Яа-я_]+", s) if t]


def compute_gap_map(rows: List[Dict[str, str]], structured: Dict[str, Any]) -> Dict[str, Any]:
    by_claim: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        cid_raw = (r.get("claim_id") or "").strip()
        try:
            cid = int(cid_raw)
        except Exception:
            continue
        by_claim[cid].append(r)

    keywords = []
    try:
        keywords = structured.get("structured_idea", {}).get("keywords_for_search") or []
        if not isinstance(keywords, list):
            keywords = []
    except Exception:
        keywords = []

    claims_text: Dict[int, str] = {}
    for cid, rs in by_claim.items():
        # claim text should be identical per cid; take first non-empty
        for r in rs:
            c = (r.get("claim") or "").strip()
            if c:
                claims_text[cid] = c
                break

    # compute per-claim stats
    claim_items: List[Dict[str, Any]] = []
    for cid in sorted(by_claim.keys()):
        rs = by_claim[cid]
        rels = [norm_rel(r.get("relation","")) for r in rs]
        certs = [norm_cert(r.get("certainty","")) for r in rs]
        reasons = [r.get("certainty_reason","") for r in rs]

        c_rel = Counter(rels)
        c_cert = Counter([c for c in certs if c])

        topic_m = sum(1 for rr in reasons if is_topic_mismatch(rr))
        reason_labels = [compress_reason(rr) for rr in reasons if rr]
        c_reason = Counter([x for x in reason_labels if x])

        supports = c_rel.get("supports", 0)
        contradicts = c_rel.get("contradicts", 0)
        unclear = c_rel.get("unclear", 0)

        n = len(rs)

        contradiction = supports > 0 and contradicts > 0

        evidence_gap_reasons = []
        if n < 3:
            evidence_gap_reasons.append("fewer_than_3_sources")
        if supports == 0 and contradicts == 0:
            evidence_gap_reasons.append("all_unclear_or_indirect")
        if n > 0:
            low_frac = c_cert.get("Low", 0) / max(1, sum(c_cert.values()))
            if low_frac >= 0.8 and sum(c_cert.values()) >= 3:
                evidence_gap_reasons.append("mostly_low_certainty")
        if topic_m / max(1, n) >= 0.6 and n >= 3:
            evidence_gap_reasons.append("mostly_topic_mismatch")

        evidence_gap = len(evidence_gap_reasons) > 0

        # gap score for ranking missing facts
        score = 0
        if contradiction:
            score += 60
        if n < 3:
            score += 45
        if supports == 0 and contradicts == 0:
            score += 55
        if "mostly_low_certainty" in evidence_gap_reasons:
            score += 25
        if "mostly_topic_mismatch" in evidence_gap_reasons:
            score += 35

        # pick key sources (supports first, then contradicts, then unclear)
        def pick(rel: str, k: int) -> List[Dict[str, str]]:
            out = [r for r in rs if norm_rel(r.get("relation","")) == rel]
            return out[:k]
        picked = pick("supports", 3) + pick("contradicts", 2)
        if not picked:
            picked = pick("unclear", 5)

        top_reasons = [x for x, _ in c_reason.most_common(3)]
        hint = claim_keyword_hint(claims_text.get(cid, ""), keywords)

        claim_items.append({
            "claim_id": cid,
            "claim": claims_text.get(cid, ""),
            "n_sources": n,
            "counts_relation": dict(c_rel),
            "counts_certainty": dict(c_cert),
            "topic_mismatch_n": topic_m,
            "top_reason_labels": top_reasons,
            "evidence_gap": evidence_gap,
            "evidence_gap_reasons": evidence_gap_reasons,
            "contradiction": contradiction,
            "gap_score": score,
            "key_sources": [
                {
                    "relation": norm_rel(r.get("relation","")),
                    "certainty": norm_cert(r.get("certainty","")),
                    "title": r.get("title",""),
                    "year": r.get("year",""),
                    "doi": r.get("doi",""),
                    "source_id": r.get("source_id",""),
                } for r in picked
            ],
            "search_hint": hint,
        })

    # overall summary
    total_rows = len(rows)
    total_topic_m = sum(1 for r in rows if is_topic_mismatch(r.get("certainty_reason","")))
    overall = {
        "generated": _now(),
        "total_rows": total_rows,
        "topic_mismatch_rows": total_topic_m,
        "topic_mismatch_fraction": (total_topic_m / max(1, total_rows)),
        "n_claims": len(claim_items),
        "n_contradictions": sum(1 for c in claim_items if c["contradiction"]),
        "n_evidence_gaps": sum(1 for c in claim_items if c["evidence_gap"]),
    }

    # decisive unknowns: top-k by gap_score then by fewest supports
    ranked = sorted(
        claim_items,
        key=lambda c: (c.get("gap_score", 0), -c.get("evidence_gap", False), -c.get("contradiction", False)),
        reverse=True,
    )
    top_unknowns = []
    for c in ranked[:5]:
        # derive a conservative "missing fact" statement (no domain hallucination)
        missing = []
        if c["contradiction"]:
            missing.append("resolve_direction_of_effect (find higher-certainty / more direct evidence)")
        if "fewer_than_3_sources" in c["evidence_gap_reasons"]:
            missing.append("add_more_direct_sources (>=3)")
        if "all_unclear_or_indirect" in c["evidence_gap_reasons"]:
            missing.append("find_direct_evidence_in_target_system (not methods-only / not off-topic)")
        if "mostly_low_certainty" in c["evidence_gap_reasons"]:
            missing.append("increase_certainty (full-text, better design, closer match)")
        if "mostly_topic_mismatch" in c["evidence_gap_reasons"]:
            missing.append("tighten_search_gate (keywords + exclude other domains)")
        if not missing:
            missing.append("increase_decisiveness (seek stronger/closer evidence)")
        top_unknowns.append({
            "claim_id": c["claim_id"],
            "claim": c["claim"],
            "why_it_matters": "high_gap_score",
            "missing_decisive_info": missing,
            "search_hint": c.get("search_hint",""),
        })

    return {
        "overall": overall,
        "claims": claim_items,
        "top_decisive_unknowns": top_unknowns,
        "keywords_for_search": (structured.get("structured_idea", {}).get("keywords_for_search") or []),
    }


def render_gap_map_md(gap: Dict[str, Any]) -> str:
    o = gap.get("overall", {})
    lines: List[str] = []
    lines.append("# Карта пробелов и противоречий (Stage D)\n")
    lines.append(f"Generated: {o.get('generated','')}\n")
    lines.append(f"- claims: **{o.get('n_claims',0)}**\n")
    lines.append(f"- total evidence rows: **{o.get('total_rows',0)}**\n")
    frac = o.get("topic_mismatch_fraction", 0.0)
    lines.append(f"- topic-mismatch rows: **{o.get('topic_mismatch_rows',0)}** ({frac:.0%})\n")
    lines.append(f"- contradictions: **{o.get('n_contradictions',0)}**\n")
    lines.append(f"- evidence gaps: **{o.get('n_evidence_gaps',0)}**\n")

    # Systemic warnings
    if o.get("total_rows", 0) > 0 and frac >= 0.6:
        lines.append("\n## Системное предупреждение\n")
        lines.append("Большая доля источников помечена как **topic mismatch / off-topic**. ")
        lines.append("Это обычно означает, что Stage B собрал не то поле. ")
        lines.append("Решение: ужесточить topic gate и/или добавить ключевые слова из structured_idea.\n")

    # Top decisive unknowns
    lines.append("\n## ТОП-5 решающих неизвестных (наибольший рычаг)\n")
    for u in gap.get("top_decisive_unknowns", [])[:5]:
        cid = u.get("claim_id")
        claim = u.get("claim","")
        missing = u.get("missing_decisive_info", [])
        hint = u.get("search_hint","")
        lines.append(f"### Утверждение {cid}\n")
        lines.append(claim + "\n")
        lines.append("Не хватает решающей информации:\n")
        for m in missing:
            lines.append(f"- {m}\n")
        if hint:
            lines.append(f"- search_hint: {hint}\n")
        lines.append("")

    # Per-claim detail
    lines.append("\n## Карта по каждому утверждению\n")
    for c in gap.get("claims", []):
        cid = c.get("claim_id")
        claim = c.get("claim","")
        n = c.get("n_sources",0)
        rel = c.get("counts_relation",{})
        cert = c.get("counts_certainty",{})
        gap_yes = c.get("evidence_gap", False)
        contr = c.get("contradiction", False)

        lines.append(f"### Утверждение {cid}\n")
        lines.append(claim + "\n")
        lines.append(f"- sources: **{n}** (supports={rel.get('supports',0)}, contradicts={rel.get('contradicts',0)}, unclear={rel.get('unclear',0)})\n")
        if cert:
            lines.append(f"- certainty: {', '.join([f'{k}={v}' for k,v in cert.items()])}\n")
        lines.append(f"- contradiction: **{'YES' if contr else 'NO'}**\n")
        lines.append(f"- evidence_gap: **{'YES' if gap_yes else 'NO'}**\n")
        if c.get("evidence_gap_reasons"):
            lines.append(f"- gap_reasons: {', '.join(c['evidence_gap_reasons'])}\n")
        if c.get("top_reason_labels"):
            lines.append(f"- main_issues: {', '.join(c['top_reason_labels'])}\n")

        ks = c.get("key_sources", [])
        if ks:
            lines.append("\n**Ключевые источники:**\n")
            for s in ks:
                lines.append(f"- {s.get('relation','')} / {s.get('certainty','')} — {s.get('title','(без названия)')} ({s.get('year','')}) {s.get('doi','')}\n")
        lines.append("")

    return "".join(lines)


def write_stub(out_md: Path, out_json: Path, reason: str) -> None:
    ensure_dir(out_md.parent)
    out_md.write_text(
        "# Карта пробелов и противоречий (Stage D)\n\n"
        f"Generated: {_now()}\n\n"
        "## НЕ ГОТОВО\n\n"
        f"Причина: {reason}\n\n"
        "Что делать:\n"
        "- Сначала запустите Stage C, чтобы появился out\\evidence_table.csv\n"
        "- Затем запустите RUN_D.bat ещё раз\n",
        encoding="utf-8"
    )
    out_json.write_text(json.dumps({
        "overall": {"generated": _now(), "status": "not_ready", "reason": reason}
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Idea folder (contains out/evidence_table.csv)")
    ap.add_argument("--k", type=int, default=5, help="Top decisive unknowns (default 5)")
    args = ap.parse_args()

    idea = Path(args.idea)
    out_dir = idea / "out"
    ensure_dir(out_dir)

    evidence_csv = out_dir / "evidence_table.csv"
    out_md = out_dir / "gap_map.md"
    out_json = out_dir / "gap_map.json"

    if not evidence_csv.exists():
        write_stub(out_md, out_json, "missing out/evidence_table.csv")
        return 0

    structured = safe_read_json(out_dir / "structured_idea.json")

    rows = read_csv_best_effort(evidence_csv)
    if not rows:
        write_stub(out_md, out_json, "could not parse evidence_table.csv (empty or unreadable)")
        return 0

    gap = compute_gap_map(rows, structured)
    # respect --k (shrink list)
    gap["top_decisive_unknowns"] = (gap.get("top_decisive_unknowns") or [])[: max(1, int(args.k))]

    out_json.write_text(json.dumps(gap, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_gap_map_md(gap), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
