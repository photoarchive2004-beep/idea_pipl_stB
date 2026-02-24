#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module C (Stage C) — Evidence Table Engine

Goal (from spec): claims → sources → support/contradict/unclear + certainty.

Design constraints:
- No paid APIs. ChatGPT use is via copy/paste only.
- Robust: never "silent" fail; always log; exit code 2 means "need LLM JSON".

Inputs (per idea folder):
- out/structured_idea.json  (from Module A)
- out/corpus.csv           (from Module B)
- optional in/pdf/*.pdf    (user-supplied PDFs; best effort)

Outputs (per idea folder):
- out/evidence_candidates.json
- out/llm_prompt_C.txt               (generated when LLM is needed)
- in/llm_evidence.json               (placeholder if missing)
- out/evidence_table.csv
- out/evidence_summary.md

Exit codes:
- 0: success
- 1: fatal error
- 2: waiting for ChatGPT JSON (in/llm_evidence.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    m = re.search(r"{[\s\S]*}", s)
    if not m:
        raise ValueError("No JSON object found")
    return json.loads(m.group(0))


def is_placeholder(obj: dict) -> bool:
    keys = set(obj.keys())
    if keys.issubset({"paste", "note", "reason_pipeline_waiting"}):
        return True
    if "paste" in obj and "Paste ChatGPT JSON" in str(obj.get("paste", "")):
        return True
    return False


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    # keep latin + cyrillic words
    toks = re.findall(r"[a-zа-яё][a-zа-яё0-9\-]{1,}", text, flags=re.IGNORECASE)
    stop = {
        # EN
        "the","and","for","with","from","into","about","this","that","those","these",
        "have","has","had","were","was","are","is","be","been","being","study","studies",
        "result","results","method","methods","data","analysis","analyses","using","use","used",
        "based","across","between","within","among","via","via",
        # RU (very small)
        "это","как","что","для","при","без","или","и","а","но","на","в","во","по","из","к",
        "у","о","об","про","над","под","между","с","со","же","ли","не","да","нет",
    }
    out = []
    for t in toks:
        t = t.strip("-")
        if len(t) < 3:
            continue
        if t in stop:
            continue
        out.append(t)
    return out


def snippet_around(text: str, tokens: List[str], max_len: int = 260) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if not tokens:
        return t[:max_len]
    low = t.lower()
    idxs = []
    for tok in tokens[:8]:
        i = low.find(tok.lower())
        if i >= 0:
            idxs.append(i)
    if not idxs:
        return t[:max_len]
    i0 = min(idxs)
    start = max(0, i0 - max_len // 2)
    end = min(len(t), start + max_len)
    sn = t[start:end]
    if start > 0:
        sn = "…" + sn
    if end < len(t):
        sn = sn + "…"
    return sn


def load_structured_idea(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    # Module A stores full object with meta/structured_idea
    if isinstance(obj, dict) and "structured_idea" in obj:
        return obj
    # allow raw structured_idea dict
    return {"meta": {"language": "ru"}, "structured_idea": obj}


def extract_claims(obj: Dict[str, Any], max_n: int) -> List[str]:
    si = obj.get("structured_idea") or {}
    claims: List[str] = []

    # Preferred explicit field (future-proof)
    c0 = si.get("claims_to_verify")
    if isinstance(c0, list):
        claims.extend([str(x).strip() for x in c0 if str(x).strip()])

    # Backward-compatible for current Module A
    kp = si.get("key_predictions")
    if isinstance(kp, list):
        claims.extend([str(x).strip() for x in kp if str(x).strip()])

    # If still empty, derive from hypotheses / problem
    if not claims:
        mh = si.get("main_hypothesis")
        if mh:
            claims.append(str(mh).strip())
        alts = si.get("alternative_hypotheses")
        if isinstance(alts, list):
            claims.extend([str(x).strip() for x in alts if str(x).strip()])
        pr = si.get("problem")
        if pr:
            claims.append(str(pr).strip())

    # De-dup (preserve order)
    seen = set()
    uniq = []
    for c in claims:
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            continue
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        uniq.append(c)
    return uniq[:max_n]


@dataclass
class Paper:
    openalex_id: str
    doi: str
    title: str
    year: str
    venue: str
    cited_by: int
    abstract: str

    @property
    def key(self) -> str:
        return self.doi or self.openalex_id or self.title


def load_corpus(path: Path) -> List[Paper]:
    rows: List[Paper] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            doi = (row.get("doi") or "").strip()
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            cb = 0
            try:
                cb = int(float((row.get("cited_by") or "0").strip() or 0))
            except Exception:
                cb = 0
            rows.append(
                Paper(
                    openalex_id=(row.get("openalex_id") or "").strip(),
                    doi=doi,
                    title=(row.get("title") or "").strip(),
                    year=(row.get("year") or "").strip(),
                    venue=(row.get("venue") or "").strip(),
                    cited_by=cb,
                    abstract=(row.get("abstract") or "").strip(),
                )
            )
    return rows


def build_idf(papers: List[Paper]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    n = len(papers)
    for p in papers:
        toks = set(tokenize((p.title or "") + " " + (p.abstract or "")))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((n + 1) / (d + 1)) + 1.0
    return idf


def rank_papers_for_claim(claim: str, papers: List[Paper], idf: Dict[str, float]) -> List[Tuple[float, Paper]]:
    ct = tokenize(claim)
    if not ct:
        return []
    ct_set = set(ct)

    ranked: List[Tuple[float, Paper]] = []
    for p in papers:
        txt_title = p.title or ""
        txt_abs = p.abstract or ""
        toks_title = set(tokenize(txt_title))
        toks_abs = set(tokenize(txt_abs))
        inter_t = ct_set & toks_title
        inter_a = ct_set & toks_abs
        if not inter_t and not inter_a:
            continue
        score = 0.0
        for t in inter_a:
            score += idf.get(t, 1.0)
        for t in inter_t:
            score += 1.6 * idf.get(t, 1.0)
        # small boost for citations (log scale)
        if p.cited_by > 0:
            score *= (1.0 + min(0.45, math.log10(p.cited_by + 1) / 5.0))
        ranked.append((score, p))
    ranked.sort(key=lambda sp: sp[0], reverse=True)
    return ranked


def validate_llm(obj: Dict[str, Any]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    if not isinstance(obj, dict):
        return False, "Response is not a JSON object.", []
    if is_placeholder(obj):
        return False, "Placeholder JSON (paste ChatGPT output).", []

    rows = obj.get("evidence_rows")
    if isinstance(rows, list):
        flat = [x for x in rows if isinstance(x, dict)]
        if not flat:
            return False, "evidence_rows is empty.", []
        return True, "OK", flat

    # Accept alternative shape: evidence -> [{claim_id, claim, sources:[...]}]
    ev = obj.get("evidence")
    if isinstance(ev, list):
        flat_rows: List[Dict[str, Any]] = []
        for e in ev:
            if not isinstance(e, dict):
                continue
            cid = e.get("claim_id")
            for s in (e.get("sources") or []):
                if isinstance(s, dict):
                    r = dict(s)
                    r["claim_id"] = cid
                    flat_rows.append(r)
        if flat_rows:
            return True, "OK", flat_rows
        return False, "Unsupported evidence format.", []

    return False, "Missing evidence_rows.", []


def write_evidence_table_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "claim_id","claim",
        "source_id","doi","title","year","venue","openalex_id",
        "relation","quote","quote_location",
        "certainty","certainty_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_summary_md(path: Path, claims: List[str], table_rows: List[Dict[str, Any]]) -> None:
    by_claim: Dict[int, List[Dict[str, Any]]] = {}
    for r in table_rows:
        try:
            cid = int(str(r.get("claim_id", "")).strip())
        except Exception:
            continue
        by_claim.setdefault(cid, []).append(r)

    lines: List[str] = []
    lines.append("# Сводка доказательств (Stage C)\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for i, c in enumerate(claims, 1):
        rows = by_claim.get(i, [])
        sup = sum(1 for r in rows if str(r.get("relation","")).lower() == "supports")
        con = sum(1 for r in rows if str(r.get("relation","")).lower() == "contradicts")
        unc = sum(1 for r in rows if str(r.get("relation","")).lower() == "unclear")
        lines.append(f"## Утверждение {i}\n")
        lines.append(c + "\n")
        lines.append(f"- sources: {len(rows)} (supports={sup}, contradicts={con}, unclear={unc})\n")

        # Evidence gap rule from spec
        gap = "NO"
        if len(rows) < 3:
            gap = "YES (fewer than 3 sources)"
        elif sup == 0 and con == 0:
            gap = "YES (all unclear / indirect)"
        lines.append(f"- evidence_gap: {gap}\n")

        # show up to 5 best sources (supports first, then contradicts)
        def pick(rel: str) -> List[Dict[str, Any]]:
            return [r for r in rows if str(r.get("relation","")).lower() == rel]
        picked = pick("supports")[:3] + pick("contradicts")[:2]
        if picked:
            lines.append("**Key sources:**\n")
            for r in picked:
                title = r.get("title") or "(no title)"
                doi = r.get("doi") or ""
                yr = r.get("year") or ""
                rel = r.get("relation") or ""
                cert = r.get("certainty") or ""
                lines.append(f"- {rel} / {cert} — {title} ({yr}) {doi}\n")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Idea folder containing out/structured_idea.json and out/corpus.csv")
    ap.add_argument("--claims-max", type=int, default=8, help="Max number of claims to process")
    ap.add_argument("--k", type=int, default=6, help="Sources per claim")
    ap.add_argument("--no-llm", action="store_true", help="Do not require ChatGPT; fill relation/unclear automatically")
    ap.add_argument("--prompt-max-abs", type=int, default=1200, help="Max abstract length passed to ChatGPT")
    args = ap.parse_args()

    idea_dir = Path(args.idea)
    in_dir, out_dir, logs_dir = idea_dir / "in", idea_dir / "out", idea_dir / "logs"
    ensure_dir(in_dir); ensure_dir(out_dir); ensure_dir(logs_dir)

    log_path = logs_dir / f"moduleC_{now_stamp()}.log"
    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        structured_path = out_dir / "structured_idea.json"
        corpus_path = out_dir / "corpus.csv"
        if not structured_path.exists():
            log("[ERROR] Missing input: out/structured_idea.json")
            return 1
        if not corpus_path.exists():
            log("[ERROR] Missing input: out/corpus.csv")
            return 1

        st = load_structured_idea(structured_path)
        claims = extract_claims(st, max_n=max(1, int(args.claims_max)))
        if not claims:
            log("[ERROR] Could not extract claims from structured_idea.json")
            return 1

        papers = load_corpus(corpus_path)
        if not papers:
            log("[ERROR] corpus.csv is empty")
            return 1

        idf = build_idf(papers)
        candidates: List[Dict[str, Any]] = []
        for i, claim in enumerate(claims, 1):
            ranked = rank_papers_for_claim(claim, papers, idf)
            top: List[Dict[str, Any]] = []
            used = set()
            for score, p in ranked:
                if len(top) >= int(args.k):
                    break
                if p.key in used:
                    continue
                used.add(p.key)
                ctoks = tokenize(claim)
                abs_trunc = (p.abstract or "")
                abs_trunc = re.sub(r"\s+", " ", abs_trunc).strip()
                if len(abs_trunc) > int(args.prompt_max_abs):
                    abs_trunc = abs_trunc[: int(args.prompt_max_abs)] + "…"
                top.append({
                    "source_id": f"S{len(top)+1}",
                    "score": round(float(score), 4),
                    "openalex_id": p.openalex_id,
                    "doi": p.doi,
                    "title": p.title,
                    "year": p.year,
                    "venue": p.venue,
                    "cited_by": p.cited_by,
                    "abstract": abs_trunc,
                    "auto_snippet": snippet_around(p.abstract or "", ctoks, max_len=260),
                })
            candidates.append({
                "claim_id": i,
                "claim": claim,
                "sources": top,
            })

        (out_dir / "evidence_candidates.json").write_text(
            json.dumps({"meta": {"generated": now_stamp()}, "candidates": candidates}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- no-LLM path: best-effort
        if args.no_llm:
            rows: List[Dict[str, Any]] = []
            for c in candidates:
                for s in c.get("sources") or []:
                    rows.append({
                        "claim_id": c.get("claim_id"),
                        "claim": c.get("claim"),
                        "source_id": s.get("source_id"),
                        "doi": s.get("doi"),
                        "title": s.get("title"),
                        "year": s.get("year"),
                        "venue": s.get("venue"),
                        "openalex_id": s.get("openalex_id"),
                        "relation": "unclear",
                        "quote": s.get("auto_snippet") or (s.get("abstract") or "")[:260],
                        "quote_location": "abstract (auto)",
                        "certainty": "Low",
                        "certainty_reason": "Auto (no LLM): abstract-only candidate matching",
                    })
            write_evidence_table_csv(out_dir / "evidence_table.csv", rows)
            write_summary_md(out_dir / "evidence_summary.md", claims, rows)
            log("[OK] Module C complete (no-llm mode).")
            return 0

        # --- LLM path
        llm_path = in_dir / "llm_evidence.json"
        prompt_path = out_dir / "llm_prompt_C.txt"

        if not llm_path.exists():
            llm_path.write_text(
                '{\n  "paste": "Paste ChatGPT JSON here (replace this file)",\n'
                '  "note": "Run RUN_C.bat again after saving valid JSON"\n}\n',
                encoding="utf-8",
            )

        if llm_path.exists():
            raw = llm_path.read_text(encoding="utf-8", errors="ignore")
            try:
                obj = extract_json(raw)
            except Exception:
                obj = {"paste": "Paste ChatGPT JSON here (replace this file)", "note": "Run RUN_C.bat again"}

            ok, why, flat_rows = validate_llm(obj if isinstance(obj, dict) else {})
            if not ok:
                # generate prompt and ask user
                tpl = (Path(__file__).resolve().parents[1] / "config" / "prompts" / "llm_moduleC_prompt.txt")
                if not tpl.exists():
                    log("[ERROR] Prompt template missing: " + str(tpl))
                    return 1
                packet = json.dumps({"claims": candidates}, ensure_ascii=False, indent=2)
                prompt = tpl.read_text(encoding="utf-8", errors="ignore").replace("{{CANDIDATES_JSON}}", packet)
                prompt_path.write_text(prompt, encoding="utf-8")
                placeholder = {
                    "paste": "Paste ChatGPT JSON here (replace this file)",
                    "note": "Run RUN_C.bat again after saving valid JSON",
                    "reason_pipeline_waiting": why,
                }
                llm_path.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
                log("[NEED] LLM evidence JSON missing/invalid: " + why)
                log("[NEED] Prompt generated: out/llm_prompt_C.txt")
                return 2

            # Merge LLM rows with candidates metadata
            cand_by_claim: Dict[int, Dict[str, Dict[str, Any]]] = {}
            claim_text: Dict[int, str] = {}
            for c in candidates:
                cid = int(c.get("claim_id"))
                claim_text[cid] = str(c.get("claim"))
                m: Dict[str, Dict[str, Any]] = {}
                for s in (c.get("sources") or []):
                    sid = str(s.get("source_id"))
                    m[sid] = s
                cand_by_claim[cid] = m

            out_rows: List[Dict[str, Any]] = []
            seen_pairs = set()
            for r in flat_rows:
                try:
                    cid = int(str(r.get("claim_id", "")).strip())
                except Exception:
                    continue
                sid = str(r.get("source_id") or r.get("sid") or "").strip()
                if not sid:
                    continue
                pair = (cid, sid)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                meta = (cand_by_claim.get(cid) or {}).get(sid)
                if not meta:
                    # ignore invented sources
                    continue

                rel = str(r.get("relation") or "unclear").strip().lower()
                if rel not in ("supports", "contradicts", "unclear"):
                    rel = "unclear"
                cert = str(r.get("certainty") or "Low").strip()
                if cert not in ("High", "Med", "Low"):
                    # accept lowercase too
                    m = cert.lower()
                    if m.startswith("h"):
                        cert = "High"
                    elif m.startswith("m"):
                        cert = "Med"
                    else:
                        cert = "Low"

                out_rows.append({
                    "claim_id": cid,
                    "claim": claim_text.get(cid, ""),
                    "source_id": sid,
                    "doi": meta.get("doi", ""),
                    "title": meta.get("title", ""),
                    "year": meta.get("year", ""),
                    "venue": meta.get("venue", ""),
                    "openalex_id": meta.get("openalex_id", ""),
                    "relation": rel,
                    "quote": (r.get("quote") or meta.get("auto_snippet") or "").strip(),
                    "quote_location": (r.get("quote_location") or "abstract").strip(),
                    "certainty": cert,
                    "certainty_reason": (r.get("certainty_reason") or "").strip(),
                })

            # If LLM returned nothing usable, fail back to prompt regeneration
            if not out_rows:
                tpl = (Path(__file__).resolve().parents[1] / "config" / "prompts" / "llm_moduleC_prompt.txt")
                packet = json.dumps({"claims": candidates}, ensure_ascii=False, indent=2)
                prompt = tpl.read_text(encoding="utf-8", errors="ignore").replace("{{CANDIDATES_JSON}}", packet)
                prompt_path.write_text(prompt, encoding="utf-8")
                placeholder = {
                    "paste": "Paste ChatGPT JSON here (replace this file)",
                    "note": "Run RUN_C.bat again after saving valid JSON",
                    "reason_pipeline_waiting": "LLM JSON parsed, but no valid rows matched candidates",
                }
                llm_path.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
                log("[NEED] LLM JSON had no rows matching candidate sources.")
                return 2

            write_evidence_table_csv(out_dir / "evidence_table.csv", out_rows)
            write_summary_md(out_dir / "evidence_summary.md", claims, out_rows)
            log("[OK] Module C complete.")
            return 0

        log("[ERROR] Unexpected state: llm_evidence.json missing")
        return 1

    except Exception as e:
        log("[ERROR] " + repr(e))
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
