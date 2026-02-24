#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module G — Better Ideas Generator (grounded, not "fantasy") — UNIVERSAL

Goal (from spec):
- Generate 5–10 improved ideas that score better on:
  - Evidence grounding (clear claims, relevant anchors)
  - Publishability (minimum publishable unit)
  - Breakthrough potential (testable bridge / decisive tests)
- Ideas must be *grounded* in the collected corpus / evidence artifacts.

Inputs (per idea folder)
- REQUIRED (best effort; module does NOT crash if missing):
  - idea.txt (raw)
- OPTIONAL (improves quality):
  - out/structured_idea.json      (Stage A)
  - out/corpus.csv               (Stage B)
  - out/field_map.md             (Stage B)
  - out/evidence_table.csv       (Stage C)
  - out/gap_map.json             (Stage D)
  - out/novelty_breakthrough.json (Stage E)
  - out/scores.json              (Stage F)

Optional LLM step (copy/paste; no API):
- If --ask-llm is used and in/llm_stageG.json is missing/invalid,
  we generate out/llm_prompt_G.txt and exit with code 2.
- Paste ONLY JSON into ideas/.../in/llm_stageG.json and re-run.

Outputs (per idea folder)
- out/better_ideas.md
- out/better_ideas.json   (machine-readable, for later stages)

Design constraints
- Robust: never crash; always writes a meaningful output stub.
- Universal: no domain-specific hardcoding; works for any topic.
- Minimal deps: stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------- low-level safety -------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_text(p: Path, encodings: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251")) -> str:
    if not p.exists():
        return ""
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


def read_csv_best_effort(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        return []
    raw = safe_read_text(p)
    if not raw.strip():
        return []
    # try common delimiters
    for delimiter in [",", ";", "\t"]:
        try:
            rows: List[Dict[str, str]] = []
            reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
            if not reader.fieldnames:
                continue
            for r in reader:
                rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
            # accept if at least 1 row and at least 2 columns
            if rows and len(reader.fieldnames) >= 2:
                return rows
        except Exception:
            continue
    return []


def safe_write_text(p: Path, s: str) -> None:
    try:
        p.write_text(s, encoding="utf-8")
    except Exception:
        try:
            p.write_bytes(s.encode("utf-8", errors="replace"))
        except Exception:
            pass


def safe_write_json(p: Path, obj: Any) -> None:
    try:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        safe_write_text(p, "{}")


def log_line(fp, msg: str) -> None:
    try:
        ts = _now()
        fp.write(f"{ts} {msg}\n")
        fp.flush()
    except Exception:
        pass


# ------------------------- extraction helpers -------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def split_keywords(x: Any) -> List[str]:
    out: List[str] = []
    if isinstance(x, list):
        for t in x:
            if isinstance(t, str):
                out.append(t)
    elif isinstance(x, str):
        out += re.split(r"[,\n;]+", x)
    # cleanup
    cleaned = []
    for t in out:
        t = normalize_ws(t)
        if not t:
            continue
        if len(t) < 3:
            continue
        cleaned.append(t)
    # de-dup, preserve order
    seen = set()
    uniq = []
    for t in cleaned:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def extract_named_entities(text: str, max_n: int = 12) -> List[str]:
    """
    Very lightweight, language-agnostic-ish:
    - Cyrillic Capitalized tokens: Иртыш, Волга...
    - Latin Capitalized tokens: Siberia, Phoxinus...
    - Keep unique, ignore common stop-words.
    """
    if not text:
        return []
    stop = set([
        "The", "And", "Or", "A", "An", "In", "On", "At", "For", "To",
        "Это", "И", "Или", "А", "Но", "В", "На", "По", "Для", "К", "С", "Из",
        "Module", "Stage", "H1", "H2", "H3"
    ])
    # sequences of 1-3 capitalized tokens (works for both Cyrillic and Latin)
    cand = re.findall(r"\b[А-ЯЁA-Z][\w\-]{2,}(?:\s+[А-ЯЁA-Z][\w\-]{2,}){0,2}\b", text)
    uniq = []
    seen = set()
    for c in cand:
        c = normalize_ws(c)
        if c in stop:
            continue
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        # discard if mostly digits
        if re.fullmatch(r"[0-9\-\._]+", c):
            continue
        uniq.append(c)
        if len(uniq) >= max_n:
            break
    return uniq


def corpus_anchor_candidates(corpus_rows: List[Dict[str, str]], keywords: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
    if not corpus_rows:
        return []
    kw = [k.lower() for k in keywords if k]
    out: List[Tuple[float, Dict[str, Any]]] = []
    for r in corpus_rows:
        title = r.get("title", "") or ""
        abstract = r.get("abstract", "") or ""
        topics = r.get("top_topics", "") or ""
        concepts = r.get("top_concepts", "") or ""
        blob = f"{title} {abstract} {topics} {concepts}".lower()
        m = 0
        hit = []
        for k in kw:
            if k and k in blob:
                m += 1
                if len(hit) < 6:
                    hit.append(k)
        cited_by = 0
        try:
            cited_by = int(float((r.get("cited_by") or "0").strip()))
        except Exception:
            cited_by = 0
        score = (m * 12.0) + math.log1p(max(0, cited_by)) * 3.0
        doi = (r.get("doi") or "").strip()
        if not doi:
            # allow without DOI but downweight
            score -= 3.0
        item = {
            "doi": doi,
            "title": normalize_ws(title)[:240],
            "year": str(r.get("year") or "").strip(),
            "venue": normalize_ws(r.get("venue",""))[:120],
            "cited_by": cited_by,
            "match_keywords": hit,
        }
        out.append((score, item))
    out.sort(key=lambda t: t[0], reverse=True)
    # de-dup by DOI+title
    seen = set()
    uniq = []
    for _, it in out:
        key = (it.get("doi","").lower(), it.get("title","").lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= top_n:
            break
    return uniq


def evidence_snapshot(evidence_rows: List[Dict[str, str]], max_rows: int = 12) -> Dict[str, Any]:
    """
    Create a compact summary and a tiny sample of rows (for prompting / debugging).
    """
    snap: Dict[str, Any] = {
        "rows_total": len(evidence_rows),
        "by_claim": {},
        "sample_rows": [],
    }
    by_claim: Dict[str, Dict[str, Any]] = {}
    for r in evidence_rows:
        cid = (r.get("claim_id") or "").strip() or "?"
        rel = (r.get("relation") or "").strip() or "unclear"
        cert = (r.get("certainty") or "").strip() or "Low"
        by_claim.setdefault(cid, {"n": 0, "relation": {}, "certainty": {}, "example_dois": []})
        d = by_claim[cid]
        d["n"] += 1
        d["relation"][rel] = d["relation"].get(rel, 0) + 1
        d["certainty"][cert] = d["certainty"].get(cert, 0) + 1
        doi = (r.get("doi") or "").strip()
        if doi and len(d["example_dois"]) < 3:
            d["example_dois"].append(doi)
    snap["by_claim"] = by_claim
    for r in evidence_rows[:max_rows]:
        snap["sample_rows"].append({
            "claim_id": (r.get("claim_id") or "").strip(),
            "doi": (r.get("doi") or "").strip(),
            "title": normalize_ws(r.get("title",""))[:180],
            "relation": (r.get("relation") or "").strip(),
            "certainty": (r.get("certainty") or "").strip(),
            "quote_location": (r.get("quote_location") or "").strip(),
        })
    return snap


def is_valid_llm_payload(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    ideas = obj.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        return False
    # minimal schema check
    good = 0
    for it in ideas[:3]:
        if isinstance(it, dict) and (it.get("title") or it.get("one_liner")):
            good += 1
    return good >= 1


# ------------------------- prompt builder -------------------------

def build_llm_prompt(
    idea_text: str,
    structured: Dict[str, Any],
    gap: Dict[str, Any],
    scores: Dict[str, Any],
    novelty: Dict[str, Any],
    anchors: List[Dict[str, Any]],
    ev_snap: Dict[str, Any],
    k_ideas: int,
) -> str:
    si = structured.get("structured_idea") if isinstance(structured, dict) else None
    si = si if isinstance(si, dict) else {}
    # compact structured fields
    compact = {
        "problem": si.get("problem", ""),
        "why_it_matters": si.get("why_it_matters", ""),
        "main_hypothesis": si.get("main_hypothesis", ""),
        "alternative_hypotheses": (si.get("alternative_hypotheses") or [])[:5],
        "key_predictions": (si.get("key_predictions") or [])[:8],
        "decisive_tests": (si.get("decisive_tests") or [])[:8],
        "minimal_publishable_unit": si.get("minimal_publishable_unit", ""),
        "adjacent_fields_to_scan": (si.get("adjacent_fields_to_scan") or [])[:8],
        "keywords_for_search": (si.get("keywords_for_search") or [])[:20],
    }

    # top gaps
    top_unknowns = gap.get("top_decisive_unknowns") or []
    top_unknowns = top_unknowns[:5] if isinstance(top_unknowns, list) else []

    # scores compact
    s_pub = ((scores.get("scores") or {}).get("publishability") or {}).get("score")
    s_brk = ((scores.get("scores") or {}).get("breakthrough") or {}).get("score")
    tmm = (((scores.get("metrics") or {}).get("evidence") or {}).get("topic_mismatch_fraction"))

    prompt = f"""You are an evidence-first research strategist.

TASK
Generate {k_ideas} better research ideas/variants that are BETTER than the current one by:
- stronger evidence grounding (explicit anchors, no hallucinated citations)
- clearer decisive tests (strong inference: main + >=2 alternatives + crucial tests)
- improved publishability and/or breakthrough potential (explicitly say which)

GROUNDING RULES (STRICT)
1) You MUST cite only from the provided "ANCHOR LITERATURE LIST" (DOI+title+year) or from the "EVIDENCE TABLE SNAPSHOT" DOIs.
2) Do NOT invent DOIs, titles, or facts. If evidence is missing or off-topic, say so and propose how to get decisive evidence.
3) Keep everything universal (no domain assumptions beyond the provided text).

OUTPUT FORMAT
Return ONLY valid JSON (no markdown). Schema:

LANGUAGE (IMPORTANT)
- Пиши все свободные текстовые поля НА РУССКОМ: title, one_liner, what_changes_vs_current, core_question, hypotheses, tests, планы, rationale, risk_flags, actions и т.д.
- НЕ придумывай и НЕ переводить DOI/год.
- Для anchor_literature[].title: оставляй заголовок строго как в предоставленном списке (НЕ переводить).
- Поле tag должно быть РОВНО одним из: publishability|breakthrough|bridge|gap-closure|method|review.


{{
  "meta": {{
    "generated_by": "LLM",
    "k": {k_ideas},
    "notes": "short"
  }},
  "ideas": [
    {{
      "id": "G1",
      "title": "short",
      "tag": "publishability|breakthrough|bridge|gap-closure|method|review",
      "one_liner": "1 sentence",
      "what_changes_vs_current": ["bullet", "..."],
      "core_question": "string",
      "main_hypothesis": "string",
      "alternatives": ["A1", "A2", "A3"],
      "decisive_tests": ["test 1", "test 2"],
      "minimal_data_plan": ["data requirement 1", "data requirement 2"],
      "analysis_plan": ["analysis step 1", "analysis step 2"],
      "why_breakthrough": "string",
      "why_publishable": "string",
      "anchor_literature": [
        {{"doi":"...","title":"...","year":"...","why_relevant":"..."}}
      ],
      "risk_flags": ["confounder", "power", "bias", "..."],
      "quick_48h_actions": ["action 1", "action 2"]
    }}
  ],
  "selection_rationale": {{
    "overall": "how this set improves scores",
    "coverage": "how ideas cover publishability vs breakthrough"
  }}
}}

CONTEXT (RAW IDEA TEXT, may be rough)
{idea_text.strip()[:2500]}

STRUCTURED IDEA (compact)
{json.dumps(compact, ensure_ascii=False, indent=2)}

GAP MAP (top decisive unknowns)
{json.dumps(top_unknowns, ensure_ascii=False, indent=2)}

CURRENT SCORES (if available)
- publishability_score: {s_pub}
- breakthrough_score: {s_brk}
- topic_mismatch_fraction: {tmm}

EVIDENCE TABLE SNAPSHOT (diagnostics + sample rows)
{json.dumps(ev_snap, ensure_ascii=False, indent=2)}

ANCHOR LITERATURE LIST (use ONLY these as citations)
{json.dumps(anchors[:25], ensure_ascii=False, indent=2)}

IMPORTANT
- Each idea must include >=3 anchors in "anchor_literature".
- If you propose "breakthrough", make the bridge TESTABLE (what new measurement/analysis closes the gap?).
- Prefer ideas that can succeed even if results are null (publishable null).
"""
    return prompt


# ------------------------- heuristic idea generator -------------------------

def choose_anchors_for_idea(anchors: List[Dict[str, Any]], prefer_kw: List[str], n: int = 5) -> List[Dict[str, Any]]:
    if not anchors:
        return []
    prefer = [k.lower() for k in prefer_kw if k]
    scored = []
    for a in anchors:
        hits = [h for h in (a.get("match_keywords") or []) if isinstance(h, str)]
        # prefer anchors that hit preferred keywords
        pref_hit = 0
        for p in prefer:
            if p and p in " ".join(hits).lower():
                pref_hit += 1
        score = pref_hit * 5 + math.log1p(int(a.get("cited_by") or 0))
        scored.append((score, a))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _, a in scored[:n]:
        out.append({
            "doi": a.get("doi",""),
            "title": a.get("title",""),
            "year": a.get("year",""),
            "why_relevant": f"Matches keywords: {', '.join(a.get('match_keywords') or [])}" if (a.get("match_keywords")) else "High-cited / relevant in collected corpus"
        })
    return out


def heuristic_generate(
    idea_text: str,
    structured: Dict[str, Any],
    gap: Dict[str, Any],
    scores: Dict[str, Any],
    novelty: Dict[str, Any],
    anchors: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    si = (structured.get("structured_idea") or {}) if isinstance(structured, dict) else {}
    main_h = normalize_ws(si.get("main_hypothesis",""))
    alts = si.get("alternative_hypotheses") or []
    alts = [normalize_ws(a) for a in alts if isinstance(a, str) and normalize_ws(a)]
    preds = si.get("key_predictions") or []
    preds = [normalize_ws(a) for a in preds if isinstance(a, str) and normalize_ws(a)]
    tests = si.get("decisive_tests") or []
    tests = [normalize_ws(a) for a in tests if isinstance(a, str) and normalize_ws(a)]
    mpu = normalize_ws(si.get("minimal_publishable_unit",""))
    kw = split_keywords(si.get("keywords_for_search") or [])
    # add gap keywords
    kw += split_keywords(gap.get("keywords_for_search") or [])
    # de-dup
    kw = split_keywords(kw)

    entities = extract_named_entities(" ".join([idea_text, main_h] + alts)[:8000])

    # diagnostics flags
    tmm = None
    try:
        tmm = (((scores.get("metrics") or {}).get("evidence") or {}).get("topic_mismatch_fraction"))
    except Exception:
        tmm = None
    bridge = None
    try:
        bridge = (((scores.get("metrics") or {}).get("novelty_breakthrough") or {}).get("bridge_feasibility"))
    except Exception:
        bridge = None

    focus_place = entities[:3]
    focus_str = ", ".join(focus_place) if focus_place else "target system(s)"

    # Build a small pool of templates, then cut to k.
    templates: List[Dict[str, Any]] = []

    # 1) MPU-first publishability
    templates.append({
        "id": "G1",
        "title": "MPU-first: one clean decisive test with prereg + sensitivity analysis",
        "tag": "publishability",
        "one_liner": f"Shrink scope to a minimum publishable unit in {focus_str}, lock decisions early, and publish even a null result.",
        "what_changes_vs_current": [
            "Narrow scope to 1 decisive test + 1 figure + 1 table (publishable even if null).",
            "Add prereg/analysis plan and sensitivity checks to reduce 'garden of forking paths'.",
            "Replace weak evidence grounding with a targeted anchor set (>=10 direct works).",
        ],
        "core_question": "Can we obtain a robust, decisive result for ONE key prediction, under a locked analysis plan?",
        "main_hypothesis": main_h or "Main hypothesis (from structured idea) — refine into a single testable statement.",
        "alternatives": (alts[:3] or ["Alternative 1 (plausible confounder)", "Alternative 2 (neutral explanation)", "Alternative 3 (method artifact)"]),
        "decisive_tests": (tests[:2] or ["Define a single crucial comparison and an acceptance threshold before data analysis.", "Run a negative control / permutation to falsify spurious signals."]),
        "minimal_data_plan": [
            "Define the smallest dataset that still tests the key prediction (n sites, n samples, minimal metadata).",
            "Collect or reuse a comparable dataset (same pipeline, same filtering).",
            "Add 1 independent validation axis (replicate subset / second split / hold-out).",
        ],
        "analysis_plan": [
            "Lock preprocessing + QC rules; store as config for reproducibility.",
            "Run primary analysis + 2 sensitivity variants (key hyperparameters).",
            "Report effect sizes + uncertainty; include null interpretation.",
        ],
        "why_breakthrough": "Low to moderate — not aiming for novelty, aiming for a clean, defensible result.",
        "why_publishable": "High — narrow MPU, prereg logic, and a result that is publishable even if negative.",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["underpowered if MPU too small", "confounding structure", "measurement mismatch"],
        "quick_48h_actions": ["Rewrite the MPU as 1 figure + 1 table + 1 paragraph claim.", "Rerun Stage B with tighter keywords to reduce topic mismatch.", "List 10 'must-cite' anchors from corpus for the MPU claim."]
    })

    # 2) Gap-closure idea (highest leverage unknown)
    top_unknowns = gap.get("top_decisive_unknowns") or []
    unk = None
    if isinstance(top_unknowns, list) and top_unknowns:
        # allow both string and dict
        if isinstance(top_unknowns[0], dict):
            val = top_unknowns[0].get("missing") or top_unknowns[0].get("missing_decisive_info") or ""
            if isinstance(val, list):
                val = "; ".join([normalize_ws(str(x)) for x in val if x])
            elif not isinstance(val, str):
                val = str(val)
            unk = normalize_ws(val)
            if not unk:
                unk = normalize_ws(top_unknowns[0].get("claim",""))
        elif isinstance(top_unknowns[0], str):
            unk = normalize_ws(top_unknowns[0])
    templates.append({
        "id": "G2",
        "title": "Gap-closure: design the single study that removes the #1 decisive uncertainty",
        "tag": "gap-closure",
        "one_liner": f"Turn the top gap into a targeted data/design task so the main hypothesis becomes testable.",
        "what_changes_vs_current": [
            "Promotes the top decisive unknown to the primary objective.",
            "Defines a crisp success/failure exam (Heilmeier-style).",
            "Improves evidence grounding by adding direct sources and/or direct measurements.",
        ],
        "core_question": f"What minimal design/data would decisively resolve the top uncertainty: {unk or 'top missing decisive info'}?",
        "main_hypothesis": main_h or "Main hypothesis — only if the decisive unknown is resolved.",
        "alternatives": (alts[:3] or ["Alternative 1: gap persists due to indirectness", "Alternative 2: gap due to wrong field map", "Alternative 3: gap due to measurement noise"]),
        "decisive_tests": [
            "Define a binary exam: what observation would clearly favor H1 vs alternatives?",
            "Add one diagnostic negative control that targets the alternative explanation."
        ],
        "minimal_data_plan": [
            "Collect only what resolves the gap (e.g., one additional data type or a better-matched sample frame).",
            "Ensure comparability (same protocol, same filters, same scales).",
            "If evidence is off-topic: rerun literature scout with system-specific keywords + exclusions.",
        ],
        "analysis_plan": [
            "Predefine decision rule(s) (thresholds, model comparison, or predictive performance).",
            "Run robustness checks that specifically attack the top confounder.",
        ],
        "why_breakthrough": "Moderate — closing a decisive gap can unlock stronger claims and a cleaner narrative.",
        "why_publishable": "High — the gap-closure itself is publishable (methods note / focused empirical result).",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["may require new data modality", "search strategy may still miss target field"],
        "quick_48h_actions": ["Copy top-5 decisive unknowns into a 1-page plan.", "Define exact 'exam' criteria (pass/fail).", "Add 5 exclusion keywords to Stage B to reduce off-topic."]
    })

    # 3) Bridge & breakthrough: make a testable bridge
    templates.append({
        "id": "G3",
        "title": "Bridge-to-breakthrough: make the cross-domain bridge testable (not a slogan)",
        "tag": "bridge",
        "one_liner": "Pick one adjacent field + one concrete measurable bridge, and test it against a strong null.",
        "what_changes_vs_current": [
            "Converts 'interdisciplinary' into a measurable bridge (data/method/analysis).",
            "Adds a decisive experiment or predictive test that would change practice if successful.",
            "Increases breakthrough chance by mixing conventional base + rare combination (Uzzi-style).",
        ],
        "core_question": "Which measurable bridge between domains changes the inference, not just the story?",
        "main_hypothesis": "A specific bridge mechanism improves explanation/prediction beyond domain-only baselines.",
        "alternatives": ["Bridge adds no predictive value (story-only).", "Bridge effect is confounded by structure/scale.", "Bridge works only in a subset (context-dependent)."],
        "decisive_tests": [
            "Out-of-sample test: domain-only baseline vs bridge-augmented model; accept only if improvement exceeds a pre-set delta.",
            "Ablation: remove bridge component → performance drops (causal contribution).",
        ],
        "minimal_data_plan": [
            "One additional measurable layer from adjacent field (a new variable class or measurement).",
            "A hold-out evaluation split (system A→B or time/space split).",
            "A negative control where bridge should NOT help.",
        ],
        "analysis_plan": [
            "Define baseline and bridge models; do cross-validation/transfer testing.",
            "Report effect sizes, uncertainty, and ablation results.",
        ],
        "why_breakthrough": "Higher — if bridge is real and generalizes, it can shift how the field interprets results.",
        "why_publishable": "Medium to high — even null bridge results are useful if designed as a decisive test.",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["bridge layer hard to obtain", "generalization may fail", "metric choice"],
        "quick_48h_actions": ["Select ONE bridge variable/method from adjacent fields.", "Define baseline vs bridge models + evaluation metric.", "Identify 3 anchor papers that already hint at a bridge."]
    })

    # 4) Transferability as prediction (method paper)
    templates.append({
        "id": "G4",
        "title": "Transferability as prediction: a portable evaluation protocol (method/perspective)",
        "tag": "method",
        "one_liner": "Reframe the core claim into an out-of-sample prediction problem and publish a reusable protocol + benchmark.",
        "what_changes_vs_current": [
            "Moves from 'list overlap' to explicit predictive transfer.",
            "Provides a reusable benchmarking artifact (dataset + pipeline + metrics).",
            "Produces a publishable deliverable even before full data collection (protocol + pilot).",
        ],
        "core_question": "How transferable are signals/models across datasets when evaluated strictly out-of-sample?",
        "main_hypothesis": "Models that survive strict out-of-sample evaluation represent robust signal; others are likely confounded/noise.",
        "alternatives": ["Transferability is dominated by hidden confounders.", "Transferability depends on scale alignment rather than biology.", "Apparent transfer is leakage/overfitting."],
        "decisive_tests": [
            "Train on dataset A, test on dataset B (no leakage).",
            "Permutation / label-shuffle negative control to estimate false transfer rate.",
        ],
        "minimal_data_plan": [
            "Two comparable datasets (or one dataset split by space/time).",
            "A fixed metric set (replication rate, sign agreement, predictive R2/AUC, calibration).",
            "A sensitivity grid for key preprocessing/controls.",
        ],
        "analysis_plan": [
            "Implement protocol; run benchmark; publish code + prereg decision rules.",
            "Report failure modes and recommended safeguards.",
        ],
        "why_breakthrough": "Moderate — can change how the field claims 'replication' by enforcing predictive tests.",
        "why_publishable": "High — method/protocol papers and benchmarks are widely publishable if transparent.",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["needs truly comparable datasets", "metric disagreement", "leakage risk"],
        "quick_48h_actions": ["Write a 1-page protocol (inputs, outputs, metrics).", "Pick evaluation splits and negative controls.", "Select 2–3 datasets/contexts to benchmark."]
    })

    # 5) Mechanistic validation (increase breakthrough)
    templates.append({
        "id": "G5",
        "title": "Mechanistic validation: add an independent axis beyond correlation",
        "tag": "breakthrough",
        "one_liner": "Add one independent validation axis (phenotype, experiment, external measurement) to turn correlations into mechanisms.",
        "what_changes_vs_current": [
            "Adds causal/mechanistic evidence, raising breakthrough credibility.",
            "Reduces risk of 'just correlations' criticism.",
            "Creates a second publishable output (validation dataset).",
        ],
        "core_question": "Do the proposed signals correspond to an independent measurable effect consistent with mechanism?",
        "main_hypothesis": "Signals predict an independent measurable response aligned with the hypothesized mechanism.",
        "alternatives": ["Signals reflect demographic confounding only.", "Phenotype is plastic/unrelated.", "Effect is context-specific and non-generalizable."],
        "decisive_tests": [
            "Independent validation dataset: does prediction hold under controlled conditions / across contexts?",
            "Dose–response or gradient test consistent with mechanism (predefined direction).",
        ],
        "minimal_data_plan": [
            "Pick ONE validation axis that is feasible (lab assay, field phenotype, proxy measurement).",
            "Small but well-controlled sample for validation (power > precision).",
            "Predefine expected direction and failure mode.",
        ],
        "analysis_plan": [
            "Link signal→validation axis with prereg model; include negative controls.",
            "Report effect sizes and uncertainty; check robustness to confounders.",
        ],
        "why_breakthrough": "Higher — mechanism/validation often makes the difference for high-impact framing.",
        "why_publishable": "Medium — validation can be publishable even if it falsifies the original narrative.",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["validation axis may be expensive", "effect may be subtle", "context dependence"],
        "quick_48h_actions": ["Choose one feasible validation axis.", "Write predicted direction + null criteria.", "Add 5 anchors on validation methods from corpus."]
    })

    # If we need more than 5, add a 'review' option
    templates.append({
        "id": "G6",
        "title": "Evidence-driven scoping mini-review: map contradictions + decisive tests",
        "tag": "review",
        "one_liner": "Convert the weakest part (evidence grounding) into a publishable scoping review with a decisive-test agenda.",
        "what_changes_vs_current": [
            "Produces a publishable artifact fast (PRISMA-lite scoping map).",
            "Locks the claim taxonomy and resolves topic mismatch.",
            "Creates an 'idea-to-tests' roadmap backed by cited anchors.",
        ],
        "core_question": "What does the best available literature really support/contradict, and what tests would decide?",
        "main_hypothesis": "A structured evidence map will reveal a small set of decisive unknowns that dominate uncertainty.",
        "alternatives": ["Field is too fragmented; evidence remains indirect.", "Apparent contradictions reflect scale mismatch.", "Literature bias distorts the map."],
        "decisive_tests": ["Predefine claim taxonomy; code evidence as supports/contradicts/unclear.", "Check robustness by excluding off-topic domains and re-scoring."],
        "minimal_data_plan": ["Use corpus.csv + targeted manual additions (10–20 must-have anchors).", "Extract key claims + evidence coding rules.", "Create gap map + test agenda."],
        "analysis_plan": ["PRISMA-lite log; reproducible inclusion/exclusion; coded evidence table; bias flags."],
        "why_breakthrough": "Low to moderate — breakthrough comes later, but this de-risks and strengthens future claims.",
        "why_publishable": "High — scoping review + evidence table + test agenda is publishable in many venues.",
        "anchor_literature": choose_anchors_for_idea(anchors, kw, n=5),
        "risk_flags": ["review may be seen as incremental", "needs good topic gate"],
        "quick_48h_actions": ["Tighten Stage B topic gate; rerun B→C→D.", "Manually add 10 cornerstone papers (DOIs).", "Write inclusion/exclusion criteria in 1 page."]
    })

    # Cut/pad to requested k
    out = templates[:max(3, min(k, len(templates)))]
    return out


# ------------------------- rendering -------------------------

def render_md(meta: Dict[str, Any], ideas: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Улучшенные варианты идеи (Stage G)\n")
    lines.append(f"Generated: **{meta.get('generated','')}**\n")
    lines.append("This file contains **improved research idea variants** grounded in your Stage B/C corpus/evidence (best effort).\n")
    if meta.get("diagnostics"):
        d = meta["diagnostics"]
        lines.append("## Diagnostics (what this module saw)\n")
        for k, v in d.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    if meta.get("note"):
        lines.append(f"> {meta['note']}\n")

    lines.append("## Ideas\n")
    for it in ideas:
        lines.append(f"### {it.get('id','G?')}. {it.get('title','(untitled)')}")
        if it.get("tag"):
            lines.append(f"- Tag: **{it.get('tag')}**")
        if it.get("one_liner"):
            lines.append(f"- One-liner: {it.get('one_liner')}")
        if it.get("what_changes_vs_current"):
            lines.append("- What changes vs current:")
            for b in it["what_changes_vs_current"]:
                lines.append(f"  - {b}")
        lines.append("")
        if it.get("core_question"):
            lines.append("**Core question**")
            lines.append(f"- {it['core_question']}\n")
        if it.get("main_hypothesis"):
            lines.append("**Main hypothesis**")
            lines.append(f"- {it['main_hypothesis']}\n")
        if it.get("alternatives"):
            lines.append("**Alternatives (>=2)**")
            for a in it["alternatives"]:
                lines.append(f"- {a}")
            lines.append("")
        if it.get("decisive_tests"):
            lines.append("**Decisive tests (crucial experiments / analyses)**")
            for t in it["decisive_tests"]:
                lines.append(f"- {t}")
            lines.append("")
        if it.get("minimal_data_plan"):
            lines.append("**Minimal data plan**")
            for p in it["minimal_data_plan"]:
                lines.append(f"- {p}")
            lines.append("")
        if it.get("analysis_plan"):
            lines.append("**Analysis plan**")
            for p in it["analysis_plan"]:
                lines.append(f"- {p}")
            lines.append("")
        if it.get("why_breakthrough") or it.get("why_publishable"):
            lines.append("**Why it can be breakthrough / publishable**")
            if it.get("why_breakthrough"):
                lines.append(f"- Breakthrough: {it['why_breakthrough']}")
            if it.get("why_publishable"):
                lines.append(f"- Publishable: {it['why_publishable']}")
            lines.append("")
        if it.get("anchor_literature"):
            lines.append("**Опорная литература (только из вашего корпуса; без выдуманных DOI)**")
            for a in it["anchor_literature"]:
                doi = a.get("doi","").strip()
                title = a.get("title","").strip()
                year = a.get("year","").strip()
                why = a.get("why_relevant","").strip()
                if doi:
                    lines.append(f"- {doi} — {title} ({year}) — {why}")
                else:
                    lines.append(f"- {title} ({year}) — {why}")
            lines.append("")
        if it.get("risk_flags"):
            lines.append("**Риски/флаги**")
            for r in it["risk_flags"]:
                lines.append(f"- {r}")
            lines.append("")
        if it.get("quick_48h_actions"):
            lines.append("**Быстрые действия на 48 часов**")
            for a in it["quick_48h_actions"]:
                lines.append(f"- {a}")
            lines.append("")
        lines.append("---\n")
    return "\n".join(lines).strip() + "\n"


def write_stub(out_md: Path, out_json: Path, reason: str) -> None:
    meta = {"generated": _now(), "status": "not_ready", "reason": reason}
    md = f"""# Улучшенные варианты идеи (Stage G)

Сгенерировано: **{meta['generated']}**

Статус: **НЕ ГОТОВО**

Причина: {reason}

Что можно сделать
- Прогнать предыдущие стадии (A–F) для этой идеи, чтобы появились:
  - out/structured_idea.json
  - out/corpus.csv
  - out/evidence_table.csv
  - out/gap_map.json
  - out/scores.json
- Затем запустить Stage G ещё раз.

"""
    safe_write_text(out_md, md)
    safe_write_json(out_json, {"meta": meta, "ideas": []})


# ------------------------- main -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Path to ideas/IDEA-... folder")
    ap.add_argument("--ask-llm", action="store_true", help="Prepare prompt for ChatGPT/Gemini and exit with code 2 if response missing")
    ap.add_argument("-k", type=int, default=8, help="How many ideas to generate (LLM prompt or heuristic templates)")
    args = ap.parse_args()

    idea = Path(args.idea)
    if not idea.exists() or not idea.is_dir():
        print(f"[FATAL] idea folder not found: {idea}", file=sys.stderr)
        return 1

    out_dir = idea / "out"
    in_dir = idea / "in"
    logs_dir = idea / "logs"
    ensure_dir(out_dir)
    ensure_dir(in_dir)
    ensure_dir(logs_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"moduleG_{stamp}.log"

    out_md = out_dir / "better_ideas.md"
    out_json = out_dir / "better_ideas.json"
    prompt_path = out_dir / "llm_prompt_G.txt"
    llm_path = in_dir / "llm_stageG.json"

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_line(log_fp, "[INFO] Module G started")
        log_line(log_fp, f"[DIAG] idea={idea}")
        log_line(log_fp, f"[DIAG] ask_llm={args.ask_llm} k={args.k}")

        idea_text = safe_read_text(idea / "idea.txt")
        structured = safe_read_json(out_dir / "structured_idea.json")
        gap = safe_read_json(out_dir / "gap_map.json")
        scores = safe_read_json(out_dir / "scores.json")
        novelty = safe_read_json(out_dir / "novelty_breakthrough.json")
        corpus_rows = read_csv_best_effort(out_dir / "corpus.csv")
        evidence_rows = read_csv_best_effort(out_dir / "evidence_table.csv")
        ev_snap = evidence_snapshot(evidence_rows)

        # Keywords for anchor selection / prompting
        si = (structured.get("structured_idea") or {}) if isinstance(structured, dict) else {}
        kw = split_keywords(si.get("keywords_for_search") or [])
        kw += split_keywords(gap.get("keywords_for_search") or [])
        kw = split_keywords(kw)
        if not kw:
            # fallback: from raw text
            kw = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{3,}", idea_text)][:15]

        anchors = corpus_anchor_candidates(corpus_rows, kw, top_n=30)

        # Diagnostics for header
        diag = {
            "structured_idea": bool(si),
            "corpus_rows": len(corpus_rows),
            "evidence_rows": len(evidence_rows),
            "gap_map": bool(gap),
            "anchors_found": len(anchors),
        }
        try:
            diag["publishability_score"] = ((scores.get("scores") or {}).get("publishability") or {}).get("score")
            diag["breakthrough_score"] = ((scores.get("scores") or {}).get("breakthrough") or {}).get("score")
            diag["topic_mismatch_fraction"] = (((scores.get("metrics") or {}).get("evidence") or {}).get("topic_mismatch_fraction"))
        except Exception:
            pass

        # If LLM mode requested, ensure payload exists/valid
        if args.ask_llm:
            llm = safe_read_json(llm_path)
            if not is_valid_llm_payload(llm):
                prompt = build_llm_prompt(
                    idea_text=idea_text or "(empty idea.txt)",
                    structured=structured,
                    gap=gap,
                    scores=scores,
                    novelty=novelty,
                    anchors=anchors,
                    ev_snap=ev_snap,
                    k_ideas=max(5, min(10, int(args.k))),
                )
                safe_write_text(prompt_path, prompt)
                log_line(log_fp, f"[NEXT] LLM response missing/invalid. Wrote prompt: {prompt_path}")
                log_line(log_fp, f"[NEXT] Paste JSON reply into: {llm_path}")
                # Also write a minimal stub so user sees something
                meta = {"generated": _now(), "status": "need_llm", "diagnostics": diag}
                safe_write_text(out_md, render_md(meta, heuristic_generate(idea_text, structured, gap, scores, novelty, anchors, k=6)))
                safe_write_json(out_json, {"meta": meta, "ideas": []})
                return 2

            ideas = llm.get("ideas") or []
            if not isinstance(ideas, list):
                ideas = []
            meta = llm.get("meta") if isinstance(llm.get("meta"), dict) else {}
            meta2 = {"generated": _now(), "status": "ok_llm", "diagnostics": diag}
            # merge meta
            meta2.update({k: v for k, v in meta.items() if k not in meta2})
            safe_write_text(out_md, render_md(meta2, ideas))
            safe_write_json(out_json, {"meta": meta2, "ideas": ideas})
            log_line(log_fp, f"[OK] Wrote: {out_md}")
            return 0

        # Autonomous mode (heuristics)
        if not (idea_text or si or corpus_rows):
            write_stub(out_md, out_json, "missing idea.txt and no structured/corpus artifacts found")
            log_line(log_fp, "[WARN] Not enough inputs. Wrote stub.")
            return 0

        ideas = heuristic_generate(idea_text, structured, gap, scores, novelty, anchors, k=max(5, min(10, int(args.k))))
        meta = {"generated": _now(), "status": "ok_heuristic", "diagnostics": diag, "note": "Heuristic mode: for best quality, rerun with -AskLLM (ChatGPT/Gemini/Claude) to ground and refine."}
        safe_write_text(out_md, render_md(meta, ideas))
        safe_write_json(out_json, {"meta": meta, "ideas": ideas})
        log_line(log_fp, f"[OK] Wrote: {out_md}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
