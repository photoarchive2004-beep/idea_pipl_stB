#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module F — Dual Scoring: Publishability vs Breakthrough + portfolio position (Universal)

Goal (from spec):
- Produce two independent scores:
    1) Publishability  (probability of producing a publishable result)
    2) Breakthrough    (chance of a paradigm-shift / high-impact leap)
- Place the idea into a simple risk × payoff portfolio matrix.

Design constraints:
- Robust: never crash; always writes output stubs when inputs are missing/invalid.
- Universal: no assumptions about a specific domain; only uses prior stage artifacts if present.
- Minimal deps: stdlib only.
- Optional ChatGPT step: when --ask-llm is used and "in/llm_stageF.json" is missing/invalid,
  we generate "out/llm_prompt_F.txt" and exit with code 2.
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
from typing import Any, Dict, List, Optional, Tuple


# ------------------------- utilities -------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_text(p: Path, encodings: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251")) -> str:
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


def safe_write_text(p: Path, s: str) -> None:
    try:
        p.write_text(s, encoding="utf-8")
    except Exception:
        try:
            p.write_bytes(s.encode("utf-8", errors="replace"))
        except Exception:
            pass


def safe_write_json(p: Path, obj: Any) -> None:
    safe_write_text(p, json.dumps(obj, ensure_ascii=False, indent=2))


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x != x:  # NaN
        return lo
    return max(lo, min(hi, x))


def to_int_score(x: float) -> int:
    return int(round(clamp(x, 0, 100)))


def short(s: str, n: int = 280) -> str:
    s2 = re.sub(r"\s+", " ", (s or "").strip())
    return s2[:n] + ("…" if len(s2) > n else "")


# ------------------------- IO loaders -------------------------

def load_structured(idea_out: Path) -> Dict[str, Any]:
    p = idea_out / "structured_idea.json"
    obj = safe_read_json(p)
    if isinstance(obj, dict) and "structured_idea" in obj and isinstance(obj["structured_idea"], dict):
        return obj["structured_idea"]
    return obj if isinstance(obj, dict) else {}


def load_gap_map(idea_out: Path) -> Dict[str, Any]:
    p = idea_out / "gap_map.json"
    obj = safe_read_json(p)
    return obj if isinstance(obj, dict) else {}


def load_novelty_breakthrough(idea_out: Path) -> Dict[str, Any]:
    p = idea_out / "novelty_breakthrough.json"
    obj = safe_read_json(p)
    return obj if isinstance(obj, dict) else {}


def load_corpus_metrics(idea_out: Path) -> Dict[str, Any]:
    p = idea_out / "corpus.csv"
    if not p.exists():
        return {"present": False}
    rows = 0
    years: List[int] = []
    cited_by: List[int] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows += 1
                y = (row.get("year") or "").strip()
                if y.isdigit():
                    years.append(int(y))
                cb = (row.get("cited_by") or "").strip()
                if cb.isdigit():
                    cited_by.append(int(cb))
    except Exception:
        return {"present": True, "rows": rows, "read_error": True}

    now_year = datetime.now().year
    recent5 = sum(1 for y in years if y >= now_year - 5) if years else 0
    recent10 = sum(1 for y in years if y >= now_year - 10) if years else 0

    return {
        "present": True,
        "rows": rows,
        "years_min": min(years) if years else None,
        "years_median": int(median(years)) if years else None,
        "years_max": max(years) if years else None,
        "recent5_fraction": (recent5 / rows) if rows else None,
        "recent10_fraction": (recent10 / rows) if rows else None,
        "cited_by_median": int(median(cited_by)) if cited_by else None,
    }


def load_evidence_metrics(idea_out: Path) -> Dict[str, Any]:
    p = idea_out / "evidence_table.csv"
    if not p.exists():
        return {"present": False}

    n_rows = 0
    claim_ids = set()
    rel = Counter()
    cert = Counter()
    topic_mismatch_hits = 0
    by_claim_sources = defaultdict(int)
    by_claim_rel = defaultdict(Counter)
    by_claim_cert = defaultdict(Counter)
    top_sources: List[Dict[str, Any]] = []

    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                n_rows += 1
                cid = (row.get("claim_id") or "").strip()
                try:
                    cid_i = int(cid)
                except Exception:
                    cid_i = None
                if cid_i is not None:
                    claim_ids.add(cid_i)
                    by_claim_sources[cid_i] += 1

                relation = (row.get("relation") or "unclear").strip().lower()
                if relation not in ("supports", "contradicts", "unclear"):
                    relation = "unclear"
                rel[relation] += 1
                if cid_i is not None:
                    by_claim_rel[cid_i][relation] += 1

                certainty = (row.get("certainty") or "Low").strip()
                if certainty not in ("High", "Med", "Low"):
                    certainty = "Low"
                cert[certainty] += 1
                if cid_i is not None:
                    by_claim_cert[cid_i][certainty] += 1

                reason = (row.get("certainty_reason") or "").lower()
                if "topic mismatch" in reason or "несоответ" in reason:
                    topic_mismatch_hits += 1

                # keep a compact list of "representative" sources (DOI + title + year)
                if len(top_sources) < 12:
                    doi = (row.get("doi") or "").strip()
                    title = (row.get("title") or "").strip()
                    year = (row.get("year") or "").strip()
                    if doi or title:
                        top_sources.append({"doi": doi, "title": short(title, 140), "year": year, "relation": relation, "certainty": certainty})
    except Exception:
        return {"present": True, "rows": n_rows, "read_error": True}

    n_claims = len(claim_ids)
    sources_per_claim = [by_claim_sources[c] for c in sorted(by_claim_sources)]
    sources_median = int(median(sources_per_claim)) if sources_per_claim else None
    sources_min = min(sources_per_claim) if sources_per_claim else None
    sources_max = max(sources_per_claim) if sources_per_claim else None

    def frac(k: str) -> Optional[float]:
        return (rel[k] / n_rows) if n_rows else None

    def cfrac(k: str) -> Optional[float]:
        return (cert[k] / n_rows) if n_rows else None

    return {
        "present": True,
        "rows": n_rows,
        "n_claims": n_claims,
        "relation_counts": dict(rel),
        "relation_fractions": {k: frac(k) for k in ("supports", "contradicts", "unclear")},
        "certainty_counts": dict(cert),
        "certainty_fractions": {k: cfrac(k) for k in ("High", "Med", "Low")},
        "topic_mismatch_hits": topic_mismatch_hits,
        "topic_mismatch_fraction_proxy": (topic_mismatch_hits / n_rows) if n_rows else None,
        "sources_per_claim_median": sources_median,
        "sources_per_claim_min": sources_min,
        "sources_per_claim_max": sources_max,
        "top_sources": top_sources,
    }


# ------------------------- scoring -------------------------

def score_design(structured: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    """Heuristic design-clarity score based on the Stage A schema."""
    notes: List[str] = []
    pts = 0.0
    comp = {}

    def present(k: str) -> bool:
        v = structured.get(k)
        if isinstance(v, str):
            return bool(v.strip())
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, dict):
            return len(v) > 0
        return v is not None

    # Core Heilmeier-like elements
    if present("problem"):
        pts += 15; comp["problem"] = 15
    else:
        notes.append("Нет явного 'problem' в structured_idea.json (стадия A).")

    if present("why_it_matters"):
        pts += 10; comp["why_it_matters"] = 10
    else:
        notes.append("Нет блока 'why_it_matters' (почему важно).")

    if present("state_of_practice_today"):
        pts += 10; comp["state_of_practice_today"] = 10
    else:
        notes.append("Нет блока 'state_of_practice_today' (как делают сейчас и ограничения).")

    if present("main_hypothesis"):
        pts += 15; comp["main_hypothesis"] = 15
    else:
        notes.append("Нет 'main_hypothesis'.")

    alts = structured.get("alternative_hypotheses")
    n_alt = len(alts) if isinstance(alts, list) else 0
    if n_alt >= 2:
        pts += 15; comp["alternative_hypotheses"] = 15
    elif n_alt == 1:
        pts += 8; comp["alternative_hypotheses"] = 8
        notes.append("Есть только 1 альтернативная гипотеза (лучше ≥2 для strong inference).")
    else:
        notes.append("Нет альтернативных гипотез (лучше ≥2).")

    preds = structured.get("key_predictions")
    n_pred = len(preds) if isinstance(preds, list) else 0
    if n_pred >= 3:
        pts += 10; comp["key_predictions"] = 10
    elif n_pred >= 1:
        pts += 5; comp["key_predictions"] = 5
    else:
        notes.append("Нет списка 'key_predictions'.")

    tests = structured.get("decisive_tests")
    n_tests = len(tests) if isinstance(tests, list) else 0
    if n_tests >= 1:
        pts += 15; comp["decisive_tests"] = 15
    else:
        notes.append("Нет 'decisive_tests' (решающих тестов).")

    if present("minimal_publishable_unit"):
        pts += 10; comp["minimal_publishable_unit"] = 10
    else:
        notes.append("Нет 'minimal_publishable_unit' (минимально публикуемый юнит).")

    kws = structured.get("keywords_for_search")
    n_kws = len(kws) if isinstance(kws, list) else 0
    if n_kws >= 8:
        pts += 10; comp["keywords_for_search"] = 10
    elif n_kws >= 4:
        pts += 6; comp["keywords_for_search"] = 6
    else:
        notes.append("Мало ключевых слов для поиска (keywords_for_search).")

    adj = structured.get("adjacent_fields_to_scan")
    n_adj = len(adj) if isinstance(adj, list) else 0
    if n_adj >= 3:
        pts += 5; comp["adjacent_fields_to_scan"] = 5

    return clamp(pts), comp, notes


def score_evidence(evm: Dict[str, Any], gapm: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    notes: List[str] = []
    if not evm.get("present"):
        return 0.0, {"missing": True}, ["Нет evidence_table.csv (стадия C)."]

    rows = evm.get("rows") or 0
    n_claims = evm.get("n_claims") or 0
    med_sources = evm.get("sources_per_claim_median") or 0

    cert_f = evm.get("certainty_fractions") or {}
    rel_f = evm.get("relation_fractions") or {}

    high = float(cert_f.get("High") or 0.0)
    med = float(cert_f.get("Med") or 0.0)
    low = float(cert_f.get("Low") or 0.0)

    unclear = float(rel_f.get("unclear") or 0.0)

    # Prefer gap_map "topic_mismatch_fraction" if present; else use proxy from certainty_reason
    tm_frac = None
    try:
        tm_frac = float(gapm.get("overall", {}).get("topic_mismatch_fraction"))
    except Exception:
        tm_frac = None
    if tm_frac is None:
        tm_frac = evm.get("topic_mismatch_fraction_proxy")
        try:
            tm_frac = float(tm_frac) if tm_frac is not None else None
        except Exception:
            tm_frac = None

    relevance = clamp(100.0 * (1.0 - (tm_frac if tm_frac is not None else 0.3)))
    if tm_frac is not None and tm_frac > 0.5:
        notes.append("Большая доля источников выглядит нерелевантной (topic mismatch) — нужно уточнить запросы Stage B.")

    # certainty score (High=1, Med=0.6, Low=0.2)
    certainty_score = clamp(100.0 * (high * 1.0 + med * 0.6 + low * 0.2))
    # coverage score: median sources per claim, saturate at 3
    coverage_score = clamp(100.0 * min(1.0, (med_sources / 3.0) if med_sources else 0.0))
    # clarity: too many "unclear" reduces grounding
    clarity_score = clamp(100.0 * (1.0 - unclear))

    total = 0.45 * certainty_score + 0.35 * coverage_score + 0.20 * relevance
    comp = {
        "relevance": to_int_score(relevance),
        "certainty_score": to_int_score(certainty_score),
        "coverage_score": to_int_score(coverage_score),
        "clarity_score": to_int_score(clarity_score),
        "rows": rows,
        "n_claims": n_claims,
        "sources_per_claim_median": med_sources,
        "topic_mismatch_fraction": tm_frac,
    }
    if rows < 10:
        notes.append("Очень мало строк в evidence_table.csv — возможно, Stage C прошла не полностью.")
    if n_claims == 0:
        notes.append("Не распознаны claims в evidence_table.csv (claim_id).")

    return clamp(total), comp, notes


def score_corpus(cm: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    notes: List[str] = []
    if not cm.get("present"):
        return 0.0, {"missing": True}, ["Нет corpus.csv (стадия B)."]

    rows = cm.get("rows") or 0
    recent10 = cm.get("recent10_fraction")
    recent10 = float(recent10) if isinstance(recent10, (int, float)) and recent10 is not None else None

    size_score = clamp(100.0 * min(1.0, rows / 300.0))  # 300 is your typical target
    recency_score = clamp(100.0 * (recent10 if recent10 is not None else 0.4))

    total = 0.7 * size_score + 0.3 * recency_score
    comp = {
        "rows": rows,
        "size_score": to_int_score(size_score),
        "recency_score": to_int_score(recency_score),
        "recent10_fraction": recent10,
        "years_median": cm.get("years_median"),
    }
    if rows < 150:
        notes.append("Корпус небольшой (<150). Для надежной карты поля лучше 200–500.")
    return clamp(total), comp, notes


def score_breakthrough(nb: Dict[str, Any], evidence_grounding: float) -> Tuple[float, Dict[str, Any], List[str]]:
    notes: List[str] = []
    scores = nb.get("scores") if isinstance(nb, dict) else None
    if not isinstance(scores, dict):
        # fallback: neutral
        base = 50.0
        notes.append("Нет novelty_breakthrough.json (стадия E) — breakthrough оценен грубо (50).")
        return base, {"missing": True}, notes

    conv = float(scores.get("conventionality") or 50.0)
    novel = float(scores.get("novel_combination") or 50.0)
    bridge = float(scores.get("bridge_feasibility") or 50.0)
    bt = float(scores.get("breakthrough") or 50.0)

    # Use Stage E breakthrough as base, but adjust with bridge/novelty and evidence grounding.
    # If evidence grounding is very low, cap breakthrough (can't claim a breakthrough without grounding).
    base = 0.65 * bt + 0.20 * novel + 0.15 * bridge

    if evidence_grounding < 30:
        base = min(base, 55.0)
        notes.append("Слабая доказательная база ограничивает заявляемый 'breakthrough' (сначала нужна релевантная литература/данные).")
    if bridge < 30:
        notes.append("Низкая bridge_feasibility: пока мало «моста» между доменами — это снижает шанс прорыва.")
    if novel > 70 and bridge < 30:
        notes.append("Комбинация выглядит необычной, но без моста это может быть «случайная новизна» (надо найти bridge-подходы).")

    comp = {
        "conventionality": to_int_score(conv),
        "novel_combination": to_int_score(novel),
        "bridge_feasibility": to_int_score(bridge),
        "breakthrough_stageE": to_int_score(bt),
        "breakthrough_combined": to_int_score(base),
    }
    return clamp(base), comp, notes


def portfolio_quadrant(pub: float, bt: float) -> Dict[str, Any]:
    # thresholds can be tuned later; keep deterministic now
    pub_hi = pub >= 60
    bt_hi = bt >= 60

    if pub_hi and bt_hi:
        q = "HighPayoff-HighPublishability"
        label = "High payoff × High publishability"
    elif pub_hi and not bt_hi:
        q = "LowPayoff-HighPublishability"
        label = "Low payoff × High publishability"
    elif (not pub_hi) and bt_hi:
        q = "HighPayoff-LowPublishability"
        label = "High payoff × Low publishability"
    else:
        q = "LowPayoff-LowPublishability"
        label = "Low payoff × Low publishability"

    return {"quadrant": q, "label": label, "thresholds": {"high": 60}}


def suggest_actions(structured: Dict[str, Any], gapm: Dict[str, Any], ev_notes: List[str], design_notes: List[str], corpus_notes: List[str]) -> Dict[str, List[str]]:
    """Generate deterministic 'next steps' lists. Keep them short and actionable."""
    a48: List[str] = []
    a2w: List[str] = []

    # Use top_decisive_unknowns when present
    tdu = gapm.get("top_decisive_unknowns")
    if isinstance(tdu, list) and tdu:
        for item in tdu[:5]:
            if isinstance(item, dict):
                txt = item.get("unknown") or item.get("gap") or item.get("text")
                if isinstance(txt, str) and txt.strip():
                    a48.append("Сформулировать и уточнить: " + short(txt, 180))
            elif isinstance(item, str):
                a48.append("Сформулировать и уточнить: " + short(item, 180))

    # If no decisive unknowns, fallback to missing_critical_info from meta (if exists in structured JSON)
    # structured_idea.json has meta separately; we don't load it here. But many times Stage A already includes critical missing info elsewhere.
    if not a48:
        a48.append("Уточнить 1–2 решающих теста (decisive_tests) и ожидания по данным/анализу.")
        a48.append("Сделать 2 поисковые стратегии Stage B: узкая (по системе/таксону) и методическая (по анализу).")

    # Evidence/design warnings
    for n in (ev_notes + design_notes + corpus_notes):
        if "topic mismatch" in n.lower():
            a48.append("Перегенерировать Stage B запросы: добавить таксон/регион/систему и исключающие термины, затем заново Stage C/D.")
            break

    # Longer horizon
    a2w.append("Добавить 10–20 ключевых работ: обзорные/методические + 5–10 работ по ближайшим системам/видам (через DOI в Zotero).")
    a2w.append("Собрать минимальный «publishable unit»: один четкий результат + 1 фигура + 1 таблица, и привязать его к 3–5 источникам.")
    a2w.append("Закрыть top-3 evidence gaps: либо найти прямые источники, либо явно описать дизайн данных/эксперимента, который их закроет.")
    a2w.append("Если идея междисциплинарная: найти 3–5 «bridge» статей и переписать формулировку гипотезы так, чтобы мост был тестируемым.")

    # Deduplicate while preserving order
    def dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = x.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    return {"48h": dedup(a48)[:8], "2w": dedup(a2w)[:8]}


# ------------------------- LLM integration -------------------------

_REQUIRED_LLM_KEYS = {"publishability", "breakthrough", "portfolio"}


def parse_llm_response(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    obj = safe_read_json(p)
    if not isinstance(obj, dict):
        return None
    # minimal schema check
    if not _REQUIRED_LLM_KEYS.issubset(set(obj.keys())):
        return None
    return obj


def make_llm_prompt(input_blob: Dict[str, Any]) -> str:
    # Keep the prompt deterministic and strict.
    return (
        "You are a strict scientific evaluator.\n"
        "Task: Score a research idea on TWO independent axes:\n"
        "  (1) Publishability (0-100) — likelihood of producing a publishable, defensible result.\n"
        "  (2) Breakthrough (0-100) — chance of a high-impact conceptual leap.\n\n"
        "Rules:\n"
        "- Ground your reasoning in the provided evidence metrics, gap map, and novelty scores.\n"
        "- Do NOT invent papers, DOIs, or quotes.\n"
        "- If evidence relevance is poor (topic mismatch), say so and penalize publishability.\n"
        "- Output ONLY valid JSON (no markdown).\n"
        "- Scores must be integers 0..100.\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "publishability": {\n'
        '    "score": 0,\n'
        '    "rationale": "...",\n'
        '    "key_risks": ["..."],\n'
        '    "fast_improvements_48h": ["..."],\n'
        '    "two_week_plan": ["..."]\n'
        "  },\n"
        '  "breakthrough": {\n'
        '    "score": 0,\n'
        '    "rationale": "...",\n'
        '    "what_makes_it_breakthrough": ["..."],\n'
        '    "what_would_raise_it": ["..."]\n'
        "  },\n"
        '  "portfolio": {\n'
        '    "quadrant": "HighPayoff-HighPublishability|LowPayoff-HighPublishability|HighPayoff-LowPublishability|LowPayoff-LowPublishability",\n'
        '    "justification": "..." \n'
        "  },\n"
        '  "criteria": {\n'
        '    "answerability": 0,\n'
        '    "feasibility": 0,\n'
        '    "novelty": 0,\n'
        '    "importance": 0,\n'
        '    "evidence_grounding": 0\n'
        "  }\n"
        "}\n\n"
        "JSON input:\n"
        + json.dumps(input_blob, ensure_ascii=False, indent=2)
        + "\n"
    )


# ------------------------- report writers -------------------------

def write_report_md(out_dir: Path, scores_obj: Dict[str, Any]) -> None:
    s = scores_obj.get("scores", {})
    pub = s.get("publishability", {}).get("score")
    bt = s.get("breakthrough", {}).get("score")
    port = scores_obj.get("portfolio", {})
    actions = scores_obj.get("actions", {})

    lines = []
    lines.append(f"# Stage F — Отчёт двойной оценки")
    lines.append("")
    lines.append(f"- Сгенерировано: {scores_obj.get('generated')}")
    lines.append(f"- Статус: {scores_obj.get('status')}")
    lines.append("")
    lines.append("## Оценки")
    lines.append(f"- **Publishability (публикуемость):** {pub}")
    lines.append(f"- **Breakthrough (прорыв):** {bt}")
    if isinstance(port, dict):
        lines.append(f"- **Позиция в портфеле:** {port.get('label')}  ({port.get('quadrant')})")
    lines.append("")

    # Key diagnostics
    m = scores_obj.get("metrics", {})
    ev = m.get("evidence", {})
    cm = m.get("corpus", {})
    nb = m.get("novelty_breakthrough", {})

    lines.append("## Диагностика (по данным)")
    if isinstance(ev, dict) and ev.get("present"):
        lines.append(f"- Строк доказательств: {ev.get('rows')} | утверждений: {ev.get('n_claims')} | медиана источников/утверждение: {ev.get('sources_per_claim_median')}")
        tm = ev.get("topic_mismatch_fraction")
        if tm is not None:
            lines.append(f"- Доля off-topic: {tm:.2f} (чем меньше, тем лучше)")
        relf = ev.get("relation_fractions") or {}
        certf = ev.get("certainty_fractions") or {}
        lines.append(f"- Доли отношений: supports={relf.get('supports')} contradicts={relf.get('contradicts')} unclear={relf.get('unclear')}")
        lines.append(f"- Доли уверенности: High={certf.get('High')} Med={certf.get('Med')} Low={certf.get('Low')}")
    else:
        lines.append("- Нет evidence-таблицы (out/evidence_table.csv).")

    if isinstance(cm, dict) and cm.get("present"):
        lines.append(f"- Размер корпуса: {cm.get('rows')} | медианный год: {cm.get('years_median')} | доля последних 10 лет: {cm.get('recent10_fraction')}")
    else:
        lines.append("- Нет корпуса (out/corpus.csv).")

    if isinstance(nb, dict) and nb.get("present"):
        lines.append(f"- Оценки Stage E: conv={nb.get('conventionality')} novel={nb.get('novel_combination')} bridge={nb.get('bridge_feasibility')} breakthrough={nb.get('breakthrough')}")
    lines.append("")

    # Actions
    lines.append("## Что улучшить быстро")
    for x in (actions.get("48h") or [])[:8]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## План на 2 недели")
    for x in (actions.get("2w") or [])[:8]:
        lines.append(f"- {x}")
    lines.append("")

    # If LLM used, include rationale excerpts
    if scores_obj.get("llm_used"):
        lines.append("## Обоснование LLM (фрагмент)")
        lp = scores_obj.get("llm", {})
        if isinstance(lp, dict):
            for k in ("publishability", "breakthrough"):
                kk = lp.get(k, {})
                if isinstance(kk, dict):
                    lines.append(f"### {k.capitalize()}")
                    lines.append(short(kk.get("rationale", ""), 700))
                    lines.append("")

    safe_write_text(out_dir / "report.md", "\n".join(lines))


def write_portfolio_md(out_dir: Path, pub: int, bt: int, port: Dict[str, Any]) -> None:
    # small deterministic matrix
    hi = port.get("thresholds", {}).get("high", 60)
    lines = []
    lines.append("# Позиция в матрице риск/выигрыш (Stage F)")
    lines.append("")
    lines.append(f"- Publishability (публикуемость): {pub}")
    lines.append(f"- Breakthrough (прорыв): {bt}")
    lines.append(f"- Порог 'High': {hi}")
    lines.append("")
    lines.append("| | Breakthrough LOW | Breakthrough HIGH |")
    lines.append("|---|---|---|")
    lines.append(f"| Publishability HIGH | LowPayoff-HighPublishability | HighPayoff-HighPublishability |")
    lines.append(f"| Publishability LOW  | LowPayoff-LowPublishability  | HighPayoff-LowPublishability  |")
    lines.append("")
    lines.append(f"**Эта идея:** {port.get('label')}  ({port.get('quadrant')})")
    safe_write_text(out_dir / "portfolio_position.md", "\n".join(lines))


# ------------------------- main -------------------------

def run_one(idea_dir: Path, ask_llm: bool) -> int:
    idea_dir = idea_dir.resolve()
    out_dir = idea_dir / "out"
    in_dir = idea_dir / "in"
    ensure_dir(out_dir)
    ensure_dir(in_dir)

    structured = load_structured(out_dir)
    gapm = load_gap_map(out_dir)
    nb = load_novelty_breakthrough(out_dir)
    cm = load_corpus_metrics(out_dir)
    evm_raw = load_evidence_metrics(out_dir)

    # compute scores
    design_score, design_comp, design_notes = score_design(structured)
    evidence_score, evidence_comp, ev_notes = score_evidence(evm_raw, gapm)
    corpus_score, corpus_comp, corpus_notes = score_corpus(cm)

    bt_score, bt_comp, bt_notes = score_breakthrough(nb, evidence_score)

    # publishability: feasibility + evidence + design + corpus
    feasibility = clamp(0.55 * design_score + 0.45 * corpus_score)
    pub = clamp(0.45 * feasibility + 0.40 * evidence_score + 0.10 * design_score + 0.05 * corpus_score)

    # If evidence relevance is very poor, cap publishability.
    tm = evidence_comp.get("topic_mismatch_fraction")
    if isinstance(tm, (int, float)) and tm is not None and tm > 0.7:
        pub = min(pub, 45.0)

    # Optional LLM step
    llm_file = in_dir / "llm_stageF.json"
    llm_obj = parse_llm_response(llm_file)

    if ask_llm and llm_obj is None:
        # prepare prompt
        input_blob = {
            "idea_dir": str(idea_dir),
            "structured_idea": {
                "problem": structured.get("problem"),
                "why_it_matters": structured.get("why_it_matters"),
                "state_of_practice_today": structured.get("state_of_practice_today"),
                "main_hypothesis": structured.get("main_hypothesis"),
                "alternative_hypotheses": structured.get("alternative_hypotheses"),
                "key_predictions": structured.get("key_predictions"),
                "decisive_tests": structured.get("decisive_tests"),
                "minimal_publishable_unit": structured.get("minimal_publishable_unit"),
            },
            "metrics": {
                "design_score": to_int_score(design_score),
                "evidence_score": to_int_score(evidence_score),
                "corpus_score": to_int_score(corpus_score),
                "evidence": evidence_comp,
                "corpus": corpus_comp,
                "novelty_breakthrough": bt_comp,
            },
            "gap_map": {
                "overall": gapm.get("overall"),
                "top_decisive_unknowns": (gapm.get("top_decisive_unknowns") or [])[:6],
                "claims": (gapm.get("claims") or [])[:4],
            },
            "representative_sources": evm_raw.get("top_sources", []),
        }

        prompt = make_llm_prompt(input_blob)
        safe_write_text(out_dir / "llm_prompt_F.txt", prompt)
        if not llm_file.exists():
            safe_write_text(llm_file, "")
        print("[NEED_LLM] Prompt written to out/llm_prompt_F.txt. Paste JSON reply into in/llm_stageF.json and run again.")
        return 2

    # Merge LLM if present and valid
    llm_used = False
    llm_pub = None
    llm_bt = None
    llm_port = None
    llm_clean: Dict[str, Any] = {}

    if llm_obj is not None:
        try:
            llm_pub = int(llm_obj.get("publishability", {}).get("score"))
            llm_bt = int(llm_obj.get("breakthrough", {}).get("score"))
            llm_port = llm_obj.get("portfolio", {})
            llm_used = True
            llm_clean = llm_obj
        except Exception:
            llm_used = False

    # If LLM exists but scores invalid, ignore safely
    if llm_used:
        if not (0 <= llm_pub <= 100 and 0 <= llm_bt <= 100):
            llm_used = False

    pub_final = to_int_score(pub)
    bt_final = to_int_score(bt_score)

    if llm_used:
        # Use LLM as the primary "decision", but keep a sanity blend with heuristics
        pub_final = to_int_score(0.7 * llm_pub + 0.3 * pub_final)
        bt_final = to_int_score(0.7 * llm_bt + 0.3 * bt_final)

    port = portfolio_quadrant(pub_final, bt_final)
    if llm_used and isinstance(llm_port, dict) and isinstance(llm_port.get("quadrant"), str):
        # accept LLM quadrant only if it's one of allowed
        allowed = {
            "HighPayoff-HighPublishability",
            "LowPayoff-HighPublishability",
            "HighPayoff-LowPublishability",
            "LowPayoff-LowPublishability",
        }
        q = llm_port.get("quadrant")
        if q in allowed:
            port["quadrant"] = q
            # adjust label
            labels = {
                "HighPayoff-HighPublishability": "High payoff × High publishability",
                "LowPayoff-HighPublishability": "Low payoff × High publishability",
                "HighPayoff-LowPublishability": "High payoff × Low publishability",
                "LowPayoff-LowPublishability": "Low payoff × Low publishability",
            }
            port["label"] = labels.get(q, port["label"])

    actions = suggest_actions(structured, gapm, ev_notes, design_notes, corpus_notes)

    # Build scores.json
    out_obj = {
        "generated": _now(),
        "status": "ok",
        "idea_dir": str(idea_dir),
        "inputs_present": {
            "structured_idea": bool(structured),
            "evidence_table": bool(evm_raw.get("present")),
            "gap_map": bool(gapm),
            "novelty_breakthrough": bool(nb),
            "corpus": bool(cm.get("present")),
        },
        "metrics": {
            "design": {"score": to_int_score(design_score), "components": design_comp, "notes": design_notes},
            "evidence": {
                "present": bool(evm_raw.get("present")),
                **evidence_comp,
                "relation_fractions": evm_raw.get("relation_fractions"),
                "certainty_fractions": evm_raw.get("certainty_fractions"),
            },
            "corpus": {"present": bool(cm.get("present")), **corpus_comp},
            "novelty_breakthrough": {"present": bool(isinstance(nb.get("scores"), dict)), **bt_comp, "notes": bt_notes},
        },
        "scores": {
            "publishability": {
                "score": pub_final,
                "heuristic_score": to_int_score(pub),
                "feasibility_component": to_int_score(feasibility),
                "evidence_component": to_int_score(evidence_score),
                "design_component": to_int_score(design_score),
                "corpus_component": to_int_score(corpus_score),
                "notes": list(dict.fromkeys(ev_notes + design_notes + corpus_notes))[:10],
            },
            "breakthrough": {
                "score": bt_final,
                "heuristic_score": to_int_score(bt_score),
                "notes": bt_notes,
            },
        },
        "portfolio": port,
        "actions": actions,
        "llm_used": llm_used,
        "llm": llm_clean if llm_used else None,
    }

    # If critical inputs missing, downgrade status but still output
    if not cm.get("present") and not evm_raw.get("present"):
        out_obj["status"] = "missing_inputs"

    safe_write_json(out_dir / "scores.json", out_obj)
    write_report_md(out_dir, out_obj)
    write_portfolio_md(out_dir, pub_final, bt_final, port)

    print(f"[OK] Stage F outputs written: {out_dir / 'scores.json'}; report.md; portfolio_position.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Path to an idea folder (ideas/IDEA-... )")
    ap.add_argument("--ask-llm", action="store_true", help="Generate/consume ChatGPT JSON to improve scoring")
    args = ap.parse_args()

    idea_dir = Path(args.idea)
    if not idea_dir.exists() or not idea_dir.is_dir():
        # try relative to current working dir
        cand = Path.cwd() / args.idea
        if cand.exists() and cand.is_dir():
            idea_dir = cand
        else:
            # write nothing but exit non-zero
            print(f"[FATAL] Idea folder not found: {args.idea}")
            return 1

    try:
        return run_one(idea_dir, ask_llm=bool(args.ask_llm))
    except Exception as e:
        # never crash: attempt to write an error stub in out/
        try:
            out_dir = idea_dir / "out"
            ensure_dir(out_dir)
            safe_write_json(out_dir / "scores.json", {
                "generated": _now(),
                "status": "error",
                "error": str(e),
                "idea_dir": str(idea_dir),
            })
            safe_write_text(out_dir / "report.md", f"# Stage F — ERROR\n\n{str(e)}\n")
        except Exception:
            pass
        print(f"[FATAL] {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
