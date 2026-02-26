#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

CSV_COLUMNS = [
    "claim_id", "claim", "source_id", "doi", "title", "year", "venue", "openalex_id",
    "relation", "quote", "quote_location", "certainty", "certainty_reason",
]
RELATIONS = {"supports", "contradicts", "unclear"}
CERTAINTY = {"High", "Med", "Low"}
CERTAINTY_REASON = {
    "topic_mismatch", "indirectness", "limited_detail", "correlational", "review_only",
    "methods_mismatch", "quote_not_found", "other",
}


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self.fp.close()

    def _log(self, lvl: str, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{lvl}] {msg}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)

    def err(self, msg: str) -> None:
        self._log("ERR", msg)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def resolve_idea_dir(root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p
    active = root / "ideas" / "_ACTIVE_PATH.txt"
    if active.exists():
        raw = active.read_text(encoding="utf-8", errors="replace").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (root / raw).resolve()
            if p.exists():
                return p
    ideas = sorted((root / "ideas").glob("IDEA-*"), reverse=True)
    if ideas:
        return ideas[0]
    raise FileNotFoundError("Не найдена папка идеи (ideas/IDEA-*).")


def update_last_log(root: Path, idea_out: Path, module_log: Path) -> None:
    for p in [idea_out / "LAST_LOG.txt", root / "launcher_logs" / "LAST_LOG.txt"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(module_log.resolve()), encoding="utf-8")


def load_claims(out_dir: Path, in_dir: Path) -> Dict[int, str]:
    path = out_dir / "structured_idea.json"
    if not path.exists():
        alt = in_dir / "llm_response.json"
        if alt.exists():
            path = alt
        else:
            return {}
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    si = obj.get("structured_idea", obj) if isinstance(obj, dict) else {}
    claims: List[str] = []
    for key in ("claims_to_verify", "key_predictions"):
        if isinstance(si.get(key), list):
            claims.extend([normalize_ws(x) for x in si[key] if normalize_ws(x)])
    if not claims:
        for key in ("main_hypothesis", "problem"):
            v = normalize_ws(si.get(key, ""))
            if v:
                claims.append(v)
    out: Dict[int, str] = {}
    for i, c in enumerate(claims, 1):
        out[i] = c
    return out


def build_pairs_from_candidates(cands: Dict[str, Any], claims_fallback: Dict[int, str]) -> Tuple[Dict[int, str], List[Dict[str, Any]], str]:
    claims_map = dict(claims_fallback)
    qc = cands.get("qc", {}) if isinstance(cands, dict) else {}
    run_id = normalize_ws((qc or {}).get("run_id", ""))
    pairs: List[Dict[str, Any]] = []

    nested_claims = cands.get("claims", []) if isinstance(cands, dict) else []
    for c in nested_claims:
        try:
            cid = int(c.get("claim_id"))
        except Exception:
            continue
        claim_text = normalize_ws(c.get("claim", claims_map.get(cid, "")))
        if claim_text:
            claims_map[cid] = claim_text
        for s in c.get("sources", []):
            sid = normalize_ws(s.get("source_id", ""))
            if not sid:
                continue
            text_for_llm = normalize_ws(s.get("text_for_llm", ""))
            if not text_for_llm:
                text_for_llm = normalize_ws((s.get("title", "") + " " + s.get("abstract", "")))
            pairs.append({
                "claim_id": cid,
                "claim": claim_text,
                "source_id": sid,
                "doi": normalize_ws(s.get("doi", "")),
                "title": normalize_ws(s.get("title", "")),
                "year": str(s.get("year", "")).strip(),
                "venue": normalize_ws(s.get("venue", "")),
                "openalex_id": normalize_ws(s.get("openalex_id", "")),
                "text_for_llm": text_for_llm,
                "quote_location_default": normalize_ws(s.get("quote_hint", "pdf_text")) or "pdf_text",
            })

    uniq: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for p in pairs:
        uniq[(int(p["claim_id"]), str(p["source_id"]))] = p
    return claims_map, list(uniq.values()), run_id




def load_pairs_from_corpus(out_dir: Path, claims_map: Dict[int, str], top_k: int = 6) -> List[Dict[str, Any]]:
    corpus = out_dir / "corpus.csv"
    if not corpus.exists():
        return []
    rows: List[Dict[str, str]] = []
    with corpus.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs: List[Dict[str, Any]] = []
    for cid in sorted(claims_map):
        for i, r in enumerate(rows[:top_k], 1):
            text_for_llm = normalize_ws((r.get("title", "") + " " + r.get("abstract", "")))
            pairs.append({
                "claim_id": cid,
                "claim": claims_map[cid],
                "source_id": f"S{i}",
                "doi": normalize_ws(r.get("doi", "")),
                "title": normalize_ws(r.get("title", "")),
                "year": str(r.get("year", "")).strip(),
                "venue": normalize_ws(r.get("venue", "")),
                "openalex_id": normalize_ws(r.get("openalex_id", "")),
                "text_for_llm": text_for_llm,
                "quote_location_default": "abstract",
            })
    return pairs



def load_pairs_from_abstract_files(in_dir: Path, claims_map: Dict[int, str], top_k: int = 6) -> List[Dict[str, Any]]:
    abs_dir = in_dir / "papers" / "abstracts"
    if not abs_dir.exists():
        return []
    files = sorted(abs_dir.glob("*.txt"))[:top_k]
    pairs: List[Dict[str, Any]] = []
    for cid in sorted(claims_map):
        for i, fp in enumerate(files, 1):
            txt = fp.read_text(encoding="utf-8", errors="replace")
            lines = [normalize_ws(x) for x in txt.splitlines() if normalize_ws(x)]
            title = lines[0] if lines else fp.stem
            body = normalize_ws(" ".join(lines[1:])) if len(lines) > 1 else normalize_ws(txt)
            pairs.append({
                "claim_id": cid,
                "claim": claims_map[cid],
                "source_id": f"S{i}",
                "doi": "",
                "title": title,
                "year": "",
                "venue": "",
                "openalex_id": fp.stem,
                "text_for_llm": normalize_ws(title + " " + body),
                "quote_location_default": "abstract",
            })
    return pairs

def build_prompt(run_id: str, idea_id: str, pairs: List[Dict[str, Any]]) -> str:
    payload = {
        "qc": {"run_id": run_id, "idea_id": idea_id},
        "task": "Ты строгий evidence judge. Используй ТОЛЬКО text_for_llm.",
        "rules": {
            "no_external_knowledge": True,
            "one_row_per_pair": True,
            "relation": ["supports", "contradicts", "unclear"],
            "quote": "Точная подстрока из text_for_llm, не более 25 слов",
            "quote_location": "title|abstract|pdf_text|pdf:pN",
            "off_topic_rule": "если off-topic: relation=unclear, certainty=Low, certainty_reason=topic_mismatch",
            "anti_stale": "meta.run_id ДОЛЖЕН БЫТЬ РАВЕН qc.run_id",
        },
        "output_schema": {
            "meta": {"run_id": "<copy qc.run_id>", "idea_id": idea_id, "language": "ru"},
            "evidence_rows": [{
                "claim_id": 1,
                "source_id": "S1",
                "relation": "supports|contradicts|unclear",
                "quote": "...",
                "quote_location": "title|abstract|pdf_text|pdf:pN",
                "certainty": "High|Med|Low",
                "certainty_reason": "topic_mismatch|indirectness|limited_detail|correlational|review_only|methods_mismatch|other",
            }],
        },
        "input_rows": [{
            "claim_id": int(p["claim_id"]),
            "claim": p["claim"],
            "source_id": p["source_id"],
            "text_for_llm": p["text_for_llm"],
        } for p in pairs],
    }
    return "Верни строго один JSON-объект без markdown и пояснений.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def read_response_json(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return {}
    return json.loads(raw)


def contains_exact_quote(quote: str, text_for_llm: str) -> bool:
    q = normalize_ws(quote)
    if not q:
        return False
    if len(q.split()) > 25:
        return False
    return q in (text_for_llm or "")


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rel_rank = {"supports": 0, "contradicts": 1, "unclear": 2}
    cert_rank = {"High": 0, "Med": 1, "Low": 2}
    return sorted(rows, key=lambda r: (rel_rank.get(r.get("relation", "unclear"), 9), cert_rank.get(r.get("certainty", "Low"), 9)))


def short_cite(r: Dict[str, Any]) -> str:
    year = str(r.get("year", "")).strip() or "n.d."
    title = normalize_ws(r.get("title", ""))
    author_guess = title.split()[0] if title else "Title"
    doi = normalize_ws(r.get("doi", "")) or "no-doi"
    venue = normalize_ws(r.get("venue", "")) or "no-venue"
    return f"{author_guess}{year}; DOI: {doi}; venue: {venue}"


def write_human_md(path: Path, claims_map: Dict[int, str], rows: List[Dict[str, Any]], warnings: List[str]) -> None:
    by_claim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_claim[int(r["claim_id"])].append(r)

    lines: List[str] = ["# Evidence Table (human-readable)", "", "## At-a-glance"]
    for cid in sorted(claims_map):
        rr = by_claim.get(cid, [])
        cnt = {"supports": 0, "contradicts": 0, "unclear": 0}
        off_topic = 0
        for r in rr:
            rel = r.get("relation", "unclear")
            cnt[rel] = cnt.get(rel, 0) + 1
            if r.get("certainty_reason") == "topic_mismatch":
                off_topic += 1
        lines.append(f"- Claim {cid}: supports={cnt['supports']}, contradicts={cnt['contradicts']}, unclear={cnt['unclear']}, off-topic={off_topic}")
    lines.append("")

    for cid in sorted(claims_map):
        lines.append(f"## Claim {cid}")
        lines.append(claims_map[cid])
        lines.append("")
        for r in sort_rows(by_claim.get(cid, [])):
            qloc = normalize_ws(r.get("quote_location", "")) or "pdf_text"
            lines.append(f"- **{short_cite(r)}**")
            lines.append(f"  - relation: `{r.get('relation')}`; certainty: `{r.get('certainty')}`; reason: `{r.get('certainty_reason')}`")
            lines.append(f"  - quote_location: `{qloc}`")
            lines.append(f"  - quote:")
            lines.append(f"    > {normalize_ws(r.get('quote', '')) or '(цитата отсутствует)'}")
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, claims_map: Dict[int, str], rows: List[Dict[str, Any]], warnings: List[str], fulltext_available: bool) -> None:
    by_claim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_claim[int(r["claim_id"])].append(r)
    lines = ["# Сводка Stage C3", ""]
    lines.append(f"- Fulltext available: {'yes' if fulltext_available else 'no (работа только по abstracts/title)'}")
    for cid in sorted(claims_map):
        rr = by_claim.get(cid, [])
        s = sum(1 for x in rr if x.get("relation") == "supports")
        c = sum(1 for x in rr if x.get("relation") == "contradicts")
        u = sum(1 for x in rr if x.get("relation") == "unclear")
        lines.append(f"- Claim {cid}: supports={s}, contradicts={c}, unclear={u}")
    lines.append(f"- Предупреждений QC: {len(warnings)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bundle(run_id: str, idea_id: str, claims_map: Dict[int, str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_claim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_claim[int(r["claim_id"])].append(r)

    per_claim_summary = []
    for cid in sorted(claims_map):
        rr = sort_rows(by_claim.get(cid, []))
        supports = [r for r in rr if r.get("relation") == "supports"]
        contradicts = [r for r in rr if r.get("relation") == "contradicts"]
        unclear = [r for r in rr if r.get("relation") == "unclear"]
        per_claim_summary.append({
            "claim_id": cid,
            "counts": {"supports": len(supports), "contradicts": len(contradicts), "unclear": len(unclear)},
            "top_supporting": [{"source_id": r.get("source_id"), "doi": r.get("doi"), "quote": r.get("quote")} for r in supports[:3]],
            "top_contradicting": [{"source_id": r.get("source_id"), "doi": r.get("doi"), "quote": r.get("quote")} for r in contradicts[:3]],
        })

    return {
        "meta": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "idea_id": idea_id,
        },
        "claims": [{"claim_id": cid, "claim_text": claims_map[cid]} for cid in sorted(claims_map)],
        "per_claim_summary": per_claim_summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C3: Evidence Table Engine")
    parser.add_argument("--idea-dir", default="", help="Папка идеи")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    idea_dir = resolve_idea_dir(root, args.idea_dir or None)
    idea_id = idea_dir.name
    out_dir = idea_dir / "out"
    in_dir = idea_dir / "in"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "module_C3.log")
    update_last_log(root, out_dir, logger.path)

    try:
        logger.info(f"Stage C3: старт. Папка идеи: {idea_dir}")
        cands_path = out_dir / "evidence_candidates.json"
        claims_map = load_claims(out_dir, in_dir)
        run_id = ""
        pairs: List[Dict[str, Any]] = []
        if cands_path.exists():
            cands = json.loads(cands_path.read_text(encoding="utf-8", errors="replace"))
            claims_map, pairs, run_id = build_pairs_from_candidates(cands, claims_map)
        else:
            logger.warn("out/evidence_candidates.json отсутствует; использую fallback по out/corpus.csv или in/papers/abstracts")
            pairs = load_pairs_from_corpus(out_dir, claims_map, top_k=6)
            if not pairs:
                pairs = load_pairs_from_abstract_files(in_dir, claims_map, top_k=6)
        if not claims_map:
            raise RuntimeError("Не удалось извлечь claims из structured_idea.json/evidence_candidates.json.")
        if not pairs:
            raise RuntimeError("Не найдены пары claim×source в out/evidence_candidates.json.")

        run_id_file = out_dir / "run_id_C3.txt"
        if not run_id:
            run_id = normalize_ws(run_id_file.read_text(encoding="utf-8", errors="replace")) if run_id_file.exists() else ""
        if not run_id:
            run_id = f"c3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id_file.write_text(run_id, encoding="utf-8")

        chat_dir = in_dir / "c3_chatgpt"
        prompt_path = chat_dir / "PROMPT.txt"
        readme_path = chat_dir / "README_WHAT_TO_DO.txt"
        response_path = chat_dir / "RESPONSE.json"
        wait_beacon_path = in_dir / "llm_evidence.json"

        response_obj: Dict[str, Any] | None = None
        if response_path.exists():
            try:
                maybe = read_response_json(response_path)
                if str((maybe.get("meta") or {}).get("run_id", "")).strip() == run_id:
                    response_obj = maybe
                else:
                    logger.warn("RESPONSE.json найден, но run_id не совпадает. Будет создан новый prompt.")
            except Exception as exc:
                logger.warn(f"RESPONSE.json невалиден: {exc}")

        if response_obj is None:
            chat_dir.mkdir(parents=True, exist_ok=True)
            prompt_text = build_prompt(run_id, idea_id, pairs)
            prompt_path.write_text(prompt_text, encoding="utf-8")
            readme_path.write_text(
                "1) Откройте PROMPT.txt и отправьте текст в ChatGPT.\n"
                "2) Скопируйте строго JSON-ответ и вставьте в RESPONSE.json.\n"
                "3) Сохраните RESPONSE.json и перезапустите RUN_C3.bat (или общий RUN_C).\n",
                encoding="utf-8",
            )
            if not response_path.exists():
                response_path.write_text(json.dumps({
                    "meta": {"run_id": run_id, "idea_id": idea_id, "language": "ru"},
                    "evidence_rows": []
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            write_json(wait_beacon_path, {
                "paste": True,
                "reason_pipeline_waiting": "Stage C3 ожидает JSON-ответ ChatGPT в in/c3_chatgpt/RESPONSE.json",
                "expected_run_id": run_id,
            })
            write_json(out_dir / "_moduleC3_checkpoint.json", {
                "phase": "awaiting_chatgpt_response",
                "run_id": run_id,
                "pairs_total": len(pairs),
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            logger.info("Откройте in/c3_chatgpt/PROMPT.txt, отправьте в ChatGPT и вставьте JSON в RESPONSE.json.")
            print("\nОжидание ChatGPT: откройте in/c3_chatgpt/PROMPT.txt")
            return 2

        by_pair = {(int(p["claim_id"]), str(p["source_id"])): p for p in pairs}
        raw_rows = response_obj.get("evidence_rows", [])
        if not isinstance(raw_rows, list):
            raise RuntimeError("RESPONSE.json: поле evidence_rows должно быть массивом.")

        warnings: List[str] = []
        final_rows: List[Dict[str, Any]] = []
        for key, pair in by_pair.items():
            cid, sid = key
            found = [r for r in raw_rows if int(r.get("claim_id", -1)) == cid and str(r.get("source_id", "")).strip() == sid]
            r = found[0] if len(found) == 1 else {}
            if len(found) != 1:
                warnings.append(f"{key}: ожидалась 1 строка, получено {len(found)}; применена деградация")

            rel = str(r.get("relation", "unclear")).strip().lower()
            if rel not in RELATIONS:
                rel = "unclear"
            cert = str(r.get("certainty", "Low")).strip()
            if cert not in CERTAINTY:
                cert = "Low"
            reason = str(r.get("certainty_reason", "other")).strip()
            if reason not in CERTAINTY_REASON:
                reason = "other"
            quote = normalize_ws(r.get("quote", ""))
            qloc = normalize_ws(r.get("quote_location", "")) or pair.get("quote_location_default") or "pdf_text"

            if not contains_exact_quote(quote, pair.get("text_for_llm", "")):
                if quote:
                    warnings.append(f"{key}: quote не найден в text_for_llm -> quote_not_found")
                rel, cert, reason, quote = "unclear", "Low", "quote_not_found", ""

            final_rows.append({
                "claim_id": cid,
                "claim": pair.get("claim", claims_map.get(cid, "")),
                "source_id": sid,
                "doi": pair.get("doi", ""),
                "title": pair.get("title", ""),
                "year": pair.get("year", ""),
                "venue": pair.get("venue", ""),
                "openalex_id": pair.get("openalex_id", ""),
                "relation": rel,
                "quote": quote,
                "quote_location": qloc,
                "certainty": cert,
                "certainty_reason": reason,
            })

        write_csv(out_dir / "evidence_table.csv", final_rows)
        fulltext_available = any("pdf" in str(r.get("quote_location", "")).lower() for r in final_rows)
        write_human_md(out_dir / "evidence_table.md", claims_map, final_rows, warnings)
        write_summary(out_dir / "evidence_summary.md", claims_map, final_rows, warnings, fulltext_available)
        bundle = build_bundle(run_id, idea_id, claims_map, final_rows)
        write_json(out_dir / "evidence_bundle.json", bundle)
        write_json(out_dir / "_moduleC3_checkpoint.json", {
            "phase": "completed",
            "run_id": run_id,
            "rows_written": len(final_rows),
            "warnings": len(warnings),
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        if warnings:
            write_json(chat_dir / "RESPONSE.WARNINGS.json", {"warnings": warnings})
            logger.warn(f"Предупреждений QC: {len(warnings)}")

        logger.info("Stage C3 завершён успешно.")
        print("\nКраткая сводка C3:")
        print(f"- Claim-Source пар: {len(by_pair)}")
        print(f"- Строк evidence: {len(final_rows)}")
        print(f"- Предупреждений: {len(warnings)}")
        print("- Файлы: out/evidence_table.csv, out/evidence_table.md, out/evidence_summary.md, out/evidence_bundle.json")
        return 0
    except Exception as exc:
        logger.err(str(exc))
        print(f"Ошибка Stage C3. Подробности в логе: {logger.path}")
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
