#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

CSV_COLUMNS = [
    "claim_id",
    "claim",
    "source_id",
    "doi",
    "title",
    "year",
    "venue",
    "openalex_id",
    "relation",
    "quote",
    "quote_location",
    "certainty",
    "certainty_reason",
]

RELATIONS = {"supports", "contradicts", "unclear"}
CERTAINTY = {"High", "Med", "Low"}
CERTAINTY_REASON = {
    "topic_mismatch",
    "indirectness",
    "limited_detail",
    "correlational",
    "review_only",
    "methods_mismatch",
    "quote_not_found",
    "other",
}
QUOTE_LOCATION = {"abstract", "title", "fulltext", "unknown"}


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def to_word_limited_quote(s: str, max_words: int = 25) -> str:
    words = normalize_ws(s).split()
    return " ".join(words[:max_words])


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


def copy_to_clipboard(text: str) -> bool:
    for cmd in (
        ["clip"],
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
        ["xclip", "-selection", "clipboard"],
        ["pbcopy"],
    ):
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.communicate(text.encode("utf-8"), timeout=10)
            if p.returncode == 0:
                return True
        except Exception:
            continue
    return False


def update_last_log(root: Path, module_log: Path) -> None:
    for p in [root / "LAST_LOG.txt", root / "launcher_logs" / "LAST_LOG.txt"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(module_log.resolve()), encoding="utf-8")


def extract_claims_sources_pairs(cands: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    claims: Dict[int, str] = {}
    for c in cands.get("claims", []) or []:
        try:
            cid = int(c.get("claim_id", c.get("id")))
        except Exception:
            continue
        claims[cid] = normalize_ws(c.get("claim", c.get("text", "")))

    sources: Dict[str, Dict[str, str]] = {}
    for s in cands.get("sources", []) or []:
        sid = str(s.get("source_id", s.get("id", ""))).strip()
        if not sid:
            continue
        sources[sid] = {
            "source_id": sid,
            "doi": normalize_ws(s.get("doi", "")),
            "title": normalize_ws(s.get("title", "")),
            "year": str(s.get("year", "")).strip(),
            "venue": normalize_ws(s.get("venue", "")),
            "openalex_id": normalize_ws(s.get("openalex_id", "")),
            "text_for_llm": str(s.get("text_for_llm", "") or ""),
        }

    raw_pairs = cands.get("pairs") or cands.get("evidence_pairs") or cands.get("candidates") or []
    pairs: List[Dict[str, Any]] = []
    for p in raw_pairs:
        try:
            cid = int(p.get("claim_id"))
            sid = str(p.get("source_id", "")).strip()
        except Exception:
            continue
        if not sid:
            continue
        claim_text = normalize_ws(p.get("claim", "")) or claims.get(cid, "")
        source_meta = sources.get(sid, {})
        row = {
            "claim_id": cid,
            "claim": claim_text,
            "source_id": sid,
            "doi": normalize_ws(p.get("doi", source_meta.get("doi", ""))),
            "title": normalize_ws(p.get("title", source_meta.get("title", ""))),
            "year": str(p.get("year", source_meta.get("year", ""))).strip(),
            "venue": normalize_ws(p.get("venue", source_meta.get("venue", ""))),
            "openalex_id": normalize_ws(p.get("openalex_id", source_meta.get("openalex_id", ""))),
            "text_for_llm": str(p.get("text_for_llm", source_meta.get("text_for_llm", "")) or ""),
        }
        if cid not in claims and claim_text:
            claims[cid] = claim_text
        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "doi": row["doi"],
                "title": row["title"],
                "year": row["year"],
                "venue": row["venue"],
                "openalex_id": row["openalex_id"],
                "text_for_llm": row["text_for_llm"],
            }
        pairs.append(row)

    return claims, sources, pairs


def build_prompt(run_id: str, rows_for_prompt: List[Dict[str, Any]]) -> str:
    payload = {
        "meta": {
            "run_id": run_id,
            "language": "ru",
            "instruction": "Верни только JSON по заданной схеме. Use ONLY text_for_llm. Нельзя использовать внешние знания.",
        },
        "rules": {
            "relation": ["supports", "contradicts", "unclear"],
            "quote": "точная подстрока из text_for_llm, <= 25 слов",
            "off_topic": "relation=unclear, certainty=Low, certainty_reason=topic_mismatch",
            "not_answering_claim": "relation=unclear, certainty=Low, certainty_reason=indirectness или methods_mismatch",
            "anti_stale": "meta.run_id должен точно совпасть",
        },
        "output_format": {
            "meta": {"run_id": "<copy qc.run_id>", "language": "ru"},
            "evidence_rows": [
                {
                    "claim_id": 1,
                    "source_id": "S1",
                    "relation": "supports|contradicts|unclear",
                    "quote": "...",
                    "quote_location": "abstract|title|fulltext|unknown",
                    "certainty": "High|Med|Low",
                    "certainty_reason": "topic_mismatch|indirectness|limited_detail|correlational|review_only|methods_mismatch|other",
                }
            ],
        },
        "input_rows": rows_for_prompt,
    }
    return "Верни строго один JSON-объект без пояснений и markdown.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def quote_found(quote: str, text_for_llm: str) -> bool:
    q = quote.strip()
    return bool(q) and (q in (text_for_llm or ""))


def relation_rank(row: Dict[str, Any]) -> Tuple[int, int]:
    rel = row.get("relation", "unclear")
    cert = row.get("certainty", "Low")
    cert_rank = {"High": 0, "Med": 1, "Low": 2}.get(cert, 3)
    if rel == "contradicts" and cert in {"High", "Med"}:
        return (0, cert_rank)
    if rel == "supports" and cert in {"High", "Med"}:
        return (1, cert_rank)
    if rel == "unclear":
        return (2, cert_rank)
    return (3, cert_rank)


def write_summary(path: Path, qc: Dict[str, Any], claims_map: Dict[int, str], final_rows: List[Dict[str, Any]]) -> None:
    by_claim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in final_rows:
        by_claim[int(r["claim_id"])].append(r)

    lines: List[str] = []
    lines.append("# Evidence Summary (C2)")
    lines.append("")
    lines.append("## QC")
    for k in ["mode", "run_id", "papers_total", "kept_after_screening", "no_abstract_pct", "allowed_domains", "domain_counts", "anchors", "gap_warnings"]:
        if k in qc:
            lines.append(f"- **{k}**: {qc.get(k)}")
    lines.append("")

    for cid in sorted(claims_map):
        rows = sorted(by_claim.get(cid, []), key=relation_rank)
        lines.append(f"## Claim {cid}")
        lines.append(claims_map[cid] or "(текст claim отсутствует)")
        lines.append("")
        lines.append("**Лучшие источники**")
        best = [r for r in rows if r.get("relation") in {"supports", "contradicts"} and r.get("certainty") in {"High", "Med"}]
        if not best:
            lines.append("- Нет источников с relation supports/contradicts и certainty >= Med")
        else:
            for r in best[:5]:
                lines.append(f"- {r['relation']} ({r['certainty']}): {r.get('title','')} ({r.get('year','')})")

        problems: List[str] = []
        for r in rows:
            cr = r.get("certainty_reason", "")
            if cr in {"topic_mismatch", "indirectness", "methods_mismatch", "quote_not_found"}:
                problems.append(f"{r.get('source_id')}: {cr}")
        lines.append("")
        lines.append("**Проблемы**")
        if problems:
            for p in problems[:10]:
                lines.append(f"- {p}")
        else:
            lines.append("- Явные проблемы не обнаружены")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readable_table(path: Path, claims_map: Dict[int, str], final_rows: List[Dict[str, Any]]) -> None:
    by_claim: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in final_rows:
        by_claim[int(r["claim_id"])].append(r)

    out: List[str] = ["# Evidence Table (readable)", ""]
    for cid in sorted(claims_map):
        rows = sorted(by_claim.get(cid, []), key=relation_rank)
        out.append(f"## Claim {cid}: {claims_map[cid]}")
        out.append("")
        out.append("| Relation | Certainty | Source (Year, Venue, DOI, Title) | Quote |")
        out.append("|---|---|---|---|")
        for r in rows:
            source = f"{r.get('source_id','')} ({r.get('year','')}, {r.get('venue','')}, DOI: {r.get('doi','')}, {r.get('title','')[:80]})"
            quote = normalize_ws(r.get("quote", ""))[:180]
            out.append(f"| {r.get('relation','')} | {r.get('certainty','')} | {source} | {quote} |")
        has_strong = any(r.get("relation") in {"supports", "contradicts"} and r.get("certainty") in {"High", "Med"} for r in rows)
        out.append("")
        out.append(f"**Evidence gap?** {'Да' if not has_strong else 'Нет'}")
        out.append("")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C2: evidence table builder")
    parser.add_argument("--idea-dir", default="", help="Папка идеи")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    idea_dir = resolve_idea_dir(root, args.idea_dir or None)
    out_dir = idea_dir / "out"
    in_dir = idea_dir / "in"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "module_C2.log")
    update_last_log(root, logger.path)

    try:
        logger.info(f"Stage C2: старт. Папка идеи: {idea_dir}")
        cands_path = out_dir / "evidence_candidates.json"
        if not cands_path.exists():
            raise FileNotFoundError("Не найден out/evidence_candidates.json. Сначала запустите Stage C1 (RUN_C1.bat).")

        cands = json.loads(cands_path.read_text(encoding="utf-8", errors="replace"))
        qc = cands.get("qc", {}) if isinstance(cands, dict) else {}
        run_id = str(qc.get("run_id", "")).strip()
        if not run_id:
            run_id = f"c2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.warn("В evidence_candidates.json нет qc.run_id, использую сгенерированный run_id.")

        claims_map, _, pairs = extract_claims_sources_pairs(cands)
        if not pairs:
            raise RuntimeError("В evidence_candidates.json не найдены пары claim_id × source_id (раздел pairs).")

        key_to_pair = {(int(p["claim_id"]), str(p["source_id"])): p for p in pairs}
        prompt_rows = [
            {
                "claim_id": int(p["claim_id"]),
                "claim": p["claim"],
                "source_id": p["source_id"],
                "text_for_llm": p.get("text_for_llm", ""),
            }
            for p in pairs
        ]

        chat_dir = in_dir / "c2_chatgpt"
        prompt_path = chat_dir / "PROMPT.txt"
        readme_path = chat_dir / "README_WHAT_TO_DO.txt"
        resp_path = chat_dir / "RESPONSE.json"

        need_prompt = True
        response_obj: Dict[str, Any] | None = None
        if resp_path.exists():
            txt = resp_path.read_text(encoding="utf-8", errors="replace").strip()
            if txt and "вставьте сюда" not in txt.lower():
                try:
                    maybe = json.loads(txt)
                    if str((maybe.get("meta") or {}).get("run_id", "")).strip() == run_id:
                        response_obj = maybe
                        need_prompt = False
                    else:
                        logger.warn("RESPONSE.json найден, но run_id не совпадает с qc.run_id. Нужен новый проход.")
                except Exception:
                    logger.warn("RESPONSE.json невалиден; запускаю режим ожидания ответа ChatGPT.")

        if need_prompt:
            chat_dir.mkdir(parents=True, exist_ok=True)
            prompt_text = build_prompt(run_id, prompt_rows)
            prompt_path.write_text(prompt_text, encoding="utf-8")
            readme_path.write_text(
                "1) Откройте файл PROMPT.txt и вставьте его в ChatGPT.\n"
                "2) Скопируйте строго JSON-ответ и вставьте в RESPONSE.json.\n"
                "3) Сохраните RESPONSE.json и запустите RUN_C2.bat повторно.\n",
                encoding="utf-8",
            )
            if not resp_path.exists():
                resp_path.write_text('{\n  "meta": {"run_id": "' + run_id + '", "language": "ru"},\n  "evidence_rows": []\n}\n', encoding="utf-8")
            copy_ok = copy_to_clipboard(prompt_text)
            logger.info("Подготовлены файлы in/c2_chatgpt: PROMPT.txt, README_WHAT_TO_DO.txt, RESPONSE.json")
            print("\nДействие пользователя: отправьте PROMPT.txt в ChatGPT и сохраните JSON в RESPONSE.json, затем перезапустите C2.")
            print("PROMPT скопирован в буфер обмена." if copy_ok else "Не удалось скопировать PROMPT в буфер обмена автоматически.")
            write_json(out_dir / "_moduleC2_checkpoint.json", {
                "phase": "awaiting_chatgpt_response",
                "stats": {"pairs_total": len(pairs)},
                "idea_dir": str(idea_dir),
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            logger.info("Stage C2: ожидается RESPONSE.json (ExitCode=2)")
            return 2

        assert response_obj is not None
        rows = response_obj.get("evidence_rows", [])
        if not isinstance(rows, list):
            raise RuntimeError("RESPONSE.json: поле evidence_rows должно быть массивом.")

        by_key: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            try:
                k = (int(r.get("claim_id")), str(r.get("source_id", "")).strip())
            except Exception:
                continue
            if k[1]:
                by_key[k].append(r)

        warnings: List[str] = []
        final_rows: List[Dict[str, Any]] = []
        for key, pair in key_to_pair.items():
            candidates = by_key.get(key, [])
            if len(candidates) != 1:
                warnings.append(f"{key}: ожидалась 1 строка, получено {len(candidates)}; применена деградация")
                rr = {}
            else:
                rr = candidates[0]

            relation = str(rr.get("relation", "unclear"))
            if relation not in RELATIONS:
                relation = "unclear"
            quote_location = str(rr.get("quote_location", "unknown"))
            if quote_location not in QUOTE_LOCATION:
                quote_location = "unknown"
            certainty = str(rr.get("certainty", "Low"))
            if certainty not in CERTAINTY:
                certainty = "Low"
            certainty_reason = str(rr.get("certainty_reason", "other"))
            if certainty_reason not in CERTAINTY_REASON:
                certainty_reason = "other"

            quote = to_word_limited_quote(str(rr.get("quote", "")))
            text_for_llm = str(pair.get("text_for_llm", ""))
            if not quote_found(quote, text_for_llm):
                if quote:
                    warnings.append(f"{key}: цитата не найдена в text_for_llm -> деградация quote_not_found")
                relation = "unclear"
                certainty = "Low"
                certainty_reason = "quote_not_found"
                quote = ""

            final_rows.append({
                "claim_id": int(pair["claim_id"]),
                "claim": pair.get("claim", claims_map.get(int(pair["claim_id"]), "")),
                "source_id": pair.get("source_id", ""),
                "doi": pair.get("doi", ""),
                "title": pair.get("title", ""),
                "year": pair.get("year", ""),
                "venue": pair.get("venue", ""),
                "openalex_id": pair.get("openalex_id", ""),
                "relation": relation,
                "quote": quote,
                "quote_location": quote_location,
                "certainty": certainty,
                "certainty_reason": certainty_reason,
            })

        write_csv(out_dir / "evidence_table.csv", final_rows, CSV_COLUMNS)
        write_summary(out_dir / "evidence_summary.md", qc, claims_map, final_rows)
        write_readable_table(out_dir / "evidence_table.md", claims_map, final_rows)

        checkpoint = {
            "phase": "completed",
            "stats": {
                "pairs_total": len(pairs),
                "rows_written": len(final_rows),
                "warnings": len(warnings),
            },
            "idea_dir": str(idea_dir),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(out_dir / "_moduleC2_checkpoint.json", checkpoint)
        if warnings:
            write_json(chat_dir / "RESPONSE.WARNINGS.json", {"warnings": warnings})
            for w in warnings[:20]:
                logger.warn(w)

        logger.info("Stage C2 завершён успешно.")
        print("\nКраткая сводка C2:")
        print(f"- Пар всего: {len(pairs)}")
        print(f"- Строк записано: {len(final_rows)}")
        print(f"- Предупреждений: {len(warnings)}")
        print("- Выходные файлы: out/evidence_table.csv, out/evidence_summary.md, out/evidence_table.md")
        return 0
    except Exception as exc:
        logger.err(str(exc))
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
