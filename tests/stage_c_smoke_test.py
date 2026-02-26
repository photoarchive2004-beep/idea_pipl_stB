#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "tools" / "run_c3.py"
IDEA = ROOT / "ideas" / "IDEA-20260224-001"


def run_cmd():
    return subprocess.run([sys.executable, str(ENGINE), "--idea-dir", str(IDEA)], cwd=ROOT)


def ensure_corpus() -> None:
    corpus = IDEA / "out" / "corpus.csv"
    if corpus.exists():
        return
    corpus.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"openalex_id": "W1", "doi": "10.0000/demo.1", "title": "Transferability and connectivity", "year": "2024", "venue": "Demo", "abstract": "Out-of-sample transferability is higher when connectivity profiles are similar."},
        {"openalex_id": "W2", "doi": "10.0000/demo.2", "title": "Latent factors and decline", "year": "2023", "venue": "Demo", "abstract": "After latent factors are added, predictive signals decline in many datasets."},
    ]
    with corpus.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def reset_chat_files() -> None:
    for p in [
        IDEA / "in" / "c3_chatgpt" / "RESPONSE.json",
        IDEA / "out" / "run_id_C3.txt",
    ]:
        if p.exists():
            p.unlink()


def build_llm_answer() -> None:
    run_id = (IDEA / "out" / "run_id_C3.txt").read_text(encoding="utf-8").strip()
    claims = json.loads((IDEA / "in" / "llm_response.json").read_text(encoding="utf-8"))["structured_idea"].get("key_predictions", [])
    rows = []
    for cid in range(1, len(claims) + 1):
        for sid in ("S1", "S2"):
            quote = "Out-of-sample transferability is higher" if sid == "S1" else "latent factors are added"
            rows.append({
                "claim_id": cid,
                "source_id": sid,
                "relation": "unclear",
                "quote": quote,
                "quote_location": "abstract",
                "certainty": "Low",
                "certainty_reason": "limited_detail",
            })
    payload = {"meta": {"run_id": run_id, "idea_id": IDEA.name, "language": "ru"}, "evidence_rows": rows}
    (IDEA / "in" / "c3_chatgpt").mkdir(parents=True, exist_ok=True)
    (IDEA / "in" / "c3_chatgpt" / "RESPONSE.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if not IDEA.exists():
        print("Smoke: отсутствует ideas/IDEA-20260224-001", file=sys.stderr)
        return 1
    ensure_corpus()
    reset_chat_files()

    rc = run_cmd().returncode
    if rc != 2:
        print("Smoke: первый запуск C3 должен вернуть код 2", file=sys.stderr)
        return 1

    for rel in [IDEA / "in" / "c3_chatgpt" / "PROMPT.txt", IDEA / "in" / "c3_chatgpt" / "README_WHAT_TO_DO.txt", IDEA / "in" / "llm_evidence.json"]:
        if not rel.exists():
            print(f"Smoke: не создан файл {rel}", file=sys.stderr)
            return 1

    build_llm_answer()
    rc = run_cmd().returncode
    if rc != 0:
        print("Smoke: второй запуск C3 завершился с ошибкой", file=sys.stderr)
        return 1

    required = [
        IDEA / "out" / "evidence_table.csv",
        IDEA / "out" / "evidence_table.md",
        IDEA / "out" / "evidence_summary.md",
        IDEA / "out" / "evidence_bundle.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Smoke: отсутствуют выходные файлы: " + ", ".join(missing), file=sys.stderr)
        return 1

    print(f"Smoke OK: {IDEA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
