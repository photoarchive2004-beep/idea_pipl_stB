#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "tools" / "c_evidence_engine.py"


def find_idea() -> Path:
    ideas = sorted((ROOT / "ideas").glob("IDEA-*"))
    if not ideas:
        raise RuntimeError("Не найдено ни одной папки IDEA-* для smoke-теста.")
    return ideas[0]


def restore_structured(idea: Path) -> None:
    dst = idea / "out" / "structured_idea.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    src = idea / "in" / "llm_response.json"
    if not src.exists():
        raise RuntimeError("Нет out/structured_idea.json и нет in/llm_response.json для восстановления.")
    raw = src.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"{[\s\S]*}", raw)
    if not m:
        raise RuntimeError("Не удалось извлечь JSON из in/llm_response.json.")
    dst.write_text(m.group(0), encoding="utf-8")


def ensure_corpus(idea: Path) -> None:
    corpus = idea / "out" / "corpus.csv"
    if corpus.exists():
        return
    rows = [
        {
            "openalex_id": "W1",
            "doi": "10.0000/demo.1",
            "title": "Cross-validation and out-of-sample transferability in structured populations",
            "year": "2024",
            "venue": "Demo Journal",
            "cited_by": "5",
            "abstract": "Out-of-sample prediction improves when connectivity and resistance metrics are integrated with genomic distance.",
        },
        {
            "openalex_id": "W2",
            "doi": "10.0000/demo.2",
            "title": "Latent factors may reduce spurious associations",
            "year": "2023",
            "venue": "Methods Letters",
            "cited_by": "3",
            "abstract": "After latent spatial factors are added, many associations lose significance and predictive accuracy declines.",
        },
    ]
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_cmd(args):
    return subprocess.run([sys.executable, str(ENGINE), "--idea", str(args[0]), *args[1:]], cwd=ROOT)


def build_llm_answer(idea: Path) -> None:
    candidates = json.loads((idea / "out" / "evidence_candidates.json").read_text(encoding="utf-8"))
    run_id = (idea / "out" / "run_id_C_pending.txt").read_text(encoding="utf-8").strip()
    rows = []
    for claim in candidates.get("claims", []):
        cid = int(claim["claim_id"])
        for src in claim.get("sources", []):
            text = (src.get("text_for_llm") or "").strip()
            quote = " ".join(text.split()[:12]) if text else ""
            rows.append(
                {
                    "claim_id": cid,
                    "source_id": src.get("source_id"),
                    "relation": "unclear",
                    "quote": quote,
                    "quote_location": "abstract_only",
                    "certainty": "Low",
                    "certainty_reason": "smoke_test",
                }
            )
    payload = {"meta": {"run_id": run_id, "language": "ru"}, "evidence_rows": rows}
    (idea / "in" / "llm_evidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    idea = find_idea()
    restore_structured(idea)
    ensure_corpus(idea)

    rc = run_cmd([idea, "--mode", "fast", "--no-llm"]).returncode
    if rc != 0:
        print("Smoke: fast --no-llm завершился с ошибкой", file=sys.stderr)
        return 1

    rc = run_cmd([idea, "--mode", "deep"]).returncode
    if rc != 2:
        print("Smoke: deep (первый прогон) должен вернуть код 2", file=sys.stderr)
        return 1

    build_llm_answer(idea)
    rc = run_cmd([idea, "--mode", "deep"]).returncode
    if rc != 0:
        print("Smoke: deep (второй прогон) завершился с ошибкой", file=sys.stderr)
        return 1

    if not (idea / "out" / "evidence_table.csv").exists() or not (idea / "out" / "evidence_summary.md").exists():
        print("Smoke: не созданы out/evidence_table.csv и out/evidence_summary.md", file=sys.stderr)
        return 1

    print(f"Smoke OK: {idea}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
