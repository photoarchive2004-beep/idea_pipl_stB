# -*- coding: utf-8 -*-
import argparse, json, re, traceback
from pathlib import Path
from datetime import datetime

def now_stamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    m = re.search(r"{[\s\S]*}", s)
    if not m:
        raise ValueError("No JSON object found in llm_stageA.json")
    return json.loads(m.group(0))

def is_placeholder(obj: dict) -> bool:
    keys = set(obj.keys())
    if keys.issubset({"paste","note","reason_pipeline_waiting"}):
        return True
    if "paste" in obj and "Paste ChatGPT JSON" in str(obj.get("paste","")):
        return True
    return False

def validate(obj: dict):
    if not isinstance(obj, dict): return False, "Response is not a JSON object."
    if is_placeholder(obj): return False, "Placeholder JSON (paste ChatGPT output)."
    if "meta" not in obj or "structured_idea" not in obj:
        return False, "Missing meta/structured_idea."
    si = obj.get("structured_idea") or {}
    need = ["problem","main_hypothesis","alternative_hypotheses","key_predictions","decisive_tests"]
    for k in need:
        if k not in si: return False, f"Missing structured_idea.{k}"
    alts = si.get("alternative_hypotheses")
    if not isinstance(alts, list) or len(alts) < 2:
        return False, "Need >=2 alternative hypotheses."
    return True, "OK"

def write_tests_md(path: Path, obj: dict):
    si = obj["structured_idea"]
    lines = ["# Решающие тесты (Stage A)\n"]
    for i, t in enumerate(si.get("decisive_tests", []) or [], 1):
        lines.append(f"## Тест {i}: {t.get('test','(без названия)')}\n")
        if t.get("data_needed"): lines.append(f"**Нужные данные:** {t['data_needed']}\n")
        if t.get("analysis"): lines.append(f"**Анализ:** {t['analysis']}\n")
        exp = t.get("expected_patterns_by_hypothesis", {}) or {}
        if exp:
            lines.append("**Ожидаемые паттерны по гипотезам:**\n")
            for hk, hv in exp.items():
                lines.append(f"- {hk}: {hv}\n")
        if t.get("midterm_exam"): lines.append(f"**Промежуточный экзамен:** {t['midterm_exam']}\n")
        if t.get("final_exam"): lines.append(f"**Финальный экзамен:** {t['final_exam']}\n")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="Idea folder containing idea.txt")
    args = ap.parse_args()

    idea_dir = Path(args.idea)
    in_dir, out_dir, logs_dir = idea_dir/"in", idea_dir/"out", idea_dir/"logs"
    ensure_dir(in_dir); ensure_dir(out_dir); ensure_dir(logs_dir)

    log_path = logs_dir / f"moduleA_{now_stamp()}.log"
    def log(msg: str):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        idea_txt_path = idea_dir/"idea.txt"
        if not idea_txt_path.exists():
            log("[ERROR] idea.txt not found")
            return 1

        idea_txt = idea_txt_path.read_text(encoding="utf-8", errors="ignore")

        prompt_tpl = (Path(__file__).resolve().parents[1]/"config"/"prompts"/"llm_moduleA_prompt.txt")
        if not prompt_tpl.exists():
            log("[ERROR] prompt template not found: " + str(prompt_tpl))
            return 1

        prompt = prompt_tpl.read_text(encoding="utf-8", errors="ignore").replace("{{IDEA_TEXT}}", idea_txt)
        # Keep naming consistent with later stages (E/F/G): llm_prompt_<Stage>.txt
        (out_dir/"llm_prompt_A.txt").write_text(prompt, encoding="utf-8")

        # Keep naming consistent with later stages: in\\llm_stageA.json
        llm_path = in_dir/"llm_stageA.json"
        if not llm_path.exists():
            llm_path.write_text('{\n  "paste": "Вставьте сюда ТОЛЬКО JSON-ответ ChatGPT (замените весь файл)",\n  "note": "Сохраните файл и запустите RUN_A.bat ещё раз"\n}\n', encoding="utf-8")
            log("[NEED] llm_stageA.json missing. Prompt generated.")
            return 2

        raw = llm_path.read_text(encoding="utf-8", errors="ignore")
        obj = extract_json(raw)
        ok, why = validate(obj)
        if not ok:
            llm_path.write_text('{\n  "paste": "Вставьте сюда ТОЛЬКО JSON-ответ ChatGPT (замените весь файл)",\n  "note": "Сохраните файл и запустите RUN_A.bat ещё раз",\n  "reason_pipeline_waiting": "' + why.replace('"',"'") + '"\n}\n', encoding="utf-8")
            log("[NEED] invalid llm json: " + why)
            return 2

        (out_dir/"structured_idea.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        write_tests_md(out_dir/"tests.md", obj)

        log("[OK] Module A complete.")
        return 0

    except Exception as e:
        log("[ERROR] " + repr(e))
        log(traceback.format_exc())
        return 1

if __name__ == "__main__":
    raise SystemExit(main())