#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
CROSSREF_BASE = "https://api.crossref.org/works"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self.fp.close()

    def _write(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)

    def err(self, msg: str) -> None:
        self._write("ERR", msg)


def read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_secrets(root: Path) -> Dict[str, str]:
    env = read_env_file(root / "config" / "secrets.env")
    for k in ["OPENALEX_API_KEY", "UNPAYWALL_EMAIL", "CROSSREF_MAILTO", "OPENALEX_MAILTO", "EUROPEPMC_EMAIL"]:
        if os.environ.get(k):
            env[k] = os.environ[k]
    if not env.get("CROSSREF_MAILTO") and env.get("UNPAYWALL_EMAIL"):
        env["CROSSREF_MAILTO"] = env["UNPAYWALL_EMAIL"]
    return env


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def slugify_id(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_")[:120] or "paper"


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def undo_inverted_index(inv: Dict[str, List[int]]) -> str:
    if not inv:
        return ""
    pairs: List[Tuple[int, str]] = []
    for token, pos_list in inv.items():
        for pos in pos_list or []:
            pairs.append((int(pos), token))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return normalize_text(" ".join(t for _, t in pairs))


def find_idea_dir(root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Папка идеи не найдена: {p}")
        return p
    ideas = root / "ideas"
    active = ideas / "_ACTIVE_PATH.txt"
    if active.exists():
        raw = active.read_text(encoding="utf-8", errors="replace").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (root / raw).resolve()
            if p.exists():
                return p
    candidates = sorted([x for x in ideas.glob("IDEA-*") if x.is_dir()], key=lambda x: x.name, reverse=True)
    if not candidates:
        raise FileNotFoundError("Не найдено ни одной папки ideas/IDEA-*")
    return candidates[0]


@dataclass
class ApiClient:
    name: str
    rps: float
    timeout: int
    logger: RunLogger

    def __post_init__(self) -> None:
        self._next_allowed = 0.0
        self.session = requests.Session()

    def _wait_rate(self) -> None:
        if self.rps <= 0:
            return
        now = time.monotonic()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
        self._next_allowed = time.monotonic() + 1.0 / self.rps

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None, retries: int = 4) -> Dict[str, Any]:
        self._wait_rate()
        backoff = 1.0
        h = headers or {}
        for attempt in range(1, retries + 1):
            try:
                r = self.session.get(url, headers=h, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "-")
                if attempt >= retries:
                    raise RuntimeError(f"{self.name}: ошибка запроса ({status}): {exc}") from exc
                self.logger.warn(f"{self.name}: повтор {attempt}/{retries} после ошибки ({status}): {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 12.0)
        return {}


def extract_keywords(structured_path: Path) -> List[str]:
    if not structured_path.exists():
        return []
    try:
        data = json.loads(structured_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    raw: List[str] = []
    keys = data.get("keywords_for_search") or data.get("keywords") or []
    if isinstance(keys, list):
        raw.extend(str(x) for x in keys)
    elif isinstance(keys, str):
        raw.extend(re.split(r"[,;\n]", keys))
    return [normalize_text(x).lower() for x in raw if normalize_text(x)]


def keyword_overlap(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw and kw in t)


def build_paper_id(row: Dict[str, str], idx: int) -> str:
    for k in ["openalex_id", "doi", "pmid", "arxiv_id"]:
        if row.get(k):
            return slugify_id(str(row[k]))
    title = slugify_id(row.get("title", "")[:80])
    year = row.get("year", "")
    return f"{title}_{year}_{idx+1}"


def load_corpus(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def choose_selection(rows: List[Dict[str, str]], keywords: List[str], max_abstracts: int, max_pdfs: int) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        score = to_float(row.get("score", 0.0), 0.0)
        cited_by = to_int(row.get("cited_by", 0), 0)
        doi = normalize_text(row.get("doi", ""))
        oa_pdf = normalize_text(row.get("oa_pdf_url", ""))
        title = normalize_text(row.get("title", ""))
        abstract = normalize_text(row.get("abstract", ""))
        hay = " ".join([title, abstract, row.get("topics", ""), row.get("concepts", "")])
        kw_hits = keyword_overlap(hay, keywords)

        priority = score
        reasons: List[str] = []
        if score > 0:
            reasons.append("высокий score")
        if doi:
            priority += 1.2
            reasons.append("есть DOI")
        if oa_pdf:
            priority += 1.2
            reasons.append("есть OA PDF")
        if cited_by > 0:
            priority += min(cited_by / 200.0, 2.5)
            if cited_by >= 20:
                reasons.append("высокие цитирования")
        if kw_hits > 0:
            priority += min(kw_hits * 0.8, 3.0)
            reasons.append("совпадение по ключевым словам")

        prepared.append(
            {
                "row": row,
                "paper_id": build_paper_id(row, i),
                "priority": priority,
                "kw_hits": kw_hits,
                "reason": "; ".join(dict.fromkeys(reasons)) if reasons else "базовый отбор",
                "pdf_candidate": bool(oa_pdf or doi or row.get("pmid")),
            }
        )

    prepared.sort(key=lambda x: (-x["priority"], x["paper_id"]))
    abs_ids = {x["paper_id"] for x in prepared[:max_abstracts]}

    pdf_sorted = sorted([x for x in prepared if x["pdf_candidate"]], key=lambda x: (-x["priority"], x["paper_id"]))
    pdf_ids = {x["paper_id"] for x in pdf_sorted[:max_pdfs]}

    result: List[Dict[str, Any]] = []
    for item in prepared:
        row = item["row"]
        result.append(
            {
                "paper_id": item["paper_id"],
                "doi": normalize_text(row.get("doi", "")),
                "openalex_id": normalize_text(row.get("openalex_id", "")),
                "title": normalize_text(row.get("title", "")),
                "year": normalize_text(row.get("year", "")),
                "select_for_abstract": 1 if item["paper_id"] in abs_ids else 0,
                "select_for_pdf": 1 if item["paper_id"] in pdf_ids else 0,
                "reason": item["reason"],
                "priority": round(item["priority"], 4),
                "source_row": row,
            }
        )
    return result


def write_selection_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "paper_id",
        "doi",
        "openalex_id",
        "title",
        "year",
        "select_for_abstract",
        "select_for_pdf",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def ensure_pdf_valid(path: Path, min_size: int = 30 * 1024) -> Tuple[bool, str]:
    if not path.exists() or path.stat().st_size < min_size:
        return False, "слишком маленький файл или отсутствует"
    with path.open("rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        return False, "файл не начинается с %PDF"
    return True, "OK"


def save_abstract(path: Path, text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t + "\n", encoding="utf-8", newline="\n")
    return True


def crossref_to_text(raw: str) -> str:
    t = re.sub(r"<[^>]+>", " ", raw or "")
    return normalize_text(t)


def fetch_openalex_abstract(row: Dict[str, str], oa: ApiClient, api_key: str, mailto: str) -> Tuple[str, str, str]:
    openalex_id = normalize_text(row.get("openalex_id", ""))
    doi = normalize_text(row.get("doi", "")).replace("https://doi.org/", "")
    params = []
    if api_key:
        params.append(f"api_key={requests.utils.quote(api_key)}")
    if mailto:
        params.append(f"mailto={requests.utils.quote(mailto)}")
    q = ("?" + "&".join(params)) if params else ""

    urls: List[str] = []
    if openalex_id:
        if openalex_id.startswith("http"):
            urls.append(f"{OPENALEX_BASE}/works/{requests.utils.quote(openalex_id.split('/')[-1], safe='')}{q}")
        else:
            urls.append(f"{OPENALEX_BASE}/works/{requests.utils.quote(openalex_id, safe='')}{q}")
    if doi:
        urls.append(f"{OPENALEX_BASE}/works/https://doi.org/{requests.utils.quote(doi, safe='')}{q}")

    for u in urls:
        j = oa.get_json(u)
        text = undo_inverted_index(j.get("abstract_inverted_index") or {})
        if text:
            return text, "openalex_inverted_index", u
    return "", "", urls[0] if urls else ""


def fetch_europepmc_abstract(pmid: str, epmc: ApiClient) -> Tuple[str, str]:
    u = f"{EUROPEPMC_BASE}/search?query=EXT_ID:{requests.utils.quote(pmid)}%20AND%20SRC:MED&format=json&pageSize=1"
    j = epmc.get_json(u)
    results = (j.get("resultList") or {}).get("result") or []
    if results:
        return normalize_text(results[0].get("abstractText", "")), u
    return "", u


def fetch_crossref_abstract(doi: str, cr: ApiClient, mailto: str) -> Tuple[str, str]:
    headers = {"User-Agent": f"IDEA_PIPELINE_C1/1.0 (mailto:{mailto})"}
    u = f"{CROSSREF_BASE}/{requests.utils.quote(doi, safe='')}?mailto={requests.utils.quote(mailto)}"
    j = cr.get_json(u, headers=headers)
    raw = ((j.get("message") or {}).get("abstract") or "")
    return crossref_to_text(raw), u


def fetch_unpaywall_pdf_urls(doi: str, up: ApiClient, email: str) -> Tuple[List[str], str, str]:
    u = f"{UNPAYWALL_BASE}/{requests.utils.quote(doi, safe='')}?email={requests.utils.quote(email)}"
    j = up.get_json(u)
    urls: List[str] = []
    best_url = ""
    best = j.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        best_url = best["url_for_pdf"]
        urls.append(best_url)
    for loc in j.get("oa_locations") or []:
        x = normalize_text(loc.get("url_for_pdf", ""))
        if x:
            urls.append(x)
    seen = set()
    uniq = []
    for x in urls:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq, u, best_url


def fetch_europepmc_pdf_urls(pmid: str, epmc: ApiClient) -> Tuple[List[str], str]:
    u = f"{EUROPEPMC_BASE}/search?query=EXT_ID:{requests.utils.quote(pmid)}%20AND%20SRC:MED&format=json&pageSize=1"
    j = epmc.get_json(u)
    results = (j.get("resultList") or {}).get("result") or []
    urls: List[str] = []
    if results:
        for ft in results[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
            if str(ft.get("documentStyle", "")).lower() == "pdf" and ft.get("url"):
                urls.append(ft["url"])
    return urls, u


def download_pdf(url: str, path: Path, timeout: int, logger: RunLogger, min_size: int = 30 * 1024) -> Tuple[bool, str, int, str]:
    try:
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=True) as r:
            code = r.status_code
            if code >= 400:
                return False, f"HTTP {code}", code, r.headers.get("Content-Type", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            ctype = (r.headers.get("Content-Type", "") or "").lower()
            ok_magic, reason_magic = ensure_pdf_valid(path, min_size=min_size)
            if "pdf" not in ctype and not ok_magic:
                path.unlink(missing_ok=True)
                return False, f"не PDF: {reason_magic}; Content-Type={ctype}", code, ctype
            if not ok_magic:
                path.unlink(missing_ok=True)
                return False, reason_magic, code, ctype
            return True, "OK", code, ctype
    except Exception as exc:
        path.unlink(missing_ok=True)
        logger.warn(f"Скачивание PDF не удалось: {exc}")
        return False, str(exc), -1, ""


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_errors_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["paper_id", "step", "url", "http_code", "message", "doi", "openalex_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C1: Selection + Harvest (abstracts + OA PDFs)")
    ap.add_argument("--idea-dir", default="", help="Путь к IDEA-* (опционально)")
    ap.add_argument("--max-abstracts", type=int, default=120)
    ap.add_argument("--max-pdfs", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--rps-openalex", type=float, default=2.0)
    ap.add_argument("--rps-unpaywall", type=float, default=1.5)
    ap.add_argument("--rps-crossref", type=float, default=1.0)
    ap.add_argument("--rps-europepmc", type=float, default=1.5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    idea_dir = find_idea_dir(root, args.idea_dir or None)

    logs_dir = idea_dir / "logs"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"moduleC1_{now}.log"
    logger = RunLogger(log_path)

    try:
        logger.info("=== Stage C1: Отбор и сбор (абстракты + OA PDF) ===")
        logger.info(f"IDEA_DIR={idea_dir}")

        secrets = load_secrets(root)

        out_dir = idea_dir / "out"
        in_papers = idea_dir / "in" / "papers"
        abstracts_dir = in_papers / "abstracts"
        pdfs_dir = in_papers / "pdfs"
        manifests_dir = in_papers / "manifests"

        corpus_path = out_dir / "corpus.csv"
        structured_path = out_dir / "structured_idea.json"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Не найден обязательный файл: {corpus_path}")

        rows = load_corpus(corpus_path)
        logger.info(f"Найдено записей в корпусе: {len(rows)}")

        keywords = extract_keywords(structured_path)
        if keywords:
            logger.info(f"Загружены ключевые слова из structured_idea.json: {len(keywords)}")
        else:
            logger.warn("structured_idea.json не найден/пустой: отбор только по score/cited_by/OA")

        selected = choose_selection(rows, keywords, args.max_abstracts, args.max_pdfs)
        write_selection_csv(in_papers / "selection.csv", selected)

        selected_abs = [x for x in selected if x["select_for_abstract"] == 1]
        selected_pdf = [x for x in selected if x["select_for_pdf"] == 1]
        logger.info(f"Выбрано: abstracts={len(selected_abs)}, pdfs={len(selected_pdf)}")

        if args.dry_run:
            logger.info("Режим --dry-run: сетевые запросы пропущены.")
            summary = (
                f"ИТОГ C1 (dry-run)\n"
                f"- найдено в корпусе: {len(rows)}\n"
                f"- выбрано abstracts: {len(selected_abs)}\n"
                f"- выбрано pdfs: {len(selected_pdf)}\n"
                f"- selection: {in_papers / 'selection.csv'}\n"
            )
            print(summary)
            (out_dir / "harvest_report.md").write_text("# Stage C1 (dry-run)\n\n" + summary, encoding="utf-8")
            (out_dir / "prisma_c1.md").write_text(
                f"# PRISMA C1 (dry-run)\n\n- selected_for_abstract: {len(selected_abs)}\n- selected_for_pdf: {len(selected_pdf)}\n",
                encoding="utf-8",
            )
            shutil.copyfile(log_path, logs_dir / "moduleC1_LAST.log")
            return 0

        oa = ApiClient("OpenAlex", args.rps_openalex, args.timeout, logger)
        up = ApiClient("Unpaywall", args.rps_unpaywall, args.timeout, logger)
        cr = ApiClient("Crossref", args.rps_crossref, args.timeout, logger)
        epmc = ApiClient("EuropePMC", args.rps_europepmc, args.timeout, logger)

        papers_rows: List[Dict[str, Any]] = []
        manifest_rows: List[Dict[str, Any]] = []
        errors_rows: List[Dict[str, Any]] = []

        abs_ok = 0
        pdf_ok = 0
        reason_counter = Counter()

        by_id = {x["paper_id"]: x for x in selected}

        for item in selected:
            row = item["source_row"]
            pid = item["paper_id"]
            abs_path = abstracts_dir / f"{pid}.txt"
            pdf_path = pdfs_dir / f"{pid}.pdf"

            paper_state: Dict[str, Any] = {
                "paper_id": pid,
                "doi": normalize_text(row.get("doi", "")),
                "openalex_id": normalize_text(row.get("openalex_id", "")),
                "pmid": normalize_text(row.get("pmid", "")),
                "title": normalize_text(row.get("title", "")),
                "year": normalize_text(row.get("year", "")),
                "select_for_abstract": item["select_for_abstract"],
                "select_for_pdf": item["select_for_pdf"],
                "reason": item["reason"],
                "source_abstract": "",
                "abstract_status": "not_selected",
                "pdf_status": "not_selected",
            }

            # ---------- ABSTRACT ----------
            if item["select_for_abstract"] == 1:
                paper_state["abstract_status"] = "failed"
                if abs_path.exists() and abs_path.stat().st_size > 10 and not args.force:
                    abs_ok += 1
                    paper_state["source_abstract"] = "resume_cached"
                    paper_state["abstract_status"] = "ok"
                else:
                    corpus_abs = normalize_text(row.get("abstract", ""))
                    if len(corpus_abs) > 200:
                        if save_abstract(abs_path, corpus_abs):
                            abs_ok += 1
                            paper_state["source_abstract"] = "corpus"
                            paper_state["abstract_status"] = "ok"
                    else:
                        try:
                            txt, src, used_url = fetch_openalex_abstract(
                                row,
                                oa,
                                secrets.get("OPENALEX_API_KEY", ""),
                                secrets.get("OPENALEX_MAILTO", ""),
                            )
                            if txt and save_abstract(abs_path, txt):
                                abs_ok += 1
                                paper_state["source_abstract"] = src
                                paper_state["abstract_status"] = "ok"
                            else:
                                raise RuntimeError("в OpenAlex нет abstract_inverted_index")
                        except Exception as exc:
                            errors_rows.append(
                                {
                                    "paper_id": pid,
                                    "step": "openalex",
                                    "url": locals().get("used_url", ""),
                                    "http_code": "",
                                    "message": f"{exc}",
                                    "doi": row.get("doi", ""),
                                    "openalex_id": row.get("openalex_id", ""),
                                }
                            )
                            pmid = normalize_text(row.get("pmid", ""))
                            got_abs = False
                            if pmid:
                                try:
                                    txt, epmc_url = fetch_europepmc_abstract(pmid, epmc)
                                    if txt and save_abstract(abs_path, txt):
                                        abs_ok += 1
                                        paper_state["source_abstract"] = "europepmc"
                                        paper_state["abstract_status"] = "ok"
                                        got_abs = True
                                    else:
                                        raise RuntimeError("Europe PMC не вернул abstract")
                                except Exception as exc2:
                                    errors_rows.append(
                                        {
                                            "paper_id": pid,
                                            "step": "europepmc",
                                            "url": locals().get("epmc_url", ""),
                                            "http_code": "",
                                            "message": f"{exc2}",
                                            "doi": row.get("doi", ""),
                                            "openalex_id": row.get("openalex_id", ""),
                                        }
                                    )
                            if (not got_abs) and normalize_text(row.get("doi", "")) and secrets.get("CROSSREF_MAILTO"):
                                try:
                                    txt, cr_url = fetch_crossref_abstract(normalize_text(row.get("doi", "")), cr, secrets["CROSSREF_MAILTO"])
                                    if txt and save_abstract(abs_path, txt):
                                        abs_ok += 1
                                        paper_state["source_abstract"] = "crossref"
                                        paper_state["abstract_status"] = "ok"
                                    else:
                                        raise RuntimeError("Crossref не содержит abstract")
                                except Exception as exc3:
                                    errors_rows.append(
                                        {
                                            "paper_id": pid,
                                            "step": "crossref",
                                            "url": locals().get("cr_url", ""),
                                            "http_code": "",
                                            "message": f"{exc3}",
                                            "doi": row.get("doi", ""),
                                            "openalex_id": row.get("openalex_id", ""),
                                        }
                                    )
                if paper_state["abstract_status"] != "ok":
                    reason_counter["не удалось получить абстракт"] += 1

            # ---------- PDF ----------
            if item["select_for_pdf"] == 1:
                paper_state["pdf_status"] = "failed"
                if pdf_path.exists() and not args.force:
                    ok_pdf, _ = ensure_pdf_valid(pdf_path)
                    if ok_pdf:
                        pdf_ok += 1
                        paper_state["pdf_status"] = "ok"
                        manifest_rows.append({"paper_id": pid, "step": "download_pdf", "status": "resume_cached", "url": str(pdf_path), "http_code": 200})
                if paper_state["pdf_status"] != "ok":
                    pdf_candidates: List[Tuple[str, str]] = []
                    c_oa = normalize_text(row.get("oa_pdf_url", ""))
                    if c_oa:
                        pdf_candidates.append(("corpus_oa_pdf_url", c_oa))

                    doi = normalize_text(row.get("doi", ""))
                    pmid = normalize_text(row.get("pmid", ""))
                    best_html = normalize_text(row.get("best_url", ""))
                    if doi and secrets.get("UNPAYWALL_EMAIL"):
                        try:
                            urls, up_url, best = fetch_unpaywall_pdf_urls(doi, up, secrets["UNPAYWALL_EMAIL"])
                            for u in urls:
                                pdf_candidates.append(("unpaywall", u))
                            if best and not best_html:
                                best_html = best
                        except Exception as exc:
                            errors_rows.append(
                                {"paper_id": pid, "step": "unpaywall", "url": locals().get("up_url", ""), "http_code": "", "message": str(exc), "doi": doi, "openalex_id": row.get("openalex_id", "")}
                            )
                    if pmid:
                        try:
                            epmc_urls, epmc_url2 = fetch_europepmc_pdf_urls(pmid, epmc)
                            for u in epmc_urls:
                                pdf_candidates.append(("europepmc", u))
                        except Exception as exc:
                            errors_rows.append(
                                {"paper_id": pid, "step": "europepmc_pdf", "url": locals().get("epmc_url2", ""), "http_code": "", "message": str(exc), "doi": doi, "openalex_id": row.get("openalex_id", "")}
                            )

                    seen = set()
                    uniq_candidates = []
                    for src, u in pdf_candidates:
                        if u and u not in seen:
                            seen.add(u)
                            uniq_candidates.append((src, u))

                    downloaded = False
                    for src, u in uniq_candidates:
                        ok, msg, http_code, ctype = download_pdf(u, pdf_path, args.timeout, logger)
                        manifest_rows.append(
                            {
                                "paper_id": pid,
                                "step": "download_pdf",
                                "status": "ok" if ok else "failed",
                                "source": src,
                                "url": u,
                                "http_code": http_code,
                                "content_type": ctype,
                                "size": pdf_path.stat().st_size if pdf_path.exists() else 0,
                                "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest() if ok and pdf_path.exists() else "",
                            }
                        )
                        if ok:
                            downloaded = True
                            pdf_ok += 1
                            paper_state["pdf_status"] = "ok"
                            break
                        errors_rows.append(
                            {
                                "paper_id": pid,
                                "step": "download_pdf",
                                "url": u,
                                "http_code": http_code,
                                "message": msg,
                                "doi": doi,
                                "openalex_id": row.get("openalex_id", ""),
                            }
                        )
                    if not downloaded:
                        reason_counter["не удалось скачать OA PDF"] += 1
                        manifest_rows.append({"paper_id": pid, "step": "best_url", "status": "saved", "url": best_html, "http_code": ""})

            papers_rows.append(paper_state)

        write_jsonl(in_papers / "papers.jsonl", papers_rows)
        write_jsonl(manifests_dir / "manifest.jsonl", manifest_rows)
        write_errors_csv(manifests_dir / "errors.csv", errors_rows)

        fail_top = reason_counter.most_common(5)
        fail_lines = [f"- {k}: {v}" for k, v in fail_top] if fail_top else ["- нет"]

        report = (
            "# Stage C1 — Отчёт по сбору\n\n"
            f"- Найдено в корпусе: **{len(rows)}**\n"
            f"- Выбрано abstracts: **{len(selected_abs)}**\n"
            f"- Выбрано pdfs: **{len(selected_pdf)}**\n"
            f"- Успешно abstracts: **{abs_ok}**\n"
            f"- Успешно pdfs: **{pdf_ok}**\n"
            f"- Ошибок: **{len(errors_rows)}**\n\n"
            "## Топ причин\n"
            + "\n".join(fail_lines)
            + "\n"
        )
        (out_dir / "harvest_report.md").write_text(report, encoding="utf-8")

        prisma = (
            "# PRISMA C1 (кратко)\n\n"
            f"- selected_for_abstract: {len(selected_abs)}\n"
            f"- selected_for_pdf: {len(selected_pdf)}\n"
            f"- downloaded_abstract_ok: {abs_ok}\n"
            f"- downloaded_pdf_ok: {pdf_ok}\n"
            f"- failed_total: {len(errors_rows)}\n"
        )
        (out_dir / "prisma_c1.md").write_text(prisma, encoding="utf-8")

        summary_txt = (
            "ИТОГ Stage C1\n"
            f"- найдено в корпусе: {len(rows)}\n"
            f"- выбрано: abstracts={len(selected_abs)}, pdfs={len(selected_pdf)}\n"
            f"- скачано успешно: abstracts_ok={abs_ok}, pdfs_ok={pdf_ok}\n"
            f"- главные причины провалов:\n" + "\n".join(fail_lines) + "\n"
            f"- LAST_LOG: {logs_dir / 'moduleC1_LAST.log'}\n"
            f"- errors.csv: {manifests_dir / 'errors.csv'}\n"
        )
        print(summary_txt)

        shutil.copyfile(log_path, logs_dir / "moduleC1_LAST.log")

        launcher_dir = root / "launcher_logs"
        launcher_dir.mkdir(exist_ok=True)
        (launcher_dir / "LAST_LOG.txt").write_text(
            f"Stage C1\nIDEA={idea_dir}\nLOG={logs_dir / 'moduleC1_LAST.log'}\nERRORS={manifests_dir / 'errors.csv'}\n",
            encoding="utf-8",
        )

        logger.info("Stage C1 завершён.")
        return 0
    except Exception as exc:
        logger.err(str(exc))
        try:
            shutil.copyfile(log_path, logs_dir / "moduleC1_LAST.log")
        except Exception:
            pass
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
