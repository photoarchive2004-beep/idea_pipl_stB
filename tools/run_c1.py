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
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
CROSSREF_BASE = "https://api.crossref.org/works"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self.fp.close()

    def _write(self, level: str, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)

    def err(self, msg: str) -> None:
        self._write("ERR", msg)


class ApiClient:
    def __init__(self, name: str, rps: float, timeout: int, logger: RunLogger):
        self.name = name
        self.rps = rps
        self.timeout = timeout
        self.logger = logger
        self.next_allowed = 0.0
        self.session = requests.Session()

    def _wait(self) -> None:
        if self.rps <= 0:
            return
        now = time.monotonic()
        if now < self.next_allowed:
            time.sleep(self.next_allowed - now)
        self.next_allowed = time.monotonic() + (1.0 / self.rps)

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None, retries: int = 4, retry_on_404: bool = False) -> Dict[str, Any]:
        self._wait()
        backoff = 1.0
        for i in range(retries):
            try:
                r = self.session.get(url, timeout=self.timeout, headers=headers or {})
                if r.status_code == 404 and not retry_on_404:
                    raise requests.HTTPError("HTTP 404", response=r)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                    if exc.response.status_code == 404 and not retry_on_404:
                        raise RuntimeError(f"{self.name}: HTTP 404 (нет данных)") from exc
                if i >= retries - 1:
                    raise RuntimeError(f"{self.name}: ошибка запроса: {exc}") from exc
                self.logger.warn(f"{self.name}: retry {i + 1}/{retries} после ошибки: {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 12.0)
        return {}


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
    for k in ["OPENALEX_API_KEY", "UNPAYWALL_EMAIL", "CROSSREF_MAILTO", "OPENALEX_MAILTO", "EUROPEPMC_EMAIL", "S2_API_KEY"]:
        if os.environ.get(k):
            env[k] = os.environ[k]
    if not env.get("CROSSREF_MAILTO") and env.get("UNPAYWALL_EMAIL"):
        env["CROSSREF_MAILTO"] = env["UNPAYWALL_EMAIL"]
    if not env.get("OPENALEX_MAILTO") and env.get("UNPAYWALL_EMAIL"):
        env["OPENALEX_MAILTO"] = env["UNPAYWALL_EMAIL"]
    return env


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def normalize_doi(doi: str) -> str:
    x = normalize_text(doi).lower()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x)
    return x


def slugify_id(s: str) -> str:
    x = normalize_text(s).lower()
    x = re.sub(r"https?://", "", x)
    x = re.sub(r"[^a-z0-9._-]+", "_", x)
    return x.strip("_")[:120] or "paper"


def undo_inverted_index(inv: Dict[str, List[int]]) -> str:
    pairs: List[Tuple[int, str]] = []
    for token, indexes in (inv or {}).items():
        for idx in indexes or []:
            pairs.append((int(idx), token))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return normalize_text(" ".join(tok for _, tok in pairs))


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def find_idea_dir(root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
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
    found = sorted([x for x in ideas.glob("IDEA-*") if x.is_dir()], key=lambda x: x.name, reverse=True)
    if not found:
        raise FileNotFoundError("Не найдено ни одной папки ideas/IDEA-*")
    return found[0]


def load_corpus(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def copy_to_clipboard(text: str) -> bool:
    try:
        import pyperclip  # type: ignore

        pyperclip.copy(text)
        return True
    except Exception:
        pass
    for cmd in (["clip"], ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"], ["xclip", "-selection", "clipboard"], ["pbcopy"]):
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.communicate(text.encode("utf-8"), timeout=10)
            if p.returncode == 0:
                return True
        except Exception:
            continue
    return False


def build_paper_id(row: Dict[str, str], idx: int) -> str:
    for k in ["openalex_id", "doi", "pmid", "arxiv_id"]:
        if normalize_text(row.get(k, "")):
            return slugify_id(normalize_text(row.get(k, "")))
    return f"paper_{idx+1}"


def extract_idea_context(idea_dir: Path) -> Tuple[str, Dict[str, Any], List[str]]:
    idea_text = ""
    structured: Dict[str, Any] = {}
    for p in [idea_dir / "in" / "idea.txt", idea_dir / "idea.txt"]:
        if p.exists():
            idea_text = p.read_text(encoding="utf-8", errors="replace")
            break
    sp = idea_dir / "out" / "structured_idea.json"
    if not sp.exists():
        sp = idea_dir / "in" / "llm_response.json"
    if sp.exists():
        raw = sp.read_text(encoding="utf-8", errors="replace")
        try:
            structured = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    structured = json.loads(m.group(0))
                except Exception:
                    structured = {}

    kw: List[str] = []
    for key in ["keywords", "keywords_for_search", "topics", "domains"]:
        v = structured.get(key)
        if isinstance(v, list):
            kw.extend(str(x) for x in v)
        elif isinstance(v, str):
            kw.extend(re.split(r"[,;\n]", v))
    if not kw and idea_text:
        stop = {
            "если", "чтобы", "когда", "между", "который", "которые", "также", "этого", "этой", "идея", "проект",
            "study", "using", "with", "from", "that", "this", "into", "across", "between", "paper", "research",
        }
        tokens = re.findall(r"[A-Za-zА-Яа-я0-9-]{4,}", idea_text)
        freq = Counter(t.lower() for t in tokens if t.lower() not in stop)
        kw = [w for w, _ in freq.most_common(20)]
    kw = [normalize_text(x).lower() for x in kw if normalize_text(x)]
    return idea_text.strip(), structured, list(dict.fromkeys(kw))[:20]


def get_openalex_work(row: Dict[str, str], oa: ApiClient, secrets: Dict[str, str]) -> Tuple[Dict[str, Any], str]:
    doi = normalize_doi(row.get("doi", ""))
    openalex_id = normalize_text(row.get("openalex_id", ""))
    q = []
    if secrets.get("OPENALEX_API_KEY"):
        q.append(f"api_key={requests.utils.quote(secrets['OPENALEX_API_KEY'])}")
    if secrets.get("OPENALEX_MAILTO"):
        q.append(f"mailto={requests.utils.quote(secrets['OPENALEX_MAILTO'])}")
    qs = ("?" + "&".join(q)) if q else ""
    urls: List[str] = []
    if openalex_id:
        short = openalex_id.split("/")[-1] if openalex_id.startswith("http") else openalex_id
        urls.append(f"{OPENALEX_BASE}/works/{requests.utils.quote(short, safe='')}{qs}")
    if doi:
        urls.append(f"{OPENALEX_BASE}/works/https://doi.org/{requests.utils.quote(doi, safe='')}{qs}")
    for url in urls:
        try:
            return oa.get_json(url), url
        except Exception:
            continue
    return {}, urls[0] if urls else ""


def discover_idea_topics(oa: ApiClient, keywords: List[str], secrets: Dict[str, str], logger: RunLogger) -> Dict[str, set]:
    topic_ids: set = set()
    domain_tokens: set = set()
    field_tokens: set = set()
    subfield_tokens: set = set()
    q = []
    if secrets.get("OPENALEX_API_KEY"):
        q.append(f"api_key={requests.utils.quote(secrets['OPENALEX_API_KEY'])}")
    if secrets.get("OPENALEX_MAILTO"):
        q.append(f"mailto={requests.utils.quote(secrets['OPENALEX_MAILTO'])}")
    tail = ("&" + "&".join(q)) if q else ""
    for kw in keywords[:8]:
        url = f"{OPENALEX_BASE}/topics?search={requests.utils.quote(kw)}&per-page=10{tail}"
        try:
            items = oa.get_json(url).get("results") or []
        except Exception as exc:
            logger.warn(f"Не удалось получить topics для '{kw}': {exc}")
            continue
        for t in items:
            tid = normalize_text(t.get("id", "")).split("/")[-1]
            if tid:
                topic_ids.add(tid)
            for key, bag in [("domain", domain_tokens), ("field", field_tokens), ("subfield", subfield_tokens)]:
                obj = t.get(key) or {}
                name = normalize_text(obj.get("display_name", "")).lower()
                if name:
                    bag.add(name)
    return {"topic_ids": topic_ids, "domains": domain_tokens, "fields": field_tokens, "subfields": subfield_tokens}


def compute_domain_match(work: Dict[str, Any], idea_topics: Dict[str, set]) -> Tuple[int, Dict[str, str]]:
    pt = work.get("primary_topic") or {}
    wid = normalize_text(pt.get("id", "")).split("/")[-1]
    domain = normalize_text((pt.get("domain") or {}).get("display_name", "")).lower()
    field = normalize_text((pt.get("field") or {}).get("display_name", "")).lower()
    subfield = normalize_text((pt.get("subfield") or {}).get("display_name", "")).lower()
    score = 0
    if wid and wid in idea_topics["topic_ids"]:
        score += 3
    if domain and domain in idea_topics["domains"]:
        score += 2
    if field and field in idea_topics["fields"]:
        score += 2
    if subfield and subfield in idea_topics["subfields"]:
        score += 1
    return score, {"primary_topic_id": wid, "domain": domain, "field": field, "subfield": subfield}


def preclean_corpus(rows: List[Dict[str, str]], oa: ApiClient, secrets: Dict[str, str], keywords: List[str], logger: RunLogger, max_candidates: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    after_basic: List[Dict[str, str]] = []
    rejected: List[Dict[str, Any]] = []
    doi_seen: Dict[str, str] = {}

    for i, r in enumerate(rows):
        rr = dict(r)
        rr["doi"] = normalize_doi(rr.get("doi", ""))
        rr["title"] = normalize_text(rr.get("title", ""))
        rr["openalex_id"] = normalize_text(rr.get("openalex_id", ""))
        if rr["doi"]:
            if rr["doi"] in doi_seen:
                rejected.append({"index": i + 1, "title": rr["title"], "doi": rr["doi"], "openalex_id": rr["openalex_id"], "reason": "дубликат DOI"})
                continue
            doi_seen[rr["doi"]] = rr["title"]
        if not rr["title"] or not (rr["doi"] or rr["openalex_id"]):
            rejected.append({"index": i + 1, "title": rr["title"], "doi": rr["doi"], "openalex_id": rr["openalex_id"], "reason": "нет title и/или (doi/openalex_id)"})
            continue
        after_basic.append(rr)

    idea_topics = discover_idea_topics(oa, keywords, secrets, logger) if keywords else {"topic_ids": set(), "domains": set(), "fields": set(), "subfields": set()}

    candidates: List[Dict[str, Any]] = []
    for idx, r in enumerate(after_basic):
        item = {
            "paper_id": build_paper_id(r, idx),
            "title": r.get("title", ""),
            "year": normalize_text(r.get("year", "")),
            "doi": r.get("doi", ""),
            "openalex_id": r.get("openalex_id", ""),
            "venue": normalize_text(r.get("venue", "")),
            "pmid": normalize_text(r.get("pmid", "")),
            "oa_pdf_url": normalize_text(r.get("oa_pdf_url", "")),
            "score": to_float(r.get("score", 0.0)),
            "cited_by": to_int(r.get("cited_by", 0)),
            "abstract_snippet": normalize_text((r.get("abstract") or "")[:600]),
            "first_author": normalize_text(r.get("first_author", "") or (r.get("authors", "").split(",")[0] if r.get("authors") else "")),
            "source_row": r,
        }
        domain_score = 1
        topic_info = {"primary_topic_id": "", "domain": "", "field": "", "subfield": ""}
        if idea_topics["topic_ids"] or idea_topics["domains"] or idea_topics["fields"] or idea_topics["subfields"]:
            work, _ = get_openalex_work(r, oa, secrets)
            if work:
                domain_score, topic_info = compute_domain_match(work, idea_topics)
            else:
                domain_score = 1
        item.update(topic_info)
        item["domain_match_score"] = domain_score
        if domain_score == 0:
            rejected.append({"index": idx + 1, "title": item["title"], "doi": item["doi"], "openalex_id": item["openalex_id"], "reason": "domain_match_score=0"})
            continue
        candidates.append(item)

    candidates.sort(key=lambda x: (-x["domain_match_score"], -x["score"], -x["cited_by"], x["paper_id"]))
    candidates = candidates[:max_candidates]

    stats = {
        "corpus_n": len(rows),
        "after_dedup_n": len(after_basic),
        "after_domain_filter_n": len(candidates),
        "rejected_pre_n": len(rejected),
    }
    return candidates, rejected, stats


def generate_screening_prompt(idea_text: str, structured: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> str:
    reduced_candidates = [
        {
            "paper_id": c["paper_id"],
            "title": c["title"],
            "year": c["year"],
            "doi": c["doi"],
            "venue": c["venue"],
            "abstract_snippet": c["abstract_snippet"],
            "first_author": c.get("first_author", ""),
            "downloadability_hint": c.get("downloadability_hint", "NONE"),
            "oa_pdf_candidates": c.get("oa_pdf_candidates", []),
            "primary_topic": {
                "domain": c.get("domain", ""),
                "field": c.get("field", ""),
                "subfield": c.get("subfield", ""),
            },
        }
        for c in candidates
    ]
    payload = {
        "idea_text": idea_text,
        "structured_idea": structured,
        "max_abstracts": max_abs,
        "max_pdfs": max_pdf,
        "strict_contract": {
            "selected": [
                {
                    "paper_id": "<id из candidates>",
                    "select_for_abstract": 1,
                    "select_for_pdf": 0,
                    "reason": "коротко по-русски",
                }
            ],
        },
        "rules": [
            "Верни только JSON без поясняющего текста.",
            "paper_id только из списка candidates.",
            "select_for_pdf=1 только если select_for_abstract=1.",
            "Приоритет релевантности идее + вероятности скачивания OA PDF.",
            "Для PDF в первую очередь выбирай HIGH/MED downloadability_hint (репозитории, PMC, arXiv, Zenodo и т.д.).",
            f"Не более {max_abs} с select_for_abstract=1.",
            f"Не более {max_pdf} с select_for_pdf=1.",
            "Не добавляй новые DOI/ID и не меняй paper_id.",
        ],
        "candidates": reduced_candidates,
    }
    return (
        "Ты выполняешь семантический screening научных работ по идее. "
        "Нужно строго выбрать релевантные статьи по смыслу и вернуть только JSON.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_screening_response(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("Не найден JSON-объект в ответе")
        return json.loads(m.group(0))


def validate_screening_response(data: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("selected"), list):
        return [], ["Корень ответа должен быть объектом с массивом 'selected'."]

    c_map = {c["paper_id"]: c for c in candidates}
    selected: List[Dict[str, Any]] = []
    abs_count = 0
    pdf_count = 0

    for i, row in enumerate(data.get("selected") or []):
        if not isinstance(row, dict):
            errors.append(f"selected[{i}] не объект")
            continue
        pid = normalize_text(str(row.get("paper_id", "")))
        if pid not in c_map:
            errors.append(f"selected[{i}].paper_id отсутствует в candidates: {pid}")
            continue
        a = int(row.get("select_for_abstract", 0))
        p = int(row.get("select_for_pdf", 0))
        if p == 1 and a != 1:
            errors.append(f"{pid}: select_for_pdf=1 требует select_for_abstract=1")
            continue
        if a not in (0, 1) or p not in (0, 1):
            errors.append(f"{pid}: select_for_abstract/select_for_pdf должны быть 0 или 1")
            continue
        if a == 1:
            abs_count += 1
        if p == 1:
            pdf_count += 1
        selected.append(
            {
                **c_map[pid],
                "select_for_abstract": a,
                "select_for_pdf": p,
                "reason": normalize_text(row.get("reason", "")) or "без причины",
            }
        )

    if abs_count > max_abs:
        errors.append(f"Превышен лимит abstracts: {abs_count} > {max_abs}")
    if pdf_count > max_pdf:
        errors.append(f"Превышен лимит pdfs: {pdf_count} > {max_pdf}")
    return selected, errors


def fetch_openalex_abstract(row: Dict[str, Any], oa: ApiClient, secrets: Dict[str, str]) -> Tuple[str, str]:
    work, url = get_openalex_work(row, oa, secrets)
    return undo_inverted_index(work.get("abstract_inverted_index") or {}), url


def fetch_europepmc_abstract(pmid: str, epmc: ApiClient) -> Tuple[str, str]:
    u = f"{EUROPEPMC_BASE}/search?query=EXT_ID:{requests.utils.quote(pmid)}%20AND%20SRC:MED&format=json&pageSize=1"
    j = epmc.get_json(u)
    items = (j.get("resultList") or {}).get("result") or []
    return normalize_text((items[0].get("abstractText", "") if items else "")), u


def crossref_to_text(raw: str) -> str:
    return normalize_text(re.sub(r"<[^>]+>", " ", raw or ""))


def fetch_crossref_abstract(doi: str, cr: ApiClient, mailto: str) -> Tuple[str, str]:
    h = {"User-Agent": f"IDEA_PIPELINE_C1/1.0 (mailto:{mailto})"}
    u = f"{CROSSREF_BASE}/{requests.utils.quote(doi, safe='')}?mailto={requests.utils.quote(mailto)}"
    j = cr.get_json(u, headers=h)
    return crossref_to_text((j.get("message") or {}).get("abstract", "")), u


def fetch_semantic_scholar(doi: str, title: str, s2: ApiClient, api_key: str) -> Tuple[Dict[str, Any], str]:
    headers = {"x-api-key": api_key} if api_key else {}
    if doi:
        u = f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{requests.utils.quote(doi, safe='')}?fields=abstract,openAccessPdf"
        return s2.get_json(u, headers=headers, retries=1, retry_on_404=False), u
    q = requests.utils.quote(title[:140])
    u = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={q}&limit=1&fields=abstract,openAccessPdf,title"
    j = s2.get_json(u, headers=headers, retries=1, retry_on_404=False)
    data = j.get("data") or []
    return (data[0] if data else {}), u


def fetch_unpaywall_pdf_urls(doi: str, up: ApiClient, email: str) -> Tuple[List[str], str]:
    u = f"{UNPAYWALL_BASE}/{requests.utils.quote(doi, safe='')}?email={requests.utils.quote(email)}"
    j = up.get_json(u)
    urls: List[str] = []
    best = (j.get("best_oa_location") or {}).get("url_for_pdf")
    if best:
        urls.append(best)
    for loc in j.get("oa_locations") or []:
        x = normalize_text(loc.get("url_for_pdf", ""))
        if x:
            urls.append(x)
    uniq = []
    seen = set()
    for x in urls:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq, u


def classify_pdf_url(url: str) -> str:
    u = (url or "").lower()
    high_tokens = ["pmc", "pubmedcentral", "arxiv.org", "zenodo.org", "hal.science", "repository", "eprints", "figshare", "osf.io"]
    low_tokens = ["sciencedirect", "wiley", "oup", "springer", "nature.com", "tandfonline", "linkinghub", "pdfdirect"]
    if any(t in u for t in high_tokens):
        return "HIGH"
    if any(t in u for t in low_tokens):
        return "LOW"
    if u:
        return "MED"
    return "NONE"


def sort_pdf_candidates(urls: List[str]) -> List[str]:
    priority = {"HIGH": 0, "MED": 1, "LOW": 2, "NONE": 3}
    uniq = []
    seen = set()
    for x in urls:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return sorted(uniq, key=lambda x: (priority.get(classify_pdf_url(x), 3), x))


def choose_hint(urls: List[str]) -> str:
    if not urls:
        return "NONE"
    ranks = [classify_pdf_url(x) for x in urls]
    if "HIGH" in ranks:
        return "HIGH"
    if "MED" in ranks:
        return "MED"
    if "LOW" in ranks:
        return "LOW"
    return "NONE"


def windows_open(path: Path) -> None:
    for cmd in (["explorer", str(path)], ["notepad", str(path)]):
        try:
            subprocess.Popen(cmd)
            return
        except Exception:
            continue


def build_cite_short(first_author: str, year: str, title: str) -> str:
    t = re.sub(r"\s+", " ", title).strip()[:100]
    fa = re.sub(r"[^A-Za-zА-Яа-я0-9]+", "", first_author or "Unknown") or "Unknown"
    yy = year or "0000"
    return f"{fa}_{yy}_{t}"


def fetch_europepmc_pdf_urls(pmid: str, epmc: ApiClient) -> Tuple[List[str], str]:
    u = f"{EUROPEPMC_BASE}/search?query=EXT_ID:{requests.utils.quote(pmid)}%20AND%20SRC:MED&format=json&pageSize=1"
    j = epmc.get_json(u)
    items = (j.get("resultList") or {}).get("result") or []
    urls: List[str] = []
    if items:
        for ft in items[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
            if str(ft.get("documentStyle", "")).lower() == "pdf" and ft.get("url"):
                urls.append(ft["url"])
    return urls, u


def ensure_pdf_valid(path: Path, min_size: int = 30 * 1024) -> Tuple[bool, str]:
    if not path.exists() or path.stat().st_size < min_size:
        return False, "слишком маленький файл или отсутствует"
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            return False, "файл не начинается с %PDF"
    return True, "OK"


def download_pdf(url: str, path: Path, timeout: int) -> Tuple[bool, str, int, str]:
    try:
        with requests.get(url, timeout=timeout, stream=True, allow_redirects=True) as r:
            code = r.status_code
            ctype = (r.headers.get("Content-Type", "") or "").lower()
            if code >= 400:
                return False, f"HTTP {code}", code, ctype
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                for ch in r.iter_content(chunk_size=65536):
                    if ch:
                        f.write(ch)
        ok, reason = ensure_pdf_valid(path)
        if not ok:
            path.unlink(missing_ok=True)
            return False, reason, code, ctype
        return True, "OK", code, ctype
    except Exception as exc:
        path.unlink(missing_ok=True)
        return False, str(exc), -1, ""


def save_abstract(path: Path, text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t + "\n", encoding="utf-8", newline="\n")
    return True


def pick_without_llm(candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> List[Dict[str, Any]]:
    ordered = sorted(candidates, key=lambda x: (-x["score"], -x["cited_by"], x["paper_id"]))
    abs_ids = {x["paper_id"] for x in ordered[:max_abs]}
    pdf_ids = {x["paper_id"] for x in ordered if x.get("oa_pdf_url") or x.get("doi")}
    pdf_ids = set(list(pdf_ids)[:max_pdf])
    out = []
    for c in ordered:
        out.append({**c, "select_for_abstract": 1 if c["paper_id"] in abs_ids else 0, "select_for_pdf": 1 if c["paper_id"] in pdf_ids else 0, "reason": "автовыбор без ChatGPT"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C1: Selection + Harvest (Selection + ChatGPT + Harvest)")
    ap.add_argument("--idea-dir", default="")
    ap.add_argument("--screening", choices=["chatgpt", "none"], default="chatgpt")
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--max-abstracts", type=int, default=120)
    ap.add_argument("--max-pdfs", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rps-openalex", type=float, default=2.0)
    ap.add_argument("--rps-unpaywall", type=float, default=1.5)
    ap.add_argument("--rps-crossref", type=float, default=1.0)
    ap.add_argument("--rps-europepmc", type=float, default=1.5)
    ap.add_argument("--rps-s2", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    idea_dir = find_idea_dir(root, args.idea_dir or None)
    logs_dir = idea_dir / "logs"
    in_dir = idea_dir / "in"
    papers_dir = in_dir / "papers"
    manifests_dir = papers_dir / "manifests"
    out_dir = idea_dir / "out"
    abstracts_dir = papers_dir / "abstracts"
    pdf_dir = papers_dir / "pdfs"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = logs_dir / f"moduleC1_{ts}.log"
    last_log = logs_dir / "moduleC1_LAST.log"
    logger = RunLogger(run_log)

    manifest_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    try:
        logger.info("=== Stage C1: Selection + Harvest ===")
        logger.info(f"IDEA_DIR={idea_dir}")

        secrets = load_secrets(root)
        corpus_path = out_dir / "corpus.csv"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Не найден обязательный файл: {corpus_path}")

        logger.info("[1/6] Читаю corpus.csv")
        rows = load_corpus(corpus_path)
        idea_text, structured, keywords = extract_idea_context(idea_dir)

        oa = ApiClient("OpenAlex", args.rps_openalex, args.timeout, logger)
        up = ApiClient("Unpaywall", args.rps_unpaywall, args.timeout, logger)
        cr = ApiClient("Crossref", args.rps_crossref, args.timeout, logger)
        epmc = ApiClient("EuropePMC", args.rps_europepmc, args.timeout, logger)
        s2 = ApiClient("SemanticScholar", args.rps_s2, args.timeout, logger)

        candidates, rejected, stats = preclean_corpus(rows, oa, secrets, keywords, logger, args.max_candidates)
        logger.info(f"[2/6] Дедуп/доменные фильтры: было={stats['corpus_n']} => стало={stats['after_domain_filter_n']}")
        logger.info("[3/6] Подготовка кандидатов и обогащение OA-линками")
        for c in candidates:
            urls = []
            if c.get("oa_pdf_url"):
                urls.append(c["oa_pdf_url"])
            if c.get("doi") and secrets.get("UNPAYWALL_EMAIL"):
                try:
                    xs, _ = fetch_unpaywall_pdf_urls(c["doi"], up, secrets["UNPAYWALL_EMAIL"])
                    urls.extend(xs)
                except Exception as exc:
                    error_rows.append({"paper_id": c["paper_id"], "step": "unpaywall_enrich", "url": "", "http_code": "", "message": str(exc), "doi": c.get("doi", ""), "openalex_id": c.get("openalex_id", "")})
            if c.get("pmid"):
                try:
                    xs, _ = fetch_europepmc_pdf_urls(c["pmid"], epmc)
                    urls.extend(xs)
                except Exception as exc:
                    error_rows.append({"paper_id": c["paper_id"], "step": "europepmc_enrich", "url": "", "http_code": "", "message": str(exc), "doi": c.get("doi", ""), "openalex_id": c.get("openalex_id", "")})
            c["oa_pdf_candidates"] = sort_pdf_candidates(urls)[:3]
            c["downloadability_hint"] = choose_hint(c["oa_pdf_candidates"])
            c["why_in_candidates"] = f"score={c.get('score', 0)}; cited_by={c.get('cited_by', 0)}; domain_match={c.get('domain_match_score', 0)}"

        write_json(papers_dir / "candidates.json", candidates)
        write_json(papers_dir / "preselection_candidates.json", candidates)
        write_csv(papers_dir / "preselection_rejected.csv", rejected, ["index", "title", "doi", "openalex_id", "reason"])

        c1_chat_dir = in_dir / "c1_chatgpt"
        prompt_path = c1_chat_dir / "PROMPT.txt"
        resp_path = c1_chat_dir / "RESPONSE.json"

        selected: List[Dict[str, Any]] = []
        if args.screening == "none":
            selected = pick_without_llm(candidates, args.max_abstracts, args.max_pdfs)
            logger.warn("Режим без ChatGPT включён вручную (--screening none)")
        else:
            logger.info("[4/6] ChatGPT screening")
            needs_wait = (not resp_path.exists()) or normalize_text(resp_path.read_text(encoding="utf-8", errors="replace")) in ("", "{}", '{"selected": []}')
            if needs_wait:
                c1_chat_dir.mkdir(parents=True, exist_ok=True)
                prompt = generate_screening_prompt(idea_text, structured, candidates, args.max_abstracts, args.max_pdfs)
                prompt_path.write_text(prompt, encoding="utf-8")
                if not resp_path.exists():
                    resp_path.write_text('{\n  "selected": []\n}\n', encoding="utf-8")
                copied = copy_to_clipboard(prompt)
                windows_open(c1_chat_dir)
                windows_open(prompt_path)
                windows_open(resp_path)
                logger.info(f"Prompt для ChatGPT создан: {prompt_path}")
                print("\n1) Вставьте PROMPT в ChatGPT\n2) Ответ вставьте в RESPONSE.json\n3) Запустите RUN_C1 ещё раз\n")
                if copied:
                    print("Prompt скопирован в буфер обмена.")
                return 2

            raw = parse_screening_response(resp_path)
            selected, validation_errors = validate_screening_response(raw, candidates, args.max_abstracts, args.max_pdfs)
            if validation_errors:
                invalid_path = c1_chat_dir / "RESPONSE.INVALID.json"
                invalid_path.write_text(resp_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                msg = "\n".join(f"- {e}" for e in validation_errors)
                logger.err("Невалидный JSON-ответ ChatGPT. Исправьте и запустите снова.")
                logger.err(msg)
                windows_open(resp_path)
                print(f"\nОтвет не прошёл валидацию. Копия сохранена: {invalid_path}\n{msg}\n")
                return 1

        write_csv(papers_dir / "selection.csv", selected, ["paper_id", "title", "year", "doi", "openalex_id", "first_author", "downloadability_hint", "select_for_abstract", "select_for_pdf", "reason", "domain", "field", "subfield"])
        write_jsonl(papers_dir / "papers.jsonl", selected)

        selected_abs = [x for x in selected if x["select_for_abstract"] == 1]
        selected_pdf = [x for x in selected if x["select_for_pdf"] == 1]
        if len(selected_abs) < 80:
            logger.warn(f"Выбрано мало абстрактов: {len(selected_abs)} (<80)")

        abs_ok = 0
        pdf_ok = 0
        reason_counter = Counter()

        logger.info("[5/6] Harvest abstracts")
        for i, item in enumerate(selected_abs, start=1):
            pid = item["paper_id"]
            row = item.get("source_row") or {}
            abs_path = abstracts_dir / f"{pid}.txt"
            got = False
            corpus_abs = normalize_text((row or {}).get("abstract", ""))
            if len(corpus_abs) >= 200 and save_abstract(abs_path, corpus_abs):
                got = True
                abs_ok += 1
                manifest_rows.append({"paper_id": pid, "step": "abstract", "status": "ok", "source": "corpus", "url": ""})
            if not got:
                try:
                    txt, u = fetch_openalex_abstract(item, oa, secrets)
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                        manifest_rows.append({"paper_id": pid, "step": "abstract", "status": "ok", "source": "openalex_inverted_index", "url": u})
                except Exception as exc:
                    error_rows.append({"paper_id": pid, "step": "openalex_abstract", "url": "", "http_code": "", "message": str(exc), "doi": item.get("doi", ""), "openalex_id": item.get("openalex_id", "")})
            if not got and item.get("pmid"):
                try:
                    txt, u = fetch_europepmc_abstract(item.get("pmid", ""), epmc)
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                        manifest_rows.append({"paper_id": pid, "step": "abstract", "status": "ok", "source": "europepmc", "url": u})
                except Exception as exc:
                    error_rows.append({"paper_id": pid, "step": "europepmc_abstract", "url": "", "http_code": "", "message": str(exc), "doi": item.get("doi", ""), "openalex_id": item.get("openalex_id", "")})
            if not got and item.get("doi") and secrets.get("CROSSREF_MAILTO"):
                try:
                    txt, u = fetch_crossref_abstract(item["doi"], cr, secrets["CROSSREF_MAILTO"])
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                        manifest_rows.append({"paper_id": pid, "step": "abstract", "status": "ok", "source": "crossref", "url": u})
                except Exception as exc:
                    error_rows.append({"paper_id": pid, "step": "crossref_abstract", "url": "", "http_code": "", "message": str(exc), "doi": item.get("doi", ""), "openalex_id": item.get("openalex_id", "")})
            if not got:
                reason_counter["NO_ABSTRACT"] += 1
                manifest_rows.append({"paper_id": pid, "step": "abstract", "status": "NO_ABSTRACT", "source": "none", "url": ""})
                logger.info(f"[abs {i}/{len(selected_abs)}] {pid}: NO_ABSTRACT")
            else:
                logger.info(f"[abs {i}/{len(selected_abs)}] {pid}: OK")

        manual_queue_path = papers_dir / "pdf_manual_queue.csv"
        existing_manual = list(csv.DictReader(manual_queue_path.open("r", encoding="utf-8-sig", newline=""))) if manual_queue_path.exists() else []
        manual_map = {r.get("paper_id", ""): r for r in existing_manual}
        for pid, q in list(manual_map.items()):
            if (q.get("status") or "") == "MISSING" and normalize_text(q.get("manual_pdf_local_path", "")):
                src = Path(q["manual_pdf_local_path"])
                if src.exists():
                    dst = pdf_dir / f"{pid}.pdf"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    q["status"] = "FOUND_MANUAL"

        logger.info("[6/6] Harvest PDFs + итоги")
        for i, item in enumerate(selected_pdf, start=1):
            pid = item["paper_id"]
            pdf_path = pdf_dir / f"{pid}.pdf"
            if pdf_path.exists() and not args.force:
                ok, _ = ensure_pdf_valid(pdf_path)
                if ok:
                    pdf_ok += 1
                    logger.info(f"[pdf {i}/{len(selected_pdf)}] {pid}: OK")
                    continue
            urls = sort_pdf_candidates(list(item.get("oa_pdf_candidates", [])))
            tried = []
            got_pdf = False
            for u in urls:
                tried.append(u)
                ok, msg, code, ctype = download_pdf(u, pdf_path, args.timeout)
                manifest_rows.append({"paper_id": pid, "step": "download_pdf", "status": "ok" if ok else "failed", "source": classify_pdf_url(u), "url": u, "http_code": code, "content_type": ctype})
                if ok:
                    got_pdf = True
                    pdf_ok += 1
                    logger.info(f"[pdf {i}/{len(selected_pdf)}] {pid}: OK")
                    break
                err_code = f"ERROR_{code}" if code in (401, 403) else "NO_OA_PDF"
                error_rows.append({"paper_id": pid, "step": "download_pdf", "url": u, "http_code": code, "message": msg, "doi": item.get("doi", ""), "openalex_id": item.get("openalex_id", "")})
                logger.info(f"[pdf {i}/{len(selected_pdf)}] {pid}: {err_code}")
            if not got_pdf:
                reason_counter["NO_OA_PDF"] += 1
                manual_map[pid] = {
                    "cite_short": build_cite_short(item.get("first_author", ""), item.get("year", ""), item.get("title", "")),
                    "paper_id": pid,
                    "doi": item.get("doi", ""),
                    "title": item.get("title", ""),
                    "year": item.get("year", ""),
                    "first_author": item.get("first_author", ""),
                    "best_oa_urls_tried": " | ".join(tried[:3]),
                    "manual_pdf_local_path": manual_map.get(pid, {}).get("manual_pdf_local_path", ""),
                    "status": "MISSING",
                    "notes": "403 publisher" if any("403" in str(e.get("http_code", "")) for e in error_rows if e.get("paper_id") == pid) else "no_oa_location",
                }

        write_csv(manual_queue_path, list(manual_map.values()), ["cite_short", "paper_id", "doi", "title", "year", "first_author", "best_oa_urls_tried", "manual_pdf_local_path", "status", "notes"])
        (papers_dir / "pdf_manual_instructions.md").write_text(
            "# Ручная очередь PDF\n\n1. Скачайте PDF вручную в любое место.\n2. Заполните столбец `manual_pdf_local_path` в `pdf_manual_queue.csv`.\n3. Запустите RUN_C1 снова — файл будет автоматически скопирован в in/papers/pdfs/<paper_id>.pdf и статус станет FOUND_MANUAL.\n",
            encoding="utf-8",
        )

        write_jsonl(manifests_dir / "manifest.jsonl", manifest_rows)
        write_csv(manifests_dir / "errors.csv", error_rows, ["paper_id", "step", "url", "http_code", "message", "doi", "openalex_id"])

        status = "OK" if not error_rows else "DEGRADED"
        top5 = reason_counter.most_common(5)
        top_txt = "\n".join([f"- {k}: {v}" for k, v in top5]) if top5 else "- нет"

        report = (
            "# Stage C1 — Harvest report\n\n"
            f"Статус: **{status}**\n\n"
            f"- corpus: {stats['corpus_n']}\n"
            f"- after dedup: {stats['after_dedup_n']}\n"
            f"- after domain filter: {stats['after_domain_filter_n']}\n"
            f"- candidates sent to ChatGPT: {len(candidates)}\n"
            f"- selected abstracts: {len(selected_abs)}\n"
            f"- selected pdfs: {len(selected_pdf)}\n"
            f"- downloaded abstracts_ok: {abs_ok}\n"
            f"- downloaded pdfs_ok: {pdf_ok}\n"
            f"- errors total: {len(error_rows)}\n\n"
            "## Top-5 причин ошибок\n"
            f"{top_txt}\n"
        )
        (out_dir / "harvest_report.md").write_text(report, encoding="utf-8")

        prisma = (
            "# PRISMA C1\n\n"
            f"- status: {status}\n"
            f"- corpus: {stats['corpus_n']}\n"
            f"- after_dedup: {stats['after_dedup_n']}\n"
            f"- after_domain_filter: {stats['after_domain_filter_n']}\n"
            f"- selected_for_abstract: {len(selected_abs)}\n"
            f"- selected_for_pdf: {len(selected_pdf)}\n"
            f"- downloaded_abstract_ok: {abs_ok}\n"
            f"- downloaded_pdf_ok: {pdf_ok}\n"
            f"- errors_total: {len(error_rows)}\n"
        )
        (out_dir / "prisma_c1.md").write_text(prisma, encoding="utf-8")

        summary = (
            "\nИТОГ Stage C1\n"
            f"- selected abstracts N: {len(selected_abs)}\n"
            f"- selected pdfs N: {len(selected_pdf)}\n"
            f"- downloaded abstracts_ok: {abs_ok}\n"
            f"- downloaded pdfs_ok: {pdf_ok}\n"
            f"- top-5 error reasons:\n{top_txt}\n"
            f"- errors.csv: {manifests_dir / 'errors.csv'}\n"
            f"- LAST.log: {last_log}\n"
            f"- статус: {status}\n"
        )
        print(summary)

        (root / "launcher_logs").mkdir(exist_ok=True)
        (root / "launcher_logs" / "LAST_LOG.txt").write_text(f"Stage C1\nIDEA={idea_dir}\nLOG={last_log}\nERRORS={manifests_dir / 'errors.csv'}\n", encoding="utf-8")
        shutil.copyfile(run_log, last_log)
        logger.info(f"Stage C1 завершен. Статус={status}")
        return 0

    except Exception as exc:
        logger.err(f"Критическая ошибка Stage C1: {exc}")
        manifests_dir.mkdir(parents=True, exist_ok=True)
        if not (manifests_dir / "errors.csv").exists():
            write_csv(manifests_dir / "errors.csv", [{"paper_id": "", "step": "fatal", "url": "", "http_code": "", "message": str(exc), "doi": "", "openalex_id": ""}], ["paper_id", "step", "url", "http_code", "message", "doi", "openalex_id"])
        return 1
    finally:
        try:
            shutil.copyfile(run_log, last_log)
        except Exception:
            pass
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
