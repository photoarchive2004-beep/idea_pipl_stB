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
import socket
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
    def __init__(self, name: str, rps: float, timeout: Tuple[int, int], logger: RunLogger):
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


def _short_structured_idea(structured: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["keywords_for_search", "claims"]:
        val = structured.get(key)
        if isinstance(val, list):
            out[key] = [normalize_text(str(x)) for x in val if normalize_text(str(x))][:12]
        elif isinstance(val, str) and normalize_text(val):
            out[key] = normalize_text(val)
    return out


def generate_light_screening_prompt(idea_text: str, structured: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> str:
    domain_counter = Counter(normalize_text(c.get("domain", "")).lower() for c in candidates if normalize_text(c.get("domain", "")))
    field_counter = Counter(normalize_text(c.get("field", "")).lower() for c in candidates if normalize_text(c.get("field", "")))
    payload = {
        "idea_text": normalize_text(idea_text)[:4000],
        "structured_idea_short": _short_structured_idea(structured),
        "candidates_stats": {
            "total_candidates": len(candidates),
            "top_domains": domain_counter.most_common(10),
            "top_fields": field_counter.most_common(10),
        },
        "required_response_contract": {
            "include_keywords": ["..."],
            "exclude_keywords": ["..."],
            "allowed_domains": ["..."] ,
            "blocked_domains": ["..."],
            "target_abstracts": max_abs,
            "target_pdfs": max_pdf,
            "notes": "optional"
        },
        "rules": [
            "Верни только JSON-объект без markdown.",
            "Не возвращай paper_id и списки статей: только правила фильтрации.",
            "include_keywords: 5-20, exclude_keywords: 5-30.",
            f"target_abstracts <= {max_abs}, target_pdfs <= {max_pdf}, target_pdfs <= target_abstracts.",
            "Учитывай неоднозначные аббревиатуры и добавляй исключающие keywords/domain для нерелевантных значений.",
        ],
    }
    return "LIGHT screening для C1: сформируй фильтры для автоматического отбора.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_full_screening_prompt(idea_text: str, structured: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> str:
    reduced_candidates = [{
        "paper_id": c["paper_id"],
        "title": c["title"],
        "year": c["year"],
        "doi": c["doi"],
        "venue": c["venue"],
        "first_author": c.get("first_author", ""),
        "abstract_snippet": c["abstract_snippet"],
        "domain": c.get("domain", ""),
        "field": c.get("field", ""),
        "subfield": c.get("subfield", ""),
    } for c in candidates]
    payload = {
        "idea_text": idea_text,
        "structured_idea": structured,
        "max_abstracts": max_abs,
        "max_pdfs": max_pdf,
        "strict_response_contract": {
            "selected": [{"paper_id": "<id из candidates>", "select_for_abstract": 1, "select_for_pdf": 0, "tier": "core|support|background", "reason": "коротко"}],
            "notes": "optional"
        },
        "candidates": reduced_candidates,
    }
    return "FULL screening C1: верни JSON по контракту с selected[].\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_screening_response(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("Не найден JSON-объект в ответе")
        return json.loads(m.group(0))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zа-я0-9-]{3,}", normalize_text(text).lower())


def apply_light_screening(data: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return [], ["Корень ответа должен быть JSON-объектом."]
    include = [normalize_text(str(x)).lower() for x in (data.get("include_keywords") or []) if normalize_text(str(x))]
    exclude = [normalize_text(str(x)).lower() for x in (data.get("exclude_keywords") or []) if normalize_text(str(x))]
    allowed_domains = {normalize_text(str(x)).lower() for x in (data.get("allowed_domains") or []) if normalize_text(str(x))}
    blocked_domains = {normalize_text(str(x)).lower() for x in (data.get("blocked_domains") or []) if normalize_text(str(x))}
    if len(include) < 1:
        errors.append("include_keywords пустой.")
    target_abs = max(1, min(max_abs, to_int(data.get("target_abstracts", max_abs), max_abs)))
    target_pdf = max(0, min(max_pdf, to_int(data.get("target_pdfs", max_pdf), max_pdf), target_abs))

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        text = " ".join([c.get("title", ""), c.get("abstract_snippet", ""), c.get("venue", ""), c.get("domain", ""), c.get("field", ""), c.get("subfield", "")]).lower()
        domain = normalize_text(c.get("domain", "")).lower()
        if allowed_domains and domain and domain not in allowed_domains:
            continue
        if domain and domain in blocked_domains:
            continue
        score = 0.0
        for kw in include:
            if kw and kw in text:
                score += 2.0
                score += 0.2 * text.count(kw)
        excluded = False
        for kw in exclude:
            if kw and kw in text:
                score -= 4.0
                excluded = True
        if excluded and score < 0:
            continue
        score += 1.5 * c.get("domain_match_score", 0)
        c2 = dict(c)
        c2["relevance_score"] = round(score, 4)
        scored.append(c2)
    scored.sort(key=lambda x: (-to_float(x.get("relevance_score", 0.0)), -to_float(x.get("score", 0.0)), -to_int(x.get("cited_by", 0)), x.get("paper_id", "")))
    selected: List[Dict[str, Any]] = []
    for i, c in enumerate(scored):
        a = 1 if i < target_abs else 0
        p = 1 if i < target_pdf else 0
        if p and c.get("downloadability_hint") not in ("HIGH", "MED"):
            p = 0
        selected.append({**c, "select_for_abstract": a, "select_for_pdf": p, "tier": "support", "reason": "light-screening"})
    return selected, errors


def validate_full_screening_response(data: Dict[str, Any], candidates: List[Dict[str, Any]], max_abs: int, max_pdf: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("selected"), list):
        return [], ["Корень ответа должен быть объектом с массивом 'selected'."]

    c_map = {c["paper_id"]: c for c in candidates}
    selected: List[Dict[str, Any]] = []
    abs_count = 0
    pdf_count = 0
    tiers = {"core", "support", "background"}

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
        tier = normalize_text(str(row.get("tier", "background"))).lower() or "background"
        if tier not in tiers:
            errors.append(f"{pid}: tier должен быть core/support/background")
            continue
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
        selected.append({**c_map[pid], "select_for_abstract": a, "select_for_pdf": p, "tier": tier, "reason": normalize_text(row.get("reason", "")) or "без причины"})

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


def fetch_semantic_scholar_pdf_url(doi: str, title: str, s2: ApiClient, api_key: str) -> Tuple[str, str]:
    headers = {"x-api-key": api_key} if api_key else {}
    if doi:
        u = f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{requests.utils.quote(doi, safe='')}?fields=openAccessPdf"
        try:
            j = s2.get_json(u, headers=headers, retries=2, retry_on_404=False)
            return normalize_text(((j.get("openAccessPdf") or {}).get("url", ""))), u
        except Exception:
            return "", u
    q = requests.utils.quote(title[:140])
    u = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={q}&limit=1&fields=openAccessPdf,title"
    try:
        j = s2.get_json(u, headers=headers, retries=2, retry_on_404=False)
        data = j.get("data") or []
        if data:
            return normalize_text(((data[0].get("openAccessPdf") or {}).get("url", ""))), u
    except Exception:
        return "", u
    return "", u


def fetch_unpaywall_pdf_urls(doi: str, up: ApiClient, email: str, base_url: str = UNPAYWALL_BASE) -> Tuple[List[str], str]:
    u = f"{base_url}/{requests.utils.quote(doi, safe='')}?email={requests.utils.quote(email)}"
    j = up.get_json(u, retries=2)
    urls: List[str] = []
    best = (j.get("best_oa_location") or {}).get("url_for_pdf")
    if best:
        urls.append(best)
    for loc in j.get("oa_locations") or []:
        x = normalize_text(loc.get("url_for_pdf", ""))
        if x:
            urls.append(x)
    uniq, seen = [], set()
    for x in urls:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq, u


def fetch_openalex_pdf_urls(row: Dict[str, Any], oa: ApiClient, secrets: Dict[str, str]) -> Tuple[List[str], str]:
    work, url = get_openalex_work(row, oa, secrets)
    urls: List[str] = []
    open_access = work.get("open_access") or {}
    if open_access.get("is_oa"):
        oa_url = normalize_text(open_access.get("oa_url", ""))
        if oa_url:
            urls.append(oa_url)
    primary_location = work.get("primary_location") or {}
    pdf_url = normalize_text(primary_location.get("pdf_url", ""))
    if pdf_url:
        urls.append(pdf_url)
    best_oa = work.get("best_oa_location") or {}
    best_pdf = normalize_text(best_oa.get("pdf_url", ""))
    if best_pdf:
        urls.append(best_pdf)
    uniq, seen = [], set()
    for x in urls:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq, url


def fetch_crossref_pdf_urls(doi: str, cr: ApiClient, mailto: str) -> Tuple[List[str], str]:
    h = {"User-Agent": f"IDEA_PIPELINE_C1/1.0 (mailto:{mailto})"}
    u = f"{CROSSREF_BASE}/{requests.utils.quote(doi, safe='')}?mailto={requests.utils.quote(mailto)}"
    j = cr.get_json(u, headers=h, retries=2)
    urls: List[str] = []
    for link in (j.get("message") or {}).get("link") or []:
        x = normalize_text(link.get("URL", ""))
        if x:
            urls.append(x)
    uniq, seen = [], set()
    for x in urls:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq, u


def is_unpaywall_connectivity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = [
        "nameresolutionerror",
        "remote disconnected",
        "remotedisconnected",
        "failed to establish a new connection",
        "temporary failure in name resolution",
        "errno 11001",
        "dns",
        "read timed out",
        "readtimeout",
        "connectionerror",
        "max retries exceeded",
        "timed out",
        "connection aborted",
        "getaddrinfo failed",
        "socket.gaierror",
    ]
    if any(t in text for t in tokens):
        return True
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, socket.gaierror, ConnectionError))


def check_unpaywall_health(up: ApiClient, email: str, logger: RunLogger, base_url: str) -> Tuple[bool, str]:
    test_doi = "10.1038/nature12373"
    try:
        fetch_unpaywall_pdf_urls(test_doi, up, email, base_url)
        return True, ""
    except Exception as exc:
        if is_unpaywall_connectivity_error(exc) or "http 5" in str(exc).lower():
            logger.warn(f"Unpaywall health-check не пройден: {exc}")
            return False, str(exc)
        logger.warn(f"Unpaywall health-check: неожиданный ответ, продолжаю с включенным Unpaywall: {exc}")
        return True, ""


def classify_pdf_url(url: str) -> str:
    u = (url or "").lower()
    high_tokens = ["pmc", "pubmedcentral", "arxiv.org", "zenodo.org", "hal.science", "repository", "eprints", "figshare", "osf.io"]
    low_tokens = ["sciencedirect", "wiley", "oup", "elsevier", "springer", "nature.com", "tandfonline", "linkinghub", "pdfdirect"]
    if any(t in u for t in high_tokens):
        return "HIGH"
    if any(t in u for t in low_tokens):
        return "LOW"
    if u:
        return "MED"
    return "NONE"


def sort_pdf_candidates(urls: List[str]) -> List[str]:
    priority = {"HIGH": 0, "MED": 1, "LOW": 2, "NONE": 3}
    uniq, seen = [], set()
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


def open_prepare_artifacts(chat_dir: Path, prompt: Path, response: Path) -> None:
    try:
        subprocess.Popen(["explorer", str(chat_dir)])
    except Exception:
        pass
    for p in [prompt, response]:
        try:
            subprocess.Popen(["notepad", str(p)])
        except Exception:
            pass


def copy_prompt_file_to_clipboard(prompt_path: Path) -> bool:
    try:
        cmd = f'cmd /c type "{prompt_path}" | clip'
        return subprocess.call(cmd, shell=True) == 0
    except Exception:
        return False


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


def download_pdf(url: str, path: Path, timeout: Tuple[int, int]) -> Tuple[bool, str, int, str]:
    try:
        connect_timeout = timeout[0]
        read_timeout = max(timeout[1], 30)
        with requests.get(url, timeout=(connect_timeout, read_timeout), stream=True, allow_redirects=True) as r:
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


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        return default


def response_ready(resp_path: Path) -> bool:
    if not resp_path.exists():
        return False
    raw = normalize_text(resp_path.read_text(encoding='utf-8', errors='replace'))
    return raw not in ('', '{}', '{"selected": []}')


def wait_for_response(resp_path: Path, logger: RunLogger, timeout_sec: int = 3600) -> bool:
    start = time.time()
    last_size = -1
    logger.info('[4/6] ChatGPT screening: ожидаю RESPONSE.json ...')
    while True:
        try:
            if resp_path.exists():
                size = resp_path.stat().st_size
                if size != last_size:
                    last_size = size
                    logger.info(f'RESPONSE.json обновлен: size={size} байт')
                if response_ready(resp_path):
                    try:
                        parse_screening_response(resp_path)
                        return True
                    except Exception:
                        pass
            elapsed = int(time.time() - start)
            if elapsed >= timeout_sec:
                logger.warn('Таймаут ожидания ответа ChatGPT.')
                return False
            if elapsed % 30 == 0:
                print(f'... Я жду RESPONSE.json ({elapsed} сек)')
            time.sleep(2)
        except KeyboardInterrupt:
            logger.warn('Ожидание RESPONSE.json прервано пользователем (Ctrl+C).')
            return False


def ask_cleanup_choice(idea_dir: Path) -> str:
    traces = [
        idea_dir / 'in' / 'papers',
        idea_dir / 'in' / 'c1_chatgpt',
        idea_dir / 'out' / 'harvest_report.md',
        idea_dir / 'out' / 'prisma_c1.md',
    ]
    if not any(p.exists() for p in traces):
        return 'N'
    print('Найдены результаты прошлого запуска C1. Удалить? [Y] Да / [N] Нет (resume) / [Q] Отмена')
    while True:
        choice = normalize_text(input('Ваш выбор [Y/N/Q]: ')).upper() or 'Y'
        if choice in ('Y', 'N', 'Q'):
            return choice


def cleanup_previous_run(idea_dir: Path, logger: RunLogger) -> None:
    papers_dir = idea_dir / 'in' / 'papers'
    chat_dir = idea_dir / 'in' / 'c1_chatgpt'
    out_dir = idea_dir / 'out'
    manual_backup = idea_dir / 'in' / 'pdf_manual_queue.prev.csv'
    src_manual = papers_dir / 'pdf_manual_queue.csv'
    if src_manual.exists():
        shutil.copyfile(src_manual, manual_backup)
    if papers_dir.exists():
        shutil.rmtree(papers_dir, ignore_errors=True)
    if chat_dir.exists():
        shutil.rmtree(chat_dir, ignore_errors=True)
    for p in [out_dir / 'harvest_report.md', out_dir / 'prisma_c1.md', out_dir / '_moduleC1_checkpoint.json']:
        p.unlink(missing_ok=True)
    logger.info('Очистка результатов прошлого C1 завершена.')


def open_final_artifacts(out_dir: Path, manifests_dir: Path, papers_dir: Path) -> None:
    report = out_dir / 'harvest_report.md'
    errors = manifests_dir / 'errors.csv'
    try:
        if report.exists():
            subprocess.Popen(['notepad', str(report)])
    except Exception:
        pass
    try:
        if errors.exists() and errors.stat().st_size > 10:
            subprocess.Popen(['notepad', str(errors)])
    except Exception:
        pass
    try:
        if papers_dir.exists():
            subprocess.Popen(['explorer', str(papers_dir)])
    except Exception:
        pass


def repair_manual_pdfs_only(papers_dir: Path, pdf_dir: Path, logger: RunLogger) -> int:
    qpath = papers_dir / 'pdf_manual_queue.csv'
    if not qpath.exists():
        logger.info('Режим repair: pdf_manual_queue.csv не найден.')
        return 0
    rows = list(csv.DictReader(qpath.open('r', encoding='utf-8-sig', newline='')))
    copied = 0
    for row in rows:
        if (row.get('status') or '') != 'MISSING':
            continue
        src_val = normalize_text(row.get('manual_pdf_local_path', ''))
        if not src_val:
            continue
        src = Path(src_val)
        if src.exists():
            dst = pdf_dir / f"{row.get('paper_id','')}.pdf"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            row['status'] = 'FOUND_MANUAL'
            copied += 1
    write_csv(qpath, rows, ['cite_short', 'paper_id', 'doi', 'title', 'year', 'first_author', 'tried_urls', 'manual_pdf_local_path', 'status', 'notes'])
    logger.info(f'Режим repair: подхвачено ручных PDF: {copied}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='Stage C1: универсальный ChatGPT-screening + harvest')
    ap.add_argument('--idea-dir', default='')
    ap.add_argument('--screening', choices=['light', 'full', 'none'], default='light')
    ap.add_argument('--max-candidates', type=int, default=400)
    ap.add_argument('--max-abstracts', type=int, default=120)
    ap.add_argument('--max-pdfs', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=0)
    ap.add_argument('--timeout-connect', type=int, default=5)
    ap.add_argument('--timeout-read', type=int, default=20)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--repair-manual-pdfs', action='store_true')
    ap.add_argument('--disable-unpaywall', action='store_true')
    ap.add_argument('--unpaywall-base', default=UNPAYWALL_BASE)
    ap.add_argument('--rps-openalex', type=float, default=2.0)
    ap.add_argument('--rps-unpaywall', type=float, default=1.5)
    ap.add_argument('--rps-crossref', type=float, default=1.0)
    ap.add_argument('--rps-europepmc', type=float, default=1.5)
    ap.add_argument('--rps-s2', type=float, default=1.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    unpaywall_base = normalize_text(args.unpaywall_base) or UNPAYWALL_BASE
    idea_dir = find_idea_dir(root, args.idea_dir or None)
    logs_dir, in_dir, papers_dir, manifests_dir = idea_dir / 'logs', idea_dir / 'in', idea_dir / 'in' / 'papers', idea_dir / 'in' / 'papers' / 'manifests'
    out_dir, abstracts_dir, pdf_dir = idea_dir / 'out', idea_dir / 'in' / 'papers' / 'abstracts', idea_dir / 'in' / 'papers' / 'pdfs'
    c1_chat_dir = in_dir / 'c1_chatgpt'
    prompt_path, resp_path = c1_chat_dir / 'PROMPT.txt', c1_chat_dir / 'RESPONSE.json'
    candidates_path = papers_dir / 'candidates.json'
    checkpoint_path = out_dir / '_moduleC1_checkpoint.json'

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log = logs_dir / f'moduleC1_{ts}.log'
    last_log = logs_dir / 'moduleC1_LAST.log'
    logger = RunLogger(run_log)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    try:
        logger.info('=== Stage C1 ===')
        logger.info(f'IDEA_DIR={idea_dir}')
        if args.repair_manual_pdfs:
            return repair_manual_pdfs_only(papers_dir, pdf_dir, logger)

        cleanup_choice = ask_cleanup_choice(idea_dir)
        if cleanup_choice == 'Q':
            logger.info('Отмена по запросу пользователя.')
            return 0
        if cleanup_choice == 'Y':
            cleanup_previous_run(idea_dir, logger)

        harvest_mode = False
        secrets = load_secrets(root)
        if args.timeout and args.timeout > 0:
            req_timeout = (min(args.timeout, 10), args.timeout)
        else:
            req_timeout = (args.timeout_connect, args.timeout_read)
        oa = ApiClient('OpenAlex', args.rps_openalex, req_timeout, logger)
        up = ApiClient('Unpaywall', args.rps_unpaywall, (5, 10), logger)
        cr = ApiClient('Crossref', args.rps_crossref, req_timeout, logger)
        epmc = ApiClient('EuropePMC', args.rps_europepmc, req_timeout, logger)
        s2 = ApiClient('SemanticScholar', args.rps_s2, req_timeout, logger)
        report_notes: List[str] = []
        unpaywall_enabled = not args.disable_unpaywall
        unpaywall_warned = False

        if args.disable_unpaywall:
            msg = 'Unpaywall недоступен/отключён → PDF будем искать без него (OpenAlex/Crossref/EuropePMC).'
            logger.warn(msg)
            report_notes.append('- Unpaywall отключён вручную через --disable-unpaywall.')
            unpaywall_warned = True
        elif not secrets.get('UNPAYWALL_EMAIL'):
            unpaywall_enabled = False
            msg = 'Unpaywall недоступен/отключён → PDF будем искать без него (OpenAlex/Crossref/EuropePMC).'
            logger.warn(msg)
            report_notes.append('- Unpaywall отключён: не задан UNPAYWALL_EMAIL в config/secrets.env.')
            unpaywall_warned = True
        else:
            ok, reason = check_unpaywall_health(up, secrets['UNPAYWALL_EMAIL'], logger, unpaywall_base)
            if not ok:
                unpaywall_enabled = False
                msg = 'Unpaywall недоступен (DNS/timeout). Продолжаю без Unpaywall. PDF-скачивание будет ограничено.'
                logger.warn(msg)
                report_notes.append(f'- {msg} Причина: {reason}')
                unpaywall_warned = True

        if not harvest_mode:
            corpus_path = out_dir / 'corpus.csv'
            if not corpus_path.exists():
                raise FileNotFoundError(f'Не найден обязательный файл: {corpus_path}')
            rows = load_corpus(corpus_path)
            logger.info(f'[1/6] Читаю corpus.csv (N={len(rows)})')
            idea_text, structured, keywords = extract_idea_context(idea_dir)
            candidates, rejected, stats = preclean_corpus(rows, oa, secrets, keywords, logger, args.max_candidates)
            stats['corpus_n'] = len(rows)
            logger.info(f"[2/6] Дедуп/фильтры => осталось {len(candidates)}")
            logger.info('[3/6] Готовлю candidates.json и PROMPT')
            for c in candidates:
                c['oa_pdf_candidates'] = sort_pdf_candidates([c.get('oa_pdf_url', '')])[:1]
                c['downloadability_hint'] = choose_hint(c['oa_pdf_candidates'])
            write_json(candidates_path, candidates)
            write_csv(papers_dir / 'preselection_rejected.csv', rejected, ['index', 'title', 'doi', 'openalex_id', 'reason'])
            write_json(checkpoint_path, {'phase': 'PREPARE_DONE', 'stats': stats, 'idea_dir': str(idea_dir), 'ts': ts})
            if args.screening in ('light', 'full'):
                c1_chat_dir.mkdir(parents=True, exist_ok=True)
                prompt = generate_light_screening_prompt(idea_text, structured, candidates, args.max_abstracts, args.max_pdfs) if args.screening == 'light' else generate_full_screening_prompt(idea_text, structured, candidates, args.max_abstracts, args.max_pdfs)
                prompt_path.write_text(prompt, encoding='utf-8')
                if not resp_path.exists():
                    resp_path.write_text('{}\n', encoding='utf-8')
                (c1_chat_dir / 'README_WHAT_TO_DO.txt').write_text('1) Откройте PROMPT.txt и отправьте в ChatGPT.\n2) Вставьте JSON-ответ в RESPONSE.json и сохраните файл.\n3) Скрипт автоматически продолжит работу.\n', encoding='utf-8')
                copied = copy_prompt_file_to_clipboard(prompt_path)
                open_prepare_artifacts(c1_chat_dir, prompt_path, resp_path)
                logger.info(f'[4/6] ChatGPT screening ({args.screening.upper()}): создан PROMPT, ожидаю RESPONSE...')
                print('\nШаги: 1) Вставьте PROMPT в ChatGPT  2) Сохраните JSON в RESPONSE.json  3) Скрипт продолжит сам.\n')
                if copied:
                    print('PROMPT скопирован в буфер обмена.')
                if not wait_for_response(resp_path, logger, timeout_sec=3600):
                    return 2

        logger.info('[4/6] HARVEST: использую готовые candidates + RESPONSE')
        candidates = load_json_if_exists(candidates_path, [])
        if not candidates:
            raise RuntimeError('Не найден candidates.json для HARVEST. Сначала выполните PREPARE.')
        c_map = {c['paper_id']: c for c in candidates}
        stats = (load_json_if_exists(checkpoint_path, {}).get('stats') or {'corpus_n': 0, 'after_dedup_n': 0, 'after_domain_filter_n': len(candidates)})
        if args.screening == 'none':
            selected = [{**c, 'select_for_abstract': 1 if i < args.max_abstracts else 0, 'select_for_pdf': 1 if i < args.max_pdfs else 0, 'tier': 'support', 'reason': 'автовыбор'} for i, c in enumerate(candidates)]
        elif args.screening == 'light':
            raw = parse_screening_response(resp_path)
            selected, validation_errors = apply_light_screening(raw, candidates, args.max_abstracts, args.max_pdfs)
            if validation_errors:
                write_json(c1_chat_dir / 'RESPONSE.INVALID.json', {'errors': validation_errors, 'response': raw})
                raise RuntimeError('RESPONSE.json (LIGHT) не прошел валидацию; см. RESPONSE.INVALID.json')
        else:
            raw = parse_screening_response(resp_path)
            selected, validation_errors = validate_full_screening_response(raw, candidates, args.max_abstracts, args.max_pdfs)
            if validation_errors:
                write_json(c1_chat_dir / 'RESPONSE.INVALID.json', {'errors': validation_errors, 'response': raw})
                raise RuntimeError('RESPONSE.json не прошел валидацию; см. RESPONSE.INVALID.json')
        selected_abs = [x for x in selected if int(x.get('select_for_abstract', 0)) == 1]
        selected_pdf = [x for x in selected if int(x.get('select_for_pdf', 0)) == 1]
        logger.info(f'[3/6] Обогащение OA-линками (best-effort) для топ PDF-кандидатов: {min(len(selected_pdf), 80)}')
        if not unpaywall_enabled and not unpaywall_warned:
            msg = 'Unpaywall недоступен/отключён → PDF будем искать без него (OpenAlex/Crossref/EuropePMC).'
            logger.warn(msg)
            report_notes.append(f'- {msg}')
            unpaywall_warned = True
        for item in selected_pdf[:80]:
            urls = list(item.get('oa_pdf_candidates', []))
            try:
                xs, _ = fetch_openalex_pdf_urls(item, oa, secrets)
                urls.extend(xs)
            except Exception as exc:
                error_rows.append({'paper_id': item['paper_id'], 'step': 'openalex_pdf_enrich', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})

            if unpaywall_enabled and item.get('doi') and secrets.get('UNPAYWALL_EMAIL'):
                try:
                    xs, _ = fetch_unpaywall_pdf_urls(item['doi'], up, secrets['UNPAYWALL_EMAIL'], unpaywall_base)
                    urls.extend(xs)
                except Exception as exc:
                    if is_unpaywall_connectivity_error(exc):
                        unpaywall_enabled = False
                        msg = 'Unpaywall недоступен (DNS/timeout). Продолжаю без Unpaywall. PDF-скачивание будет ограничено.'
                        if not unpaywall_warned:
                            logger.warn(msg)
                            report_notes.append(f'- {msg} Причина: {exc}')
                            unpaywall_warned = True
                    else:
                        error_rows.append({'paper_id': item['paper_id'], 'step': 'unpaywall_enrich', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})

            if item.get('doi') and secrets.get('CROSSREF_MAILTO'):
                try:
                    xs, _ = fetch_crossref_pdf_urls(item['doi'], cr, secrets['CROSSREF_MAILTO'])
                    urls.extend(xs)
                except Exception as exc:
                    error_rows.append({'paper_id': item['paper_id'], 'step': 'crossref_pdf_enrich', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if item.get('pmid'):
                try:
                    xs, _ = fetch_europepmc_pdf_urls(item['pmid'], epmc)
                    urls.extend(xs)
                except Exception as exc:
                    error_rows.append({'paper_id': item['paper_id'], 'step': 'europepmc_enrich', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if not urls and (item.get('doi') or item.get('title')):
                s2u, _ = fetch_semantic_scholar_pdf_url(item.get('doi', ''), item.get('title', ''), s2, secrets.get('S2_API_KEY', ''))
                if s2u:
                    urls.append(s2u)
            item['oa_pdf_candidates'] = sort_pdf_candidates(urls)[:5]
            item['downloadability_hint'] = choose_hint(item['oa_pdf_candidates'])
        write_csv(papers_dir / 'selection.csv', selected, ['paper_id', 'title', 'year', 'doi', 'openalex_id', 'first_author', 'tier', 'downloadability_hint', 'select_for_abstract', 'select_for_pdf', 'reason', 'domain', 'field', 'subfield'])
        write_jsonl(papers_dir / 'papers.jsonl', selected)

        reason_counter: Counter[str] = Counter()
        abs_ok = 0
        logger.info('[5/6] Качаю абстракты')
        for i, item in enumerate(selected_abs, start=1):
            pid = item['paper_id']
            abs_path = abstracts_dir / f'{pid}.txt'
            if abs_path.exists() and not args.force:
                abs_ok += 1
                logger.info(f'[abs {i}/{len(selected_abs)}] {pid}: RESUME_OK')
                continue
            got = False
            corpus_abs = normalize_text((c_map.get(pid) or {}).get('source_row', {}).get('abstract', ''))
            if len(corpus_abs) >= 200 and save_abstract(abs_path, corpus_abs):
                got = True
                abs_ok += 1
            if not got:
                try:
                    txt, _ = fetch_openalex_abstract(item, oa, secrets)
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                except Exception as exc:
                    error_rows.append({'paper_id': pid, 'step': 'openalex_abstract', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if not got and item.get('pmid'):
                try:
                    txt, _ = fetch_europepmc_abstract(item.get('pmid', ''), epmc)
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                except Exception as exc:
                    error_rows.append({'paper_id': pid, 'step': 'europepmc_abstract', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if not got and item.get('doi') and secrets.get('CROSSREF_MAILTO'):
                try:
                    txt, _ = fetch_crossref_abstract(item['doi'], cr, secrets['CROSSREF_MAILTO'])
                    if save_abstract(abs_path, txt):
                        got = True
                        abs_ok += 1
                except Exception as exc:
                    error_rows.append({'paper_id': pid, 'step': 'crossref_abstract', 'url': '', 'http_code': '', 'message': str(exc), 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if not got:
                reason_counter['NO_ABSTRACT'] += 1
                manifest_rows.append({'paper_id': pid, 'step': 'abstract', 'status': 'NO_ABSTRACT', 'source': 'none', 'url': ''})
            else:
                manifest_rows.append({'paper_id': pid, 'step': 'abstract', 'status': 'ok', 'source': 'multi', 'url': ''})

        manual_queue_path = papers_dir / 'pdf_manual_queue.csv'
        existing_manual = list(csv.DictReader(manual_queue_path.open('r', encoding='utf-8-sig', newline=''))) if manual_queue_path.exists() else []
        manual_map = {r.get('paper_id', ''): r for r in existing_manual}
        for pid, q in list(manual_map.items()):
            if (q.get('status') or '') == 'MISSING' and normalize_text(q.get('manual_pdf_local_path', '')):
                src = Path(q['manual_pdf_local_path'])
                if src.exists():
                    dst = pdf_dir / f'{pid}.pdf'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    q['status'] = 'FOUND_MANUAL'

        pdf_ok = 0
        logger.info('[6/6] Качаю PDF')
        for i, item in enumerate(selected_pdf, start=1):
            pid = item['paper_id']
            pdf_path = pdf_dir / f'{pid}.pdf'
            if pdf_path.exists() and not args.force:
                ok, _ = ensure_pdf_valid(pdf_path)
                if ok:
                    pdf_ok += 1
                    logger.info(f'[pdf {i}/{len(selected_pdf)}] {pid}: RESUME_OK')
                    continue
            urls = sort_pdf_candidates(list(item.get('oa_pdf_candidates', [])))
            tried, got_pdf = [], False
            for u in urls:
                tried.append(u)
                ok, msg, code, ctype = download_pdf(u, pdf_path, req_timeout)
                manifest_rows.append({'paper_id': pid, 'step': 'download_pdf', 'status': 'ok' if ok else 'failed', 'source': classify_pdf_url(u), 'url': u, 'http_code': code, 'content_type': ctype})
                if ok:
                    got_pdf = True
                    pdf_ok += 1
                    break
                error_rows.append({'paper_id': pid, 'step': 'download_pdf', 'url': u, 'http_code': code, 'message': msg, 'doi': item.get('doi', ''), 'openalex_id': item.get('openalex_id', '')})
            if not got_pdf:
                reason_counter['NO_OA_PDF'] += 1
                manual_map[pid] = {
                    'cite_short': build_cite_short(item.get('first_author', ''), item.get('year', ''), item.get('title', '')),
                    'paper_id': pid,
                    'doi': item.get('doi', ''),
                    'title': item.get('title', ''),
                    'year': item.get('year', ''),
                    'first_author': item.get('first_author', ''),
                    'tried_urls': ' | '.join(tried[:3]),
                    'manual_pdf_local_path': manual_map.get(pid, {}).get('manual_pdf_local_path', ''),
                    'status': 'MISSING',
                    'notes': '403 publisher' if any(str(e.get('http_code','')) in ('401', '403') for e in error_rows if e.get('paper_id') == pid) else 'no_oa_location',
                }
            elif pid in manual_map:
                manual_map[pid]['status'] = 'DOWNLOADED_OK'

        write_csv(manual_queue_path, list(manual_map.values()), ['cite_short', 'paper_id', 'doi', 'title', 'year', 'first_author', 'tried_urls', 'manual_pdf_local_path', 'status', 'notes'])
        write_jsonl(manifests_dir / 'manifest.jsonl', manifest_rows)
        write_csv(manifests_dir / 'errors.csv', error_rows, ['paper_id', 'step', 'url', 'http_code', 'message', 'doi', 'openalex_id'])

        status = 'OK' if (not error_rows and unpaywall_enabled) else 'DEGRADED'
        top_txt = '\n'.join([f'- {k}: {v}' for k, v in reason_counter.most_common(5)]) if reason_counter else '- нет'
        notes_block = "\n".join(report_notes) if report_notes else "- нет"
        unpaywall_status = 'OFF (недоступен/отключён)' if not unpaywall_enabled else 'ON'
        (out_dir / 'harvest_report.md').write_text(
            f"# Stage C1 — Harvest report\n\nСтатус: **{status}**\n\n- selected abstracts: {len(selected_abs)}\n- selected pdfs: {len(selected_pdf)}\n- downloaded abstracts_ok: {abs_ok}\n- downloaded pdfs_ok: {pdf_ok}\n- errors total: {len(error_rows)}\n- Unpaywall: {unpaywall_status}\n\n## Top-5 причин\n{top_txt}\n\n## Предупреждения и деградация\n{notes_block}\n",
            encoding='utf-8',
        )
        (out_dir / 'prisma_c1.md').write_text(
            f"# PRISMA C1\n\n- status: {status}\n- corpus: {stats.get('corpus_n',0)}\n- after_dedup: {stats.get('after_dedup_n',0)}\n- after_domain_filter: {stats.get('after_domain_filter_n',0)}\n- selected_for_abstract: {len(selected_abs)}\n- selected_for_pdf: {len(selected_pdf)}\n- downloaded_abstract_ok: {abs_ok}\n- downloaded_pdf_ok: {pdf_ok}\n- errors_total: {len(error_rows)}\n",
            encoding='utf-8',
        )
        summary_lines = [
            "ИТОГ Stage C1:",
            f"- corpus: {stats.get('corpus_n', 0)}",
            f"- candidates: {stats.get('after_domain_filter_n', len(candidates))}",
            f"- selected_for_abstract: {len(selected_abs)} + abstracts_ok={abs_ok}",
            f"- selected_for_pdf: {len(selected_pdf)} + pdf_ok={pdf_ok}",
            f"- Unpaywall: {unpaywall_status}",
            f"- selection.csv: {papers_dir / 'selection.csv'}",
        ]
        errors_path = manifests_dir / 'errors.csv'
        if errors_path.exists():
            summary_lines.append(f"- errors.csv: {errors_path}")
        summary_lines.append(f"- moduleC1_LAST.log: {last_log}")
        summary_text = "\n".join(summary_lines)
        logger.info(summary_text.replace("\n", " | "))
        print(summary_text)
        open_final_artifacts(out_dir, manifests_dir, papers_dir)
        return 0

    except Exception as exc:
        logger.err(f'Критическая ошибка Stage C1: {exc}')
        if not (manifests_dir / 'errors.csv').exists():
            write_csv(manifests_dir / 'errors.csv', [{'paper_id': '', 'step': 'fatal', 'url': '', 'http_code': '', 'message': str(exc), 'doi': '', 'openalex_id': ''}], ['paper_id', 'step', 'url', 'http_code', 'message', 'doi', 'openalex_id'])
        return 1
    finally:
        try:
            shutil.copyfile(run_log, last_log)
        except Exception:
            pass
        logger.close()


if __name__ == '__main__':
    raise SystemExit(main())
