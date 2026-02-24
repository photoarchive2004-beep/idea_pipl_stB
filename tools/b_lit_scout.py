#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

BASE = "https://api.openalex.org"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF_BASE = "https://api.crossref.org/works"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV_BASE = "https://export.arxiv.org/api/query"
BIORXIV_BASE = "https://api.biorxiv.org/details"
VERSION = "v10.0-universal"

DEFAULT_STOP_RU = {
    "и", "или", "что", "как", "при", "для", "это", "этот", "эта", "эти", "того", "также", "через", "между", "после", "перед", "если", "ли", "по", "на",
    "из", "в", "к", "о", "об", "у", "над", "под", "без", "же", "бы", "мы", "вы", "они", "оно", "она", "данные", "анализ", "метод", "методы", "результат",
    "результаты", "работа", "исследование", "исследования", "используя", "использование", "новый", "новые", "может", "могут", "более", "менее",
}

DEFAULT_STOP_EN = {
    "the", "and", "or", "for", "with", "from", "into", "about", "this", "that", "those", "these", "have", "has", "had", "were", "was", "are", "is", "be", "been",
    "being", "study", "studies", "result", "results", "method", "methods", "data", "analysis", "analyses", "using", "use", "used", "based", "across", "between",
    "also", "such", "their", "there", "within", "without", "under", "over", "via", "can", "could", "may", "might", "more", "most", "less", "least", "new", "on",
    "in", "at", "to", "by", "of", "a", "an", "as", "it", "its", "we", "our", "they", "them", "if", "than", "then",
}

STOP: Set[str] = set(DEFAULT_STOP_RU) | set(DEFAULT_STOP_EN)

CSV_COLS = ["source", "openalex_id", "pmid", "arxiv_id", "doi", "title", "year", "publication_date", "type", "venue", "authors", "cited_by", "language",
            "primary_domain_id", "primary_field_id", "primary_topic", "topics", "concepts", "abstract", "oa_pdf_url", "best_url", "score"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(fp, msg: str) -> None:
    fp.write(f"[{now_local()}] {msg}\n")
    fp.flush()


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def read_env_file(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_sources_config(root_dir: str) -> Dict[str, Any]:
    defaults = {
        "enable": {"openalex": True, "ncbi": True, "crossref": True, "unpaywall": True, "europepmc": True, "arxiv": True, "biorxiv": True},
        "request_caps": {"openalex": 160, "ncbi": 80, "crossref": 50, "unpaywall": 250},
        "drift": {"threshold": 0.22, "explore_budget": 0.15},
        "diversity": {"max_per_venue": 14, "max_per_topic": 50, "mmr_lambda": 0.72},
        "relevance_feedback": {"top_n": 120, "phrases": 30, "queries": 10},
        "targets": {"focused": [800, 1500], "balanced": [1500, 3000], "wide": [3000, 8000]},
    }
    path = os.path.join(root_dir, "config", "stage_b_sources.json")
    if not os.path.exists(path):
        return defaults
    try:
        user = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return defaults
    def deep(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(dst)
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep(out[k], v)
            else:
                out[k] = v
        return out
    return deep(defaults, user)


def get_secrets(root_dir: str) -> Dict[str, str]:
    env = read_env_file(os.path.join(root_dir, "config", "secrets.env"))
    for k in ["OPENALEX_API_KEY", "OPENALEX_MAILTO", "NCBI_API_KEY", "NCBI_EMAIL", "CROSSREF_MAILTO", "UNPAYWALL_EMAIL"]:
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def load_stopwords(root_dir: str) -> Set[str]:
    stop: Set[str] = set(STOP)
    for name in ["stopwords_ru.txt", "stopwords_en.txt"]:
        path = os.path.join(root_dir, "config", name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    t = line.strip().lower()
                    if t and not t.startswith("#"):
                        stop.add(t)
        except Exception:
            pass
    return stop


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\w]+", text or "", flags=re.UNICODE) if re.search(r"[A-Za-zА-Яа-яЁё]", t)]


def informative_token(token: str, stopwords: Set[str]) -> bool:
    t = (token or "").strip().lower()
    if len(t) < 3:
        return False
    if t in stopwords:
        return False
    if re.fullmatch(r"\d+", t):
        return False
    if not re.search(r"[A-Za-zА-Яа-яЁё]", t):
        return False
    return True


def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def rank_terms(text: str, stopwords: Set[str], max_terms: int = 120) -> List[str]:
    freq = Counter(t for t in tokenize(text) if informative_token(t, stopwords))
    return [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]]


def make_ngrams(tokens: List[str], n: int, stopwords: Set[str]) -> Counter:
    c: Counter = Counter()
    for i in range(len(tokens) - n + 1):
        ng = tokens[i:i + n]
        if any(not informative_token(t, stopwords) for t in ng):
            continue
        c[" ".join(ng)] += 1
    return c


def quote_term(term: str) -> str:
    t = clean_spaces(term)
    return f'"{t}"' if " " in t else t


def _normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def first_author(authors: str) -> str:
    return clean_spaces((authors or "").split(";")[0]).lower()


def dedup_key(row: Dict[str, Any]) -> str:
    if row.get("doi"):
        return "doi:" + row.get("doi", "").lower().replace("https://doi.org/", "")
    if row.get("pmid"):
        return "pmid:" + str(row.get("pmid"))
    if row.get("openalex_id"):
        return "oa:" + row.get("openalex_id", "")
    if row.get("arxiv_id"):
        return "arxiv:" + row.get("arxiv_id", "")
    return f"title:{_normalize_title(row.get('title',''))}|year:{row.get('year','')}|fa:{first_author(row.get('authors',''))}"


def input_blob(idea_dir: str) -> str:
    out_dir = os.path.join(idea_dir, "out")
    parts = [read_text(os.path.join(out_dir, "structured_idea.json")), read_text(os.path.join(idea_dir, "idea.txt")), read_text(os.path.join(idea_dir, "in", "idea.txt"))]
    return "\n".join([x for x in parts if x and x.strip()])


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})


def count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dots = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dots / (na * nb) if na > 0 and nb > 0 else 0.0


@dataclass
class HttpClient:
    timeout: int = 35
    rps: float = 2.0
    max_retries: int = 2
    request_count: int = 0

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None, log_fp=None) -> Dict[str, Any]:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.rps > 0:
                    time.sleep(1.0 / self.rps)
                self.request_count += 1
                req = urllib.request.Request(url, headers=headers or {})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception as e:
                last = e
                if log_fp:
                    log(log_fp, f"[WARN] HTTP ошибка {type(e).__name__}: {e}")
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(str(last))


class OAClient(HttpClient):
    def __init__(self, api_key: str, mailto: str, **kw: Any):
        super().__init__(**kw)
        self.api_key = api_key
        self.mailto = mailto

    def build(self, params: Dict[str, Any]) -> str:
        p = dict(params)
        p["api_key"] = self.api_key
        if self.mailto:
            p["mailto"] = self.mailto
        return BASE + "/works?" + urllib.parse.urlencode(p, quote_via=urllib.parse.quote)

    def list_works(self, *, search: Optional[str], filter_parts: List[str], sort: str, pages: int, select_fields: str, per_page: int = 200, log_fp=None) -> Iterable[Dict[str, Any]]:
        cursor = "*"
        for _ in range(max(1, pages)):
            params: Dict[str, Any] = {"per-page": str(min(200, max(1, per_page))), "cursor": cursor, "sort": sort, "select": select_fields}
            if search:
                params["search"] = search
            if filter_parts:
                params["filter"] = ",".join(filter_parts)
            j = self.get_json(self.build(params), headers={"User-Agent": "IdeaPipeline-StageB/v10"}, log_fp=log_fp)
            res = j.get("results") or []
            if not res:
                break
            for w in res:
                yield w
            cursor = (j.get("meta") or {}).get("next_cursor")
            if not cursor:
                break


def oa_id_short(x: str) -> str:
    m = re.search(r"/(W\d+)$", x or "")
    return m.group(1) if m else (x or "")


def reconstruct_abstract(inv: Dict[str, List[int]]) -> str:
    pos: Dict[int, str] = {}
    for w, ps in (inv or {}).items():
        for p in ps:
            try:
                pos[int(p)] = w
            except Exception:
                pass
    return " ".join(pos[i] for i in sorted(pos.keys()))


def row_from_work(w: Dict[str, Any], source: str) -> Dict[str, Any]:
    topics = [(t.get("id") or "").replace("https://openalex.org/", "") for t in (w.get("topics") or [])]
    return {
        "source": source,
        "openalex_id": oa_id_short(w.get("id", "")),
        "pmid": "",
        "arxiv_id": "",
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": (w.get("title") or w.get("display_name") or "").strip(),
        "year": str(w.get("publication_year") or ""),
        "publication_date": w.get("publication_date") or "",
        "type": w.get("type") or "",
        "venue": (((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""),
        "authors": "; ".join([((a.get("author") or {}).get("display_name") or "") for a in (w.get("authorships") or []) if ((a.get("author") or {}).get("display_name"))]),
        "cited_by": int(w.get("cited_by_count") or 0),
        "language": (w.get("language") or ""),
        "primary_domain_id": ((((w.get("primary_topic") or {}).get("domain") or {}).get("id") or "").replace("https://openalex.org/", "")),
        "primary_field_id": ((((w.get("primary_topic") or {}).get("field") or {}).get("id") or "").replace("https://openalex.org/", "")),
        "primary_topic": ((w.get("primary_topic") or {}).get("display_name") or ""),
        "topics": "; ".join([(t.get("display_name") or "") for t in (w.get("topics") or [])[:8]]),
        "concepts": "; ".join([(c.get("display_name") or "") for c in (w.get("concepts") or [])[:10]]),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index") or {}),
        "oa_pdf_url": "",
        "best_url": "",
        "score": 0.0,
        "_topic_ids": [x for x in topics if x],
    }


def row_from_ncbi(rec: Dict[str, Any], source: str) -> Dict[str, Any]:
    article_ids = rec.get("articleids") or []
    doi = ""
    arxiv = ""
    for a in article_ids:
        if (a or {}).get("idtype") == "doi":
            doi = (a or {}).get("value") or ""
        if (a or {}).get("idtype") == "arxiv":
            arxiv = (a or {}).get("value") or ""
    authors = "; ".join([(x or {}).get("name") or "" for x in (rec.get("authors") or []) if (x or {}).get("name")])
    pubdate = rec.get("pubdate") or ""
    y = re.search(r"(19|20)\d{2}", str(pubdate))
    return {
        "source": source, "openalex_id": "", "pmid": str(rec.get("_pmid") or ""), "arxiv_id": arxiv, "doi": doi,
        "title": (rec.get("title") or "").strip(), "year": y.group(0) if y else "", "publication_date": pubdate,
        "type": rec.get("pubtype") or "article", "venue": rec.get("fulljournalname") or rec.get("source") or "",
        "authors": authors, "cited_by": 0, "language": "", "primary_domain_id": "", "primary_field_id": "", "primary_topic": "",
        "topics": "", "concepts": "", "abstract": "", "oa_pdf_url": "", "best_url": "", "score": 0.0, "_topic_ids": []
    }


def extract_phrases(text: str, stopwords: Set[str], max_phrases: int = 40) -> List[str]:
    tokens = tokenize(text)
    grams: Counter = Counter()
    for n in (2, 3, 4):
        grams.update(make_ngrams(tokens, n, stopwords))
    phrases = [k for k, _ in grams.most_common(max_phrases) if len(k) >= 8]
    return phrases


def query_is_valid(query: str, stopwords: Set[str]) -> bool:
    quoted = re.findall(r'"([^"]+)"', query or "")
    if any(len(clean_spaces(q)) >= 8 for q in quoted):
        return True
    tokens = [t for t in tokenize(query) if informative_token(t, stopwords)]
    return len(tokens) >= 2


def seed_queries(terms: List[str], phrases: List[str], stopwords: Set[str], limit: int = 12) -> Tuple[List[str], Dict[str, Any]]:
    candidates: List[str] = []
    for ph in phrases[:8]:
        candidates.append(f'"{clean_spaces(ph)}"')
    for i in range(min(10, len(terms))):
        a = terms[i]
        b = terms[(i + 1) % len(terms)] if len(terms) > 1 else ""
        if a and b and a != b:
            candidates.append(f'{quote_term(a)} AND {quote_term(b)}')
    for i in range(0, min(12, len(terms) - 2), 2):
        trio = [terms[i], terms[i + 1], terms[i + 2]]
        if len({*trio}) >= 2:
            candidates.append(" AND ".join(quote_term(x) for x in trio))

    kept: List[str] = []
    seen: Set[str] = set()
    dropped = 0
    for raw in candidates:
        q = clean_spaces(raw)
        if not q or q in seen:
            continue
        seen.add(q)
        if query_is_valid(q, stopwords):
            kept.append(q)
        else:
            dropped += 1
        if len(kept) >= limit:
            break

    stats = {
        "generated": len(candidates),
        "filtered_as_noise": dropped,
        "accepted": len(kept),
        "top5": kept[:5],
    }
    return kept, stats


def ncbi_queries(terms: List[str], topn: int = 15) -> List[str]:
    blocks = [terms[0:6], terms[6:12], terms[12:18]]
    blocks = [b for b in blocks if b]
    out: List[str] = []
    if len(blocks) >= 3:
        for i in range(min(6, len(blocks[0]))):
            q = f"({blocks[0][i]}[tiab]) AND ({blocks[1][i % len(blocks[1])]}[tiab]) AND ({blocks[2][i % len(blocks[2])]}[tiab])"
            out.append(q)
    for i in range(min(6, len(terms))):
        out.append(f"({terms[i]}[tiab]) AND ({terms[(i + 1) % len(terms)]}[tiab])")
    uniq = []
    seen = set()
    for q in out:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq[:topn]


def row_tokens(r: Dict[str, Any], stopwords: Set[str]) -> Counter:
    t = tokenize(" ".join([r.get("title", ""), r.get("abstract", ""), r.get("topics", ""), r.get("concepts", "")]))
    return Counter([x for x in t if informative_token(x, stopwords)])


def build_profile(rows: List[Dict[str, Any]], stopwords: Set[str], top_n: int = 120) -> Tuple[Counter, Counter]:
    token_profile: Counter = Counter()
    topic_profile: Counter = Counter()
    for r in rows[:top_n]:
        token_profile.update(row_tokens(r, stopwords))
        topic_profile.update(r.get("_topic_ids") or [])
    return token_profile, topic_profile


def drift_score(row: Dict[str, Any], seed_tokens: Counter, seed_topics: Counter, stopwords: Set[str]) -> float:
    t_sim = cosine(row_tokens(row, stopwords), seed_tokens)
    top = Counter(row.get("_topic_ids") or [])
    topic_sim = cosine(top, seed_topics)
    return 0.75 * t_sim + 0.25 * topic_sim


def mmr_select(rows: List[Dict[str, Any]], n: int, max_per_venue: int, max_per_topic: int, mmr_lambda: float, stopwords: Set[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    venue_count: Counter = Counter()
    topic_count: Counter = Counter()
    pool = list(rows)
    while pool and len(selected) < n:
        best = None
        best_mmr = -10**9
        for r in pool[:800]:
            venue = (r.get("venue") or "").lower()
            if venue and venue_count[venue] >= max_per_venue:
                continue
            tps = r.get("_topic_ids") or []
            if tps and max(topic_count[x] for x in tps) >= max_per_topic:
                continue
            rel = float(r.get("score") or 0.0)
            sim = 0.0
            rt = row_tokens(r, stopwords)
            for s in selected[-30:]:
                sim = max(sim, cosine(rt, row_tokens(s, stopwords)))
            mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = r
        if not best:
            break
        selected.append(best)
        venue = (best.get("venue") or "").lower()
        if venue:
            venue_count[venue] += 1
        for x in (best.get("_topic_ids") or []):
            topic_count[x] += 1
        pool.remove(best)
    if len(selected) < n:
        for r in pool:
            if len(selected) >= n:
                break
            selected.append(r)
    return selected


def write_stage_summary(path: str, data: Dict[str, Any]) -> str:
    lines = [
        "=== Сводка Stage B ===",
        f"Статус: {data['status']}",
        f"Старт UTC: {data['started_utc']}",
        f"Длительность (сек): {data['elapsed_sec']}",
        f"Запросов OpenAlex: {data['requests'].get('openalex', 0)}",
        f"Запросов NCBI: {data['requests'].get('ncbi', 0)}",
        f"Запросов Crossref: {data['requests'].get('crossref', 0)}",
        f"Запросов Unpaywall: {data['requests'].get('unpaywall', 0)}",
        f"Добавлено по источникам: {json.dumps(data['source_stats'], ensure_ascii=False)}",
        f"Дедуп: до={data['dedup_before']} после={data['dedup_after']}",
        f"corpus_all: {data['corpus_all']} | corpus: {data['corpus']}",
        f"Покрытие DOI: {data['doi_coverage']:.1%}",
        f"OA rate: {data['oa_rate']:.1%}",
        f"Диверсификация: venue={data['unique_venues']} topic={data['unique_topics']}",
        f"Drift отфильтровано: {data['drift_filtered_pct']:.1%}",
        f"Причины: {' | '.join(data['errors'][:3]) if data['errors'] else 'нет'}",
    ]
    text = "\n".join(lines[:20]) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scope", choices=["balanced", "wide", "focused"], default="balanced")
    ap.add_argument("--from-year", type=int, default=1990)
    ap.add_argument("--to-year", type=int, default=2100)
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--request-cap", type=int, default=220)
    args = ap.parse_args()

    started_ts = time.time()
    started_utc = utc_now()
    idea_dir = os.path.abspath(args.idea_dir)
    out_dir = os.path.join(idea_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    root_dir = os.path.abspath(os.path.join(idea_dir, "..", ".."))
    cfg = load_sources_config(root_dir)
    secrets = get_secrets(root_dir)
    stopwords = load_stopwords(root_dir)

    log_path = os.path.join(out_dir, "module_B.log")
    search_log_path = os.path.join(out_dir, "search_log.json")
    summary_path = os.path.join(out_dir, "stageB_summary.txt")
    corpus_path = os.path.join(out_dir, "corpus.csv")
    corpus_all_path = os.path.join(out_dir, "corpus_all.csv")
    ckpt_path = os.path.join(out_dir, "_moduleB_checkpoint.json")
    field_map_path = os.path.join(out_dir, "field_map.md")
    prisma_path = os.path.join(out_dir, "prisma_lite.md")

    status = "OK"
    errors: List[str] = []
    phases: List[Dict[str, Any]] = []
    source_stats: Dict[str, int] = Counter()
    rows_all: List[Dict[str, Any]] = []
    dedup_seen: Set[str] = set()
    dedup_before = 0
    drift_filtered = 0
    query_quality: Dict[str, Any] = {"generated": 0, "filtered_as_noise": 0, "accepted": 0, "top5": []}

    def add_row(r: Dict[str, Any]) -> bool:
        nonlocal dedup_before
        dedup_before += 1
        k = dedup_key(r)
        if k in dedup_seen:
            return False
        dedup_seen.add(k)
        rows_all.append(r)
        src = (r.get("source") or "unknown").split(":")[0]
        source_stats[src] += 1
        return True

    with open(log_path, "a", encoding="utf-8") as L:
        try:
            blob = input_blob(idea_dir)
            if not blob.strip():
                raise RuntimeError("Нет входного текста идеи")
            terms = rank_terms(blob, stopwords, max_terms=120)
            phrases = extract_phrases(blob, stopwords, max_phrases=40)
            s_queries, query_quality = seed_queries(terms, phrases, stopwords, limit=12)
            phases.append({"phase": "seed", "queries": s_queries[:8], "terms": terms[:20], "phrases": phrases[:8]})

            if cfg["enable"].get("openalex") and secrets.get("OPENALEX_API_KEY"):
                oa = OAClient(api_key=secrets.get("OPENALEX_API_KEY", ""), mailto=secrets.get("OPENALEX_MAILTO", ""), rps=args.rps)
                sel = "id,doi,title,display_name,publication_year,publication_date,type,cited_by_count,language,authorships,primary_location,abstract_inverted_index,topics,concepts,primary_topic"
                base = [f"publication_year:{args.from_year}-{args.to_year}"]
                for q in s_queries:
                    if oa.request_count >= min(int(args.request_cap), int(cfg["request_caps"].get("openalex", 160))):
                        errors.append("Достигнут лимит OpenAlex request_cap")
                        status = "DEGRADED"
                        break
                    added = 0
                    for w in oa.list_works(search=q, filter_parts=base, sort="relevance_score:desc", pages=1, select_fields=sel, log_fp=L):
                        if add_row(row_from_work(w, "openalex:seed")):
                            added += 1
                    phases.append({"phase": "topic/concept_map", "query": q, "added": added})

                seed_sorted = sorted(rows_all, key=lambda r: int(r.get("cited_by") or 0), reverse=True)
                token_profile, topic_profile = build_profile(seed_sorted, stopwords, top_n=min(150, len(seed_sorted)))

                top_seed = seed_sorted[:max(30, min(cfg["relevance_feedback"]["top_n"], len(seed_sorted)))]
                grams = Counter()
                for r in top_seed:
                    tks = [t for t in tokenize(" ".join([r.get("title", ""), r.get("abstract", "")])) if informative_token(t, stopwords)]
                    grams.update(make_ngrams(tks, 2, stopwords))
                    grams.update(make_ngrams(tks, 3, stopwords))
                rf_terms = [k for k, _ in grams.most_common(cfg["relevance_feedback"]["phrases"])]
                rf_queries = [quote_term(x) for x in rf_terms[:cfg["relevance_feedback"]["queries"]]]
                for q in rf_queries:
                    if oa.request_count >= min(int(args.request_cap), int(cfg["request_caps"].get("openalex", 160))):
                        status = "DEGRADED"
                        break
                    term_clean = q.replace('"', "")
                    for w in oa.list_works(search=None, filter_parts=base + [f"title.search:{term_clean}"], sort="cited_by_count:desc", pages=1, select_fields=sel, log_fp=L):
                        r = row_from_work(w, "openalex:feedback")
                        d = drift_score(r, token_profile, topic_profile, stopwords)
                        r["_drift"] = d
                        if d < float(cfg["drift"]["threshold"]):
                            if len(rows_all) % max(1, int(1 / max(0.01, float(cfg["drift"]["explore_budget"])))) != 0:
                                drift_filtered += 1
                                continue
                        add_row(r)
                phases.append({"phase": "harvest_filter", "rf_queries": rf_queries[:8], "added_total": len(rows_all)})
                oa_requests = oa.request_count
            else:
                oa_requests = 0
                status = "DEGRADED"
                errors.append("OpenAlex отключен или отсутствует OPENALEX_API_KEY")
                phases.append({"phase": "topic/concept_map", "status": "skipped", "reason": "OpenAlex отключен"})
                phases.append({"phase": "harvest_filter", "status": "skipped", "reason": "Нет seed-корпуса OpenAlex"})

            class NCBIClient(HttpClient):
                def call(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
                    p = dict(params)
                    p["retmode"] = "json"
                    if secrets.get("NCBI_API_KEY"):
                        p["api_key"] = secrets["NCBI_API_KEY"]
                    if secrets.get("NCBI_EMAIL"):
                        p["email"] = secrets["NCBI_EMAIL"]
                    return self.get_json(NCBI_BASE + endpoint + "?" + urllib.parse.urlencode(p, quote_via=urllib.parse.quote), headers={"User-Agent": "IdeaPipeline-StageB/v10"})

            ncbi_requests = 0
            if cfg["enable"].get("ncbi"):
                try:
                    nc = NCBIClient(rps=args.rps)
                    qlist = ncbi_queries(terms, topn=12)
                    for q in qlist:
                        if nc.request_count >= int(cfg["request_caps"].get("ncbi", 80)):
                            status = "DEGRADED"
                            errors.append("Достигнут лимит NCBI")
                            break
                        ids = (((nc.call("/esearch.fcgi", {"db": "pubmed", "term": q, "retmax": "40", "sort": "relevance"}).get("esearchresult") or {}).get("idlist")) or [])
                        added = 0
                        dups = 0
                        if ids:
                            sm = nc.call("/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids)})
                            obj = sm.get("result") or {}
                            for pid in (obj.get("uids") or []):
                                rec = obj.get(pid) or {}
                                rec["_pmid"] = pid
                                if add_row(row_from_ncbi(rec, "ncbi")):
                                    added += 1
                                else:
                                    dups += 1
                        phases.append({"phase": "ncbi_search", "query": q, "found": len(ids), "added": added, "duplicates": dups})
                    ncbi_requests = nc.request_count
                except Exception as e:
                    status = "DEGRADED"
                    errors.append(f"NCBI недоступен: {type(e).__name__}: {e}")
                    phases.append({"phase": "ncbi_search", "status": "degraded", "reason": str(e)[:160]})
            else:
                phases.append({"phase": "ncbi_search", "status": "skipped", "reason": "NCBI отключен в config"})

            crossref_requests = 0
            if cfg["enable"].get("crossref") and secrets.get("CROSSREF_MAILTO"):
                cr = HttpClient(rps=max(1.0, args.rps))
                citation_events = 0
                for r in rows_all[:1000]:
                    if r.get("doi"):
                        continue
                    title = r.get("title") or ""
                    if not title:
                        continue
                    q = urllib.parse.urlencode({"query.title": title, "rows": "3", "mailto": secrets["CROSSREF_MAILTO"]})
                    try:
                        j = cr.get_json(CROSSREF_BASE + "?" + q, headers={"User-Agent": "IdeaPipeline-StageB/v10"}, log_fp=L)
                        cand = ((j.get("message") or {}).get("items") or [])
                        if not cand:
                            continue
                        c0 = cand[0]
                        doi = c0.get("DOI") or ""
                        ctitle = ((c0.get("title") or [""])[0] or "")
                        sim = cosine(Counter(tokenize(title)), Counter(tokenize(ctitle)))
                        if doi and sim >= 0.80:
                            r["doi"] = doi
                            r["_crossref_confidence"] = round(sim, 3)
                        elif doi:
                            phases.append({"phase": "citation_chase", "title": title[:120], "doi": doi, "confidence": round(sim, 3)})
                            citation_events += 1
                    except Exception as e:
                        status = "DEGRADED"
                        errors.append(f"Crossref ошибка: {type(e).__name__}")
                        break
                crossref_requests = cr.request_count
                if citation_events == 0:
                    phases.append({"phase": "citation_chase", "status": "ok", "matches": 0})
            else:
                phases.append({"phase": "citation_chase", "status": "skipped", "reason": "Crossref отключен или отсутствует CROSSREF_MAILTO"})

            seed_tokens, seed_topics = build_profile(rows_all, stopwords, top_n=160)
            for r in rows_all:
                drift = drift_score(r, seed_tokens, seed_topics, stopwords)
                text = " ".join([r.get("title", ""), r.get("abstract", ""), r.get("topics", ""), r.get("concepts", "")]).lower()
                lexical = sum(1 for t in terms[:40] if t in text)
                rec = 1.0 if str(r.get("year") or "").isdigit() and int(r.get("year") or 0) >= 2019 else 0.0
                found = math.log1p(max(0, int(r.get("cited_by") or 0))) / 6.0
                r["score"] = round(1.8 * drift + 0.6 * lexical + 0.25 * rec + 0.35 * found, 4)

            rows_ranked = sorted(rows_all, key=lambda x: (float(x.get("score") or 0.0), int(x.get("cited_by") or 0)), reverse=True)
            div = cfg["diversity"]
            final_n = args.n if args.scope != "focused" else min(300, max(200, args.n))
            final_rows = mmr_select(rows_ranked, final_n, int(div["max_per_venue"]), int(div["max_per_topic"]), float(div["mmr_lambda"]), stopwords)
            phases.append({"phase": "final_select", "picked": len(final_rows), "from_ranked": len(rows_ranked)})

            unpaywall_requests = 0
            if cfg["enable"].get("unpaywall") and secrets.get("UNPAYWALL_EMAIL"):
                up = HttpClient(rps=max(1.0, args.rps))
                for r in final_rows[:min(1000, len(final_rows))]:
                    doi = r.get("doi")
                    if not doi:
                        continue
                    try:
                        j = up.get_json(f"{UNPAYWALL_BASE}/{urllib.parse.quote(doi)}?email={urllib.parse.quote(secrets['UNPAYWALL_EMAIL'])}", log_fp=L)
                        r["oa_pdf_url"] = ((j.get("best_oa_location") or {}).get("url_for_pdf") or "")
                        r["best_url"] = ((j.get("best_oa_location") or {}).get("url") or "")
                        if r.get("oa_pdf_url") or r.get("best_url"):
                            r["score"] = round(float(r.get("score") or 0.0) + 0.15, 4)
                    except Exception:
                        status = "DEGRADED"
                unpaywall_requests = up.request_count

            final_rows = sorted(final_rows, key=lambda x: float(x.get("score") or 0.0), reverse=True)

            target_min, target_max = cfg["targets"].get(args.scope, [1500, 3000])
            corpus_all_cap = max(target_min, min(target_max, max(8000 if args.scope == "wide" else 3000, args.n * 8)))
            save_csv(corpus_path, [{k: v for k, v in r.items() if not k.startswith("_")} for r in final_rows])
            save_csv(corpus_all_path, [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_ranked[:corpus_all_cap]])

            if count_csv_rows(corpus_all_path) < int(target_min):
                status = "DEGRADED"
                errors.append(f"Не достигнут минимальный объём corpus_all для режима {args.scope}")

            with open(field_map_path, "w", encoding="utf-8") as f:
                f.write("# Карта поля (Stage B)\n\n")
                f.write(f"- Версия: {VERSION}\n")
                f.write(f"- Статус: {status}\n")
                f.write(f"- Топ-термы seed: {', '.join(terms[:20])}\n")
                f.write(f"- Уникальных topics: {len(set(sum([r.get('_topic_ids', []) for r in rows_all], [])))}\n")

            with open(prisma_path, "w", encoding="utf-8") as f:
                f.write("# PRISMA-lite (Stage B)\n\n")
                f.write(f"- Статус: {status}\n")
                f.write(f"- Добавлено: {dict(source_stats)}\n")
                f.write(f"- До дедуп: {dedup_before}\n")
                f.write(f"- После дедуп: {len(dedup_seen)}\n")
                f.write(f"- corpus_all: {count_csv_rows(corpus_all_path)}\n")
                f.write(f"- corpus: {count_csv_rows(corpus_path)}\n")

            search_log = {
                "module": "B", "version": VERSION, "datetime_utc": utc_now(), "status": status,
                "target_n": final_n, "requests": {"openalex": oa_requests, "ncbi": ncbi_requests, "crossref": crossref_requests, "unpaywall": unpaywall_requests},
                "source_stats": dict(source_stats), "dedup": {"before": dedup_before, "after": len(dedup_seen)},
                "seed_queries": s_queries, "query_quality": query_quality, "errors": errors, "phases": phases,
            }
            with open(search_log_path, "w", encoding="utf-8") as f:
                json.dump(search_log, f, ensure_ascii=False, indent=2)

            ckpt = {"version": VERSION, "input_hash": hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest(), "scope": args.scope,
                    "rows_doi": [{k: v for k, v in r.items() if not k.startswith("_")} for r in final_rows if r.get("doi")],
                    "rows_all": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_ranked[:max(args.n, 1500)]], "search_log": search_log}
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)

        except Exception as e:
            status = "FAILED" if status == "OK" else status
            errors.append(f"FATAL: {type(e).__name__}: {e}")
            log(L, f"[ERR] Необработанная ошибка: {type(e).__name__}: {e}")
            save_csv(corpus_path, [])
            save_csv(corpus_all_path, [])
            with open(search_log_path, "w", encoding="utf-8") as f:
                json.dump({"module": "B", "version": VERSION, "status": status, "errors": errors, "phases": phases}, f, ensure_ascii=False, indent=2)
        finally:
            oa_rate = 0.0
            if rows_all:
                oa_rate = sum(1 for r in final_rows if r.get("oa_pdf_url") or r.get("best_url")) / max(1, len(final_rows))
            doi_cov = sum(1 for r in final_rows if r.get("doi")) / max(1, len(final_rows))
            uniq_venues = len(set((r.get("venue") or "").strip().lower() for r in final_rows if (r.get("venue") or "").strip()))
            uniq_topics = len(set(sum([r.get("_topic_ids") or [] for r in final_rows], [])))
            summary = write_stage_summary(summary_path, {
                "status": status,
                "started_utc": started_utc,
                "elapsed_sec": round(time.time() - started_ts, 1),
                "requests": {
                    "openalex": oa_requests if 'oa_requests' in locals() else 0,
                    "ncbi": ncbi_requests if 'ncbi_requests' in locals() else 0,
                    "crossref": crossref_requests if 'crossref_requests' in locals() else 0,
                    "unpaywall": unpaywall_requests if 'unpaywall_requests' in locals() else 0,
                },
                "source_stats": dict(source_stats),
                "dedup_before": dedup_before,
                "dedup_after": len(dedup_seen),
                "corpus_all": count_csv_rows(corpus_all_path),
                "corpus": count_csv_rows(corpus_path),
                "doi_coverage": doi_cov,
                "oa_rate": oa_rate,
                "unique_venues": uniq_venues,
                "unique_topics": uniq_topics,
                "drift_filtered_pct": drift_filtered / max(1, dedup_before),
                "errors": errors,
            })

    return 0 if status in ("OK", "DEGRADED") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(f"[FATAL] {type(e).__name__}: {e}\n")
        raise SystemExit(1)
