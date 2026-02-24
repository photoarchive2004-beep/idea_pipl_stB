#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage B — Literature Scout (OpenAlex + NCBI) — v9.0

Strategy:
1) SEED ("pearl growing"): boolean search over Works.search.
2) Infer field: top primary_topic.domain + top topics.id from seed.
3) EXPAND via filters (topics.id / primary_topic.domain.id) sorted by recency and citations.
4) CITATION CHASING using filters:
   - cites:W... (incoming citations)
   - cited_by:W... (outgoing citations)
5) Quality gates: min corpus size + anchor hit-rate.

Outputs (idea_dir/out):
- corpus.csv (DOI only)
- corpus_all.csv (includes works without DOI)
- search_log.json
- field_map.md
- prisma_lite.md
- module_B.log
- _moduleB_checkpoint.json
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys, time, urllib.parse, urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

BASE = "https://api.openalex.org"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
VERSION = "v9.0-multisource"

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def now_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(fp, msg: str) -> None:
    fp.write(f"[{now_local()}] {msg}\n"); fp.flush()

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

def get_api_key(root_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    env = read_env_file(os.path.join(root_dir, "config", "secrets.env"))
    key = env.get("OPENALEX_API_KEY") or os.environ.get("OPENALEX_API_KEY")
    mailto = env.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_MAILTO")
    ncbi_key = env.get("NCBI_API_KEY") or os.environ.get("NCBI_API_KEY")
    return key, mailto, ncbi_key

def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def has_cyr(s: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", s or ""))

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def oa_id_short(x: str) -> str:
    if not x:
        return ""
    m = re.search(r"/(W\d+)$", x)
    if m:
        return m.group(1)
    m = re.match(r"^(W\d+)$", x)
    return m.group(1) if m else x

def reconstruct_abstract(inv: Dict[str, List[int]]) -> str:
    if not inv:
        return ""
    pos: Dict[int, str] = {}
    for w, ps in inv.items():
        for p in ps:
            try:
                pos[int(p)] = w
            except Exception:
                pass
    if not pos:
        return ""
    return " ".join(pos[i] for i in sorted(pos.keys())).replace(" ,", ",").replace(" .", ".")

@dataclass
class OAClient:
    api_key: str
    mailto: Optional[str] = None
    rps: float = 2.0
    timeout: int = 40
    max_retries: int = 2
    user_agent: str = "IdeaPipeline-StageB/v8.0"
    request_count: int = 0

    def _sleep(self) -> None:
        if self.rps and self.rps > 0:
            time.sleep(1.0 / float(self.rps))

    def build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        params = dict(params)
        params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        return BASE + endpoint + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

    def get_json(self, url: str, log_fp=None) -> Dict[str, Any]:
        last = None
        for attempt in range(self.max_retries):
            try:
                self._sleep()
                self.request_count += 1
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                return json.loads(data.decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = e
                wait = min(40, 2 ** attempt)
                if log_fp:
                    log(log_fp, f"[WARN] HTTP {e.code} {e.reason}; retry {wait}s; url={url}")
                time.sleep(wait)
            except Exception as e:
                last = e
                wait = min(40, 2 ** attempt)
                if log_fp:
                    log(log_fp, f"[WARN] NET {type(e).__name__}: {e}; retry {wait}s; url={url}")
                time.sleep(wait)
        raise RuntimeError(f"HTTP failed after retries: {last}")

    def list_works(self, *, search: Optional[str], filter_parts: List[str], sort: Optional[str],
                   per_page: int, max_pages: int, select_fields: str, log_fp=None) -> Iterable[Dict[str, Any]]:
        cursor = "*"
        pages = 0
        while cursor and pages < max_pages:
            pages += 1
            params: Dict[str, Any] = {"per-page": str(min(max(per_page, 1), 200)), "cursor": cursor}
            if search:
                params["search"] = search
            if filter_parts:
                params["filter"] = ",".join(filter_parts)
            if sort:
                params["sort"] = sort
            if select_fields:
                params["select"] = select_fields
            url = self.build_url("/works", params)
            j = self.get_json(url, log_fp=log_fp)
            res = j.get("results") or []
            cursor = (j.get("meta") or {}).get("next_cursor")
            if not res:
                break
            for w in res:
                yield w

@dataclass
class NCBIClient:
    api_key: Optional[str] = None
    mailto: Optional[str] = None
    rps: float = 2.0
    timeout: int = 40
    max_retries: int = 2
    request_count: int = 0

    def _sleep(self) -> None:
        if self.rps and self.rps > 0:
            time.sleep(1.0 / float(self.rps))

    def _call(self, endpoint: str, params: Dict[str, Any], log_fp=None) -> Dict[str, Any]:
        p = dict(params)
        p["retmode"] = "json"
        if self.api_key:
            p["api_key"] = self.api_key
        if self.mailto:
            p["email"] = self.mailto
        url = NCBI_BASE + endpoint + "?" + urllib.parse.urlencode(p, quote_via=urllib.parse.quote)
        last = None
        for attempt in range(self.max_retries):
            try:
                self._sleep()
                self.request_count += 1
                req = urllib.request.Request(url, headers={"User-Agent": "IdeaPipeline-StageB/v9.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                return json.loads(data.decode("utf-8"))
            except Exception as e:
                last = e
                wait = min(40, 2 ** attempt)
                if log_fp:
                    log(log_fp, f"[WARN] NCBI {type(e).__name__}: {e}; retry {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"NCBI failed after retries: {last}")

    def esearch(self, query: str, retmax: int = 60) -> List[str]:
        j = self._call("/esearch.fcgi", {"db": "pubmed", "term": query, "retmax": str(retmax), "sort": "relevance"})
        return (((j.get("esearchresult") or {}).get("idlist")) or [])

    def esummary(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        j = self._call("/esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids)})
        result = j.get("result") or {}
        out = []
        for pid in (result.get("uids") or []):
            rec = result.get(pid) or {}
            rec["_pmid"] = pid
            out.append(rec)
        return out

RU2EN = {
    "геномика":"genomics","геномный":"genomic","популяция":"population","популяционный":"population",
    "адаптация":"adaptation","интрогрессия":"introgression","демография":"demography",
    "связность":"connectivity","изоляция":"isolation","дистанция":"distance","сопротивление":"resistance",
    "расселение":"range expansion","рефугиум":"refugium",
}
STOP = set("""the and for with from into about this that those these have has had were was are is be been being
study studies result results method methods data analysis analyses using use used based across between
""".split())
GENERIC_METHOD = set("""bayesian framework frameworks model models algorithm algorithms machine learning deep learning neural network
inference estimation pipeline approach approaches""".split())

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]{2,}", text or "")

def find_binomials(text: str, limit: int = 8) -> List[str]:
    pat = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})(?:\s+([a-z]{3,}))?\b")
    out=[]
    for m in pat.finditer(text or ""):
        g,s,ss=m.group(1),m.group(2),m.group(3)
        out.append(f"{g} {s}" + (f" {ss}" if ss else ""))
        if len(out)>=limit:
            break
    seen=set(); u=[]
    for x in out:
        if x not in seen:
            u.append(x); seen.add(x)
    return u

def norm_tok(t: str) -> str:
    t = (t or "").strip().lower()
    if not t or t in STOP:
        return ""
    if has_cyr(t):
        t = RU2EN.get(t, "")
        if not t:
            return ""
    t = clean_spaces(t)
    if not t or t in STOP:
        return ""
    return t

def rank_terms(text: str, max_terms: int = 50) -> List[str]:
    freq: Dict[str, int] = {}
    for tok in tokenize(text):
        nt = norm_tok(tok)
        if not nt:
            continue
        if re.fullmatch(r"\d+", nt):
            continue
        freq[nt] = freq.get(nt, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k,_ in ranked[:max_terms]]

def quote_term(t: str) -> str:
    t = clean_spaces(t)
    if " " in t or "-" in t:
        return f"\"{t}\""
    return t

def split_domain_method(terms: List[str], binom: List[str]) -> Tuple[List[str], List[str]]:
    domain=[]; method=[]
    for x in terms:
        if x in GENERIC_METHOD:
            method.append(x)
        else:
            domain.append(x)
    for b in binom[:3]:
        if b.lower() not in [d.lower() for d in domain]:
            domain.insert(0, b)
    def uniq(xs):
        out=[]; seen=set()
        for z in xs:
            z = clean_spaces(z)
            if not z:
                continue
            k=z.lower()
            if k in seen:
                continue
            out.append(z); seen.add(k)
        return out
    return uniq(domain)[:10], uniq(method)[:12]

def build_seed_queries(domain: List[str], method: List[str]) -> List[str]:
    method2=[m for m in method if m.lower() not in GENERIC_METHOD]
    if not domain:
        domain = ["research"]
    dom_q = [quote_term(d) for d in domain[:6]]
    if len(dom_q) >= 2:
        dom_group = "(" + " OR ".join(dom_q[:4]) + ")"
    else:
        dom_group = dom_q[0]
    safe_methods = [quote_term(m) for m in method2[:6]]
    safe_methods += ["genomics", "phylogeography", "meta analysis", "systematic review",
                      "causal inference", "network analysis", "time series", "cohort", "benchmark", "simulation"]
    seen=set(); mp=[]
    for x in safe_methods:
        k=x.lower()
        if k in seen:
            continue
        mp.append(x); seen.add(k)
        if len(mp) >= 10:
            break
    met_group = "(" + " OR ".join(mp) + ")"
    q1 = f"{dom_group} AND {met_group}"
    q2 = f"{dom_group} AND (review OR \"meta-analysis\" OR overview)"
    qs=[q1,q2]
    for b in dom_q[:3]:
        if b.startswith("\"") and b.endswith("\""):
            qs.append(f"{b} AND (genomics OR phylogeography OR population)")
    out=[]; s=set()
    for q in qs:
        q = clean_spaces(q)
        if q and q not in s:
            out.append(q); s.add(q)
    return out[:12]

CSV_COLS = ["source","openalex_id","pmid","arxiv_id","doi","title","year","publication_date","type","venue","authors","cited_by","language",
            "primary_domain_id","primary_field_id","primary_topic","topics","concepts","abstract"]

def extract_authors(w: Dict[str,Any], n: int = 25) -> str:
    names=[]
    for a in (w.get("authorships") or [])[:n]:
        dn = ((a or {}).get("author") or {}).get("display_name")
        if dn:
            names.append(dn)
    return "; ".join(names)

def extract_venue(w: Dict[str,Any]) -> str:
    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    return (src.get("display_name") or "")

def extract_topics(w: Dict[str,Any], k: int = 8) -> Tuple[List[str], List[str]]:
    ids=[]; names=[]
    for t in (w.get("topics") or [])[:k]:
        tid = (t or {}).get("id") or ""
        nm  = (t or {}).get("display_name") or ""
        if tid:
            ids.append(tid.replace("https://openalex.org/",""))
        if nm:
            names.append(nm)
    return ids, names

def extract_concepts(w: Dict[str,Any], k: int = 12) -> Tuple[List[str], List[str]]:
    cs = w.get("concepts") or []
    try:
        cs = sorted(cs, key=lambda c: (c or {}).get("score",0), reverse=True)
    except Exception:
        pass
    ids=[]; names=[]
    for c in cs[:k]:
        cid = (c or {}).get("id") or ""
        nm  = (c or {}).get("display_name") or ""
        if cid:
            ids.append(cid.replace("https://openalex.org/",""))
        if nm:
            names.append(nm)
    return ids, names

def row_from_work(w: Dict[str,Any], source: str) -> Dict[str,Any]:
    oid = oa_id_short(w.get("id","") or w.get("openalex_id",""))
    doi = (w.get("doi") or "").replace("https://doi.org/","").strip()
    title = (w.get("title") or w.get("display_name") or "").strip()
    year = w.get("publication_year") or ""
    pdate = w.get("publication_date") or ""
    wtype = w.get("type") or ""
    cited = int(w.get("cited_by_count") or 0)
    lang = (w.get("language") or "").lower()
    pt = w.get("primary_topic") or {}
    pt_name = pt.get("display_name") or ""
    dom_id = ((pt.get("domain") or {}).get("id") or "").replace("https://openalex.org/","")
    fld_id = ((pt.get("field") or {}).get("id") or "").replace("https://openalex.org/","")
    t_ids, t_names = extract_topics(w, k=8)
    c_ids, c_names = extract_concepts(w, k=12)
    abstr = reconstruct_abstract(w.get("abstract_inverted_index") or {})
    return {
        "source":source,"openalex_id":oid,"pmid":"","arxiv_id":"","doi":doi,"title":title,"year":year,"publication_date":pdate,"type":wtype,
        "venue":extract_venue(w),"authors":extract_authors(w),"cited_by":cited,"language":lang,
        "primary_domain_id":dom_id,"primary_field_id":fld_id,"primary_topic":pt_name,
        "topics":"; ".join(t_names),"concepts":"; ".join(c_names),"abstract":abstr,
        "_topic_ids":t_ids
    }


def row_from_ncbi(rec: Dict[str,Any], source: str) -> Dict[str,Any]:
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
    year = ""
    m = re.search(r"(19|20)\d{2}", str(pubdate))
    if m:
        year = m.group(0)
    return {
        "source": source,
        "openalex_id": "",
        "pmid": str(rec.get("_pmid") or ""),
        "arxiv_id": arxiv,
        "doi": doi.strip(),
        "title": (rec.get("title") or "").strip(),
        "year": year,
        "publication_date": pubdate,
        "type": rec.get("pubtype") or "article",
        "venue": rec.get("fulljournalname") or rec.get("source") or "",
        "authors": authors,
        "cited_by": 0,
        "language": "",
        "primary_domain_id": "",
        "primary_field_id": "",
        "primary_topic": "",
        "topics": "",
        "concepts": "",
        "abstract": "",
        "_topic_ids": []
    }

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

def save_csv(path: str, rows: List[Dict[str,Any]]) -> None:
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in CSV_COLS})

def build_anchor_set(domain_terms: List[str], method_terms: List[str]) -> Set[str]:
    a=set()
    for x in domain_terms[:10]:
        a.add(x.lower())
    for x in method_terms[:12]:
        if x.lower() in GENERIC_METHOD:
            continue
        a.add(x.lower())
    for x in ["genomics","phylogeography","population","introgression","demography","connectivity","gene flow","geographic","systematic review","meta analysis"]:
        a.add(x.lower())
    return a

def score_row(r: Dict[str,Any], anchors: Set[str]) -> int:
    text = " ".join([r.get("title",""), r.get("abstract",""), r.get("topics",""), r.get("concepts","")]).lower()
    hits=0
    for a in anchors:
        if a and a in text:
            hits += 1
    t = (r.get("title","") or "").lower()
    for a in anchors:
        if a and a in t:
            hits += 2
    return hits

def top_domain(rows: List[Dict[str,Any]]) -> Tuple[str,float,Dict[str,int]]:
    cnt={}
    for r in rows:
        d = (r.get("primary_domain_id") or "").strip()
        if not d:
            continue
        cnt[d]=cnt.get(d,0)+1
    if not cnt:
        return ("",0.0,cnt)
    total=sum(cnt.values())
    best=max(cnt.items(), key=lambda kv: kv[1])
    return (best[0], best[1]/max(total,1), cnt)

def top_topic_ids(rows: List[Dict[str,Any]], topk: int = 12) -> List[str]:
    cnt={}
    for r in rows:
        for tid in r.get("_topic_ids", []) or []:
            if not tid:
                continue
            cnt[tid]=cnt.get(tid,0)+1
    return [k for k,_ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]]

def input_blob(idea_dir: str) -> str:
    out_dir=os.path.join(idea_dir,"out")
    parts=[
        read_text(os.path.join(out_dir,"structured_idea.json")),
        read_text(os.path.join(idea_dir,"idea.txt")),
        read_text(os.path.join(idea_dir,"in","idea.txt")),
    ]
    return "\n".join([p for p in parts if p and p.strip()])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scope", choices=["balanced","wide","focused"], default="balanced")
    ap.add_argument("--from-year", type=int, default=1990)
    ap.add_argument("--to-year", type=int, default=2100)
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--request-cap", type=int, default=220)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--min-keep", type=int, default=80)
    ap.add_argument("--min-anchor-hit", type=float, default=0.20)
    args = ap.parse_args()

    idea_dir=os.path.abspath(args.idea_dir)
    out_dir=os.path.join(idea_dir,"out")
    os.makedirs(out_dir, exist_ok=True)

    log_path=os.path.join(out_dir,"module_B.log")
    root_dir=os.path.abspath(os.path.join(idea_dir,"..",".."))
    api_key, mailto, ncbi_key = get_api_key(root_dir)

    with open(log_path,"a",encoding="utf-8") as L:
        log(L, f"Stage B {VERSION} старт UTC={utc_now()} scope={args.scope} target={args.n}")
        blob = input_blob(idea_dir)
        if not blob.strip():
            log(L, "[ERR] Нет входного текста")
            return 2

        status = "OK"
        errors = []
        ih = sha256(blob)
        ckpt_path=os.path.join(out_dir,"_moduleB_checkpoint.json")
        if os.path.exists(ckpt_path) and not args.fresh:
            try:
                ck=json.load(open(ckpt_path,"r",encoding="utf-8"))
                if ck.get("version")==VERSION and ck.get("input_hash")==ih and ck.get("scope")==args.scope and int(ck.get("target_n",0))==int(args.n):
                    rows = ck.get("rows_doi",[]) or []
                    if len(rows) >= min(int(args.n), max(50,int(args.min_keep))):
                        log(L, f"[INFO] Использован checkpoint rows_doi={len(rows)}")
                        return 0
            except Exception:
                pass

        rows_all=[]
        dedup_seen=set()
        phases=[]
        source_stats={"openalex":0,"ncbi":0}

        def add_row(r: Dict[str,Any]) -> bool:
            k = dedup_key(r)
            if k in dedup_seen:
                return False
            dedup_seen.add(k)
            rows_all.append(r)
            src = "ncbi" if "ncbi" in str(r.get("source","")).lower() else "openalex"
            source_stats[src] = source_stats.get(src, 0) + 1
            return True

        domain_terms=[]
        method_terms=[]
        tgate=[]
        dom_gate=""
        dom_share=0.0
        dom_cnt={}
        oa_requests=0
        ncbi_requests=0
        seed_queries=[]

        if api_key:
            try:
                client = OAClient(api_key=api_key, mailto=mailto, rps=float(args.rps), user_agent="IdeaPipeline-StageB/v9.0")
                binom = find_binomials(blob, limit=8)
                terms = rank_terms(blob, max_terms=60)
                domain_terms, method_terms = split_domain_method(terms, binom)
                seed_queries = build_seed_queries(domain_terms, method_terms)
                seed_queries = seed_queries[:20]

                year_filter = f"publication_year:{int(args.from_year)}-{int(args.to_year)}"
                type_filter = "type:article|review"
                if args.scope == "wide":
                    type_filter = "type:article|review|preprint|book-chapter"
                base_filters=[year_filter, type_filter]
                select_fields=",".join([
                    "id","doi","title","display_name","publication_year","publication_date","type","cited_by_count","language",
                    "authorships","primary_location","abstract_inverted_index","topics","concepts","primary_topic"
                ])

                def run_openalex_phase(name: str, queries: List[str], sort: str, pages_each: int):
                    ph={"phase":name,"source":"openalex","queries":[],"requests_start":client.request_count}
                    for q in queries:
                        before = len(rows_all)
                        collected = 0
                        for w in client.list_works(search=q, filter_parts=base_filters, sort=sort,
                                                   per_page=200, max_pages=pages_each, select_fields=select_fields, log_fp=L):
                            if add_row(row_from_work(w, source=name)):
                                collected += 1
                            if len(rows_all) >= int(args.n) * 2:
                                break
                        ph["queries"].append({"query":q,"added":collected})
                        if len(rows_all) >= int(args.n) * 2:
                            break
                    ph["requests_end"] = client.request_count
                    ph["added_total"] = len(rows_all) - before
                    phases.append(ph)

                run_openalex_phase("seed", seed_queries[:6], "relevance_score:desc", 2)
                dom_id, dom_share, dom_cnt = top_domain(rows_all)
                if dom_id and dom_share >= 0.40:
                    dom_gate = dom_id
                    base_filters.append(f"primary_topic.domain.id:{dom_gate}")
                tgate = top_topic_ids(rows_all, topk=10)
                if tgate:
                    base_filters.append("topics.id:" + "|".join(tgate[:10]))

                harvest_queries = seed_queries[6:18] if len(seed_queries) > 6 else seed_queries[:6]
                run_openalex_phase("harvest_filter", harvest_queries[:10], "publication_date:desc", 1)
                run_openalex_phase("harvest_filter", harvest_queries[:10], "cited_by_count:desc", 1)

                seeds_sorted = sorted([r for r in rows_all if r.get("openalex_id")], key=lambda r: -int(r.get("cited_by") or 0))
                for seed in seeds_sorted[:3]:
                    wid = seed.get("openalex_id")
                    for filt in [f"cites:{wid}", f"cited_by:{wid}"]:
                        ph={"phase":"citation_chase","source":"openalex","filter":filt,"requests_start":client.request_count}
                        added=0
                        for w in client.list_works(search=None, filter_parts=base_filters+[filt], sort="cited_by_count:desc",
                                                   per_page=200, max_pages=1, select_fields=select_fields, log_fp=L):
                            if add_row(row_from_work(w, source="citation_chase")):
                                added += 1
                        ph["requests_end"] = client.request_count
                        ph["added_total"] = added
                        phases.append(ph)
                oa_requests = client.request_count
            except Exception as e:
                status = "DEGRADED"
                errors.append(f"OpenAlex: {type(e).__name__}: {e}")
                phases.append({"phase":"seed","source":"openalex","status":"error","reason":str(e)})
                phases.append({"phase":"harvest_filter","source":"openalex","status":"error","reason":"пропущено из-за ошибки seed"})
        else:
            status = "DEGRADED"
            errors.append("OpenAlex: отсутствует OPENALEX_API_KEY")
            phases.append({"phase":"seed","source":"openalex","status":"error","reason":"нет OPENALEX_API_KEY"})
            phases.append({"phase":"harvest_filter","source":"openalex","status":"error","reason":"пропущено: нет OPENALEX_API_KEY"})

        try:
            nc = NCBIClient(api_key=ncbi_key, mailto=mailto, rps=float(args.rps))
            ncbi_terms = domain_terms[:5] + method_terms[:5]
            if not ncbi_terms:
                ncbi_terms = rank_terms(blob, max_terms=10)[:8]
            query = "(" + " OR ".join([quote_term(x) for x in ncbi_terms[:8]]) + ")"
            ph={"phase":"ncbi_search","source":"ncbi","query":query,"requests_start":nc.request_count}
            ids = nc.esearch(query, retmax=min(120, int(args.n)*2))
            added = 0
            for i in range(0, len(ids), 40):
                pack = ids[i:i+40]
                for rec in nc.esummary(pack):
                    if add_row(row_from_ncbi(rec, source="ncbi_search")):
                        added += 1
            ph["requests_end"] = nc.request_count
            ph["pmids"] = len(ids)
            ph["added_total"] = added
            phases.append(ph)
            ncbi_requests = nc.request_count
        except Exception as e:
            status = "DEGRADED"
            errors.append(f"NCBI: {type(e).__name__}: {e}")
            phases.append({"phase":"ncbi_search","source":"ncbi","status":"error","reason":str(e)})

        anchors = build_anchor_set(domain_terms, method_terms)
        for r in rows_all:
            r["_score"] = score_row(r, anchors)
        rows_sorted = sorted(rows_all, key=lambda r: (int(r.get("_score") or 0), int(r.get("cited_by") or 0), str(r.get("publication_date") or "")), reverse=True)

        target_n = int(args.n)
        rows_doi = [r for r in rows_sorted if r.get("doi")]
        final_rows = rows_doi[:target_n] if len(rows_doi) >= int(args.min_keep) else rows_sorted[:target_n]
        top100 = final_rows[:100]
        hit = sum(1 for r in top100 if int(r.get("_score") or 0) >= 2)
        hit_rate = hit / max(1, len(top100))
        phases.append({"phase":"final_select","source":"pipeline","selected":len(final_rows),"doi_selected":len([r for r in final_rows if r.get('doi')])})

        def strip(rs):
            out=[]
            for r in rs:
                rr=dict(r)
                rr.pop("_score", None)
                rr.pop("_topic_ids", None)
                out.append(rr)
            return out

        save_csv(os.path.join(out_dir,"corpus.csv"), strip(final_rows))

        search_log = {
            "module":"B","version":VERSION,"datetime_utc":utc_now(),"status":status,
            "target_n":target_n,"kept":len(final_rows),"kept_doi":len([r for r in final_rows if r.get("doi")]),
            "requests":{"openalex":oa_requests,"ncbi":ncbi_requests},
            "seed_queries":seed_queries,"domain_terms":domain_terms,"method_terms":method_terms,
            "domain_gate":dom_gate,"domain_share_seed":dom_share,"domain_counts_seed":dom_cnt,
            "topic_gate_ids":[f"https://openalex.org/{x}" for x in tgate[:10]] if tgate else [],
            "quality":{"anchor_hit_rate_top100":hit_rate,"top100_hits_ge2":hit,"anchors_count":len(anchors)},
            "source_stats":source_stats,
            "dedup_keys":len(dedup_seen),
            "errors":errors,
            "phases":phases,
        }
        with open(os.path.join(out_dir,"search_log.json"),"w",encoding="utf-8") as f:
            json.dump(search_log, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir,"field_map.md"),"w",encoding="utf-8") as f:
            f.write("# Карта поля (Stage B)\n\n")
            f.write(f"- Версия: {VERSION}\n")
            f.write(f"- Статус: **{status}**\n")
            f.write(f"- OpenAlex gate domain: `{dom_gate or 'OFF'}` share={dom_share:.2f}\n")
            f.write(f"- Top topics gate: {len(tgate[:10]) if tgate else 0}\n")

        with open(os.path.join(out_dir,"prisma_lite.md"),"w",encoding="utf-8") as f:
            f.write("# PRISMA-lite (Stage B)\n\n")
            f.write(f"- Источники: OpenAlex, NCBI\n")
            f.write(f"- Статус: {status}\n")
            f.write(f"- Добавлено OpenAlex: {source_stats.get('openalex',0)}\n")
            f.write(f"- Добавлено NCBI: {source_stats.get('ncbi',0)}\n")
            f.write(f"- В итоговом корпусе: {len(final_rows)}\n")

        ckpt={"version":VERSION,"input_hash":ih,"scope":args.scope,"target_n":target_n,
              "rows_doi":strip(final_rows),"rows_all":strip(rows_sorted[:max(target_n,500)]),"search_log":search_log}
        with open(os.path.join(out_dir,"_moduleB_checkpoint.json"),"w",encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)

        if len(final_rows) < int(args.min_keep):
            status = "DEGRADED"
            log(L, f"[WARN] Мало записей: {len(final_rows)} < min_keep={int(args.min_keep)}")
        if len(top100) >= 50 and hit_rate < float(args.min_anchor_hit):
            status = "DEGRADED"
            log(L, f"[WARN] Низкая релевантность: {hit_rate:.2f} < {float(args.min_anchor_hit):.2f}")

        if status == "OK":
            log(L, "[OK] Stage B завершён")
            return 0
        log(L, "[WARN] Stage B завершён в режиме DEGRADED")
        return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(f"[FATAL] {type(e).__name__}: {e}\n")
        raise SystemExit(1)
