#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage B — Literature Scout (OpenAlex) — v8.0 (best-practice search)

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
VERSION = "v8.0-best"

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

def get_api_key(root_dir: str) -> Tuple[Optional[str], Optional[str]]:
    env = read_env_file(os.path.join(root_dir, "config", "secrets.env"))
    key = env.get("OPENALEX_API_KEY") or os.environ.get("OPENALEX_API_KEY")
    mailto = env.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_MAILTO")
    return key, mailto

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
    max_retries: int = 6
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

RU2EN = {
    "река":"river","реки":"river","озеро":"lake","рыба":"fish","рыбы":"fish",
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
    safe_methods += [ "\"landscape genomics\"", "\"riverscape genetics\"", "genomics", "phylogeography",
                      "\"population genomics\"", "introgression", "demography", "connectivity", "GEA", "IBD", "IBR" ]
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
    return out[:8]

CSV_COLS = ["source","openalex_id","doi","title","year","publication_date","type","venue","authors","cited_by","language",
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
        "source":source,"openalex_id":oid,"doi":doi,"title":title,"year":year,"publication_date":pdate,"type":wtype,
        "venue":extract_venue(w),"authors":extract_authors(w),"cited_by":cited,"language":lang,
        "primary_domain_id":dom_id,"primary_field_id":fld_id,"primary_topic":pt_name,
        "topics":"; ".join(t_names),"concepts":"; ".join(c_names),"abstract":abstr,
        "_topic_ids":t_ids
    }

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
    for x in ["genomics","phylogeography","population","introgression","demography","connectivity","landscape genomics","riverscape genetics","gene flow","geographic"]:
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
    ap.add_argument("--keep-no-doi", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--min-keep", type=int, default=80)
    ap.add_argument("--min-anchor-hit", type=float, default=0.20)
    args = ap.parse_args()

    idea_dir=os.path.abspath(args.idea_dir)
    out_dir=os.path.join(idea_dir,"out")
    os.makedirs(out_dir, exist_ok=True)

    log_path=os.path.join(out_dir,"module_B.log")
    root_dir=os.path.abspath(os.path.join(idea_dir,"..",".."))
    api_key, mailto = get_api_key(root_dir)

    with open(log_path,"a",encoding="utf-8") as L:
        log(L, f"Stage B {VERSION} start UTC={utc_now()} scope={args.scope} target={args.n}")

        if not api_key:
            log(L, "[ERR] OPENALEX_API_KEY missing in config/secrets.env or env var")
            return 2

        blob = input_blob(idea_dir)
        if not blob.strip():
            log(L, "[ERR] No input text found")
            return 2
        ih = sha256(blob)

        ckpt_path=os.path.join(out_dir,"_moduleB_checkpoint.json")
        if os.path.exists(ckpt_path) and not args.fresh:
            try:
                ck=json.load(open(ckpt_path,"r",encoding="utf-8"))
                if ck.get("version")==VERSION and ck.get("input_hash")==ih and ck.get("scope")==args.scope and int(ck.get("target_n",0))==int(args.n):
                    rows = ck.get("rows_doi",[]) or []
                    if len(rows) >= min(int(args.n), max(50,int(args.min_keep))):
                        log(L, f"[INFO] Using checkpoint rows_doi={len(rows)}")
                        return 0
            except Exception:
                pass

        client = OAClient(api_key=api_key, mailto=mailto, rps=float(args.rps), user_agent="IdeaPipeline-StageB/v8.0")

        binom = find_binomials(blob, limit=8)
        terms = rank_terms(blob, max_terms=60)
        domain_terms, method_terms = split_domain_method(terms, binom)
        seed_queries = build_seed_queries(domain_terms, method_terms)

        log(L, "[INFO] Domain anchors: " + " | ".join(domain_terms[:8]))
        log(L, "[INFO] Method hints: " + " | ".join(method_terms[:10]))
        log(L, "[INFO] Seed queries: " + " || ".join(seed_queries))

        year_filter = f"publication_year:{int(args.from_year)}-{int(args.to_year)}"
        type_filter = "type:article|review"
        if args.scope == "wide":
            type_filter = "type:article|review|preprint|book-chapter"
        base_filters=[year_filter, type_filter]

        select_fields=",".join([
            "id","doi","title","display_name","publication_year","publication_date","type","cited_by_count","language",
            "authorships","primary_location","abstract_inverted_index","topics","concepts","primary_topic"
        ])

        rows_all=[]; rows_doi=[]; seen=set()
        phases=[]

        def add_work(w: Dict[str,Any], source: str):
            r = row_from_work(w, source=source)
            oid = r.get("openalex_id","")
            if not oid or oid in seen:
                return
            seen.add(oid)
            rows_all.append(r)
            if args.keep_no_doi or r.get("doi",""):
                rows_doi.append(r)

        def run_search(name: str, qs: List[str], sort: str, pages_each: int, target: int):
            ph={"phase":name,"queries":[],"requests_start":client.request_count,"requests_end":None,
                "rows_doi_start":len(rows_doi),"rows_doi_end":None}
            for q in qs:
                if client.request_count >= int(args.request_cap):
                    break
                if len(rows_doi) >= target and name != "seed":
                    break
                kept0=len(rows_doi); all0=len(rows_all); seen0=len(seen)
                collected_seen=0
                for w in client.list_works(search=q, filter_parts=base_filters, sort=sort,
                                           per_page=200, max_pages=pages_each, select_fields=select_fields, log_fp=L):
                    before=len(seen)
                    add_work(w, source=name)
                    if len(seen) > before:
                        collected_seen += 1
                    if len(rows_doi) >= target and name != "seed":
                        break
                ph["queries"].append({"query":q,"collected_seen":collected_seen,
                                      "kept_doi":len(rows_doi)-kept0,"kept_all":len(rows_all)-all0})
                log(L, f"[INFO] [{name}] q='{q}' kept_doi+{len(rows_doi)-kept0} kept_all+{len(rows_all)-all0} collected_seen={collected_seen}")
            ph["requests_end"]=client.request_count
            ph["rows_doi_end"]=len(rows_doi)
            phases.append(ph)

        # 1) SEED
        seed_target = max(120, min(250, int(args.n)))
        run_search("seed", seed_queries, "relevance_score:desc", pages_each=3, target=seed_target)

        # infer field
        dom_id, dom_share, dom_cnt = top_domain(rows_all)
        dom_gate = ""
        if dom_id and dom_share >= 0.55 and len(rows_all) >= 80:
            dom_gate = dom_id
            base_filters.append(f"primary_topic.domain.id:{dom_gate}")

        tgate = top_topic_ids(rows_all, topk=12)
        if tgate:
            base_filters.append("topics.id:" + "|".join(tgate[:12]))

        log(L, f"[INFO] Seed rows_all={len(rows_all)} rows_doi={len(rows_doi)}")
        log(L, f"[INFO] Domain gate={'ON '+dom_gate if dom_gate else 'OFF'} share={dom_share:.2f}")
        if tgate:
            log(L, f"[INFO] Topic gate count={len(tgate)} top={tgate[:6]}")

        # 2) EXPAND
        target_n = int(args.n)
        if len(rows_doi) < target_n and client.request_count < int(args.request_cap):
            run_search("expand_recent", seed_queries[:4], "publication_date:desc", pages_each=3, target=target_n)
        if len(rows_doi) < target_n and client.request_count < int(args.request_cap):
            run_search("expand_foundational", seed_queries[:4], "cited_by_count:desc", pages_each=3, target=target_n)

        # 3) CITATION CHASING
        seeds_sorted = sorted(rows_all, key=lambda r: -int(r.get("cited_by") or 0))
        seed_ids = [r.get("openalex_id") for r in seeds_sorted if r.get("openalex_id")][:8]

        def run_filter_phase(name: str, filt: str):
            ph={"phase":name,"filter":filt,"requests_start":client.request_count,"requests_end":None,
                "rows_doi_start":len(rows_doi),"rows_doi_end":None, "collected_seen":0}
            if client.request_count >= int(args.request_cap):
                phases.append(ph); return
            collected=0
            for w in client.list_works(search=None, filter_parts=base_filters + [filt], sort="cited_by_count:desc",
                                       per_page=200, max_pages=1, select_fields=select_fields, log_fp=L):
                before=len(seen)
                add_work(w, source=name)
                if len(seen) > before:
                    collected += 1
                if len(rows_doi) >= target_n:
                    break
            ph["requests_end"]=client.request_count
            ph["rows_doi_end"]=len(rows_doi)
            ph["collected_seen"]=collected
            phases.append(ph)
            log(L, f"[INFO] [{name}] filter='{filt}' collected_seen={collected} kept_doi={len(rows_doi)}")

        if len(rows_doi) < target_n and seed_ids and client.request_count < int(args.request_cap):
            for wid in seed_ids:
                if client.request_count >= int(args.request_cap) or len(rows_doi) >= target_n:
                    break
                run_filter_phase("cites_incoming", f"cites:{wid}")
                if client.request_count >= int(args.request_cap) or len(rows_doi) >= target_n:
                    break
                run_filter_phase("cited_by_outgoing", f"cited_by:{wid}")

        # 4) SCORING + TRIM
        anchors = build_anchor_set(domain_terms, method_terms)
        for r in rows_all:
            r["_score"] = score_row(r, anchors)
        for r in rows_doi:
            r["_score"] = score_row(r, anchors)

        rows_doi_sorted = sorted(rows_doi, key=lambda r: (int(r.get("_score") or 0), int(r.get("cited_by") or 0), str(r.get("publication_date") or "")), reverse=True)
        rows_all_sorted = sorted(rows_all, key=lambda r: (int(r.get("_score") or 0), int(r.get("cited_by") or 0), str(r.get("publication_date") or "")), reverse=True)

        top100 = rows_doi_sorted[:100]
        hit = sum(1 for r in top100 if int(r.get("_score") or 0) >= 2)
        hit_rate = hit / max(1, len(top100))

        def strip(rs):
            out=[]
            for r in rs:
                rr=dict(r)
                rr.pop("_score", None)
                rr.pop("_topic_ids", None)
                out.append(rr)
            return out

        save_csv(os.path.join(out_dir,"corpus.csv"), strip(rows_doi_sorted[:target_n]))
        save_csv(os.path.join(out_dir,"corpus_all.csv"), strip(rows_all_sorted[:max(target_n, 500)]))

        search_log = {
            "module":"B","version":VERSION,"datetime_utc":utc_now(),
            "target_n":target_n,"kept_doi":len(rows_doi_sorted[:target_n]),"kept_all":len(rows_all_sorted),
            "requests_used":client.request_count,"request_cap":int(args.request_cap),
            "seed_queries":seed_queries,"domain_terms":domain_terms,"method_terms":method_terms,
            "domain_gate":dom_gate,"domain_share_seed":dom_share,"domain_counts_seed":dom_cnt,
            "topic_gate_ids":[f"https://openalex.org/{x}" for x in tgate[:12]] if tgate else [],
            "quality":{"anchor_hit_rate_top100":hit_rate,"top100_hits_ge2":hit,"anchors_count":len(anchors)},
            "phases":phases,
        }
        with open(os.path.join(out_dir,"search_log.json"),"w",encoding="utf-8") as f:
            json.dump(search_log, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir,"field_map.md"),"w",encoding="utf-8") as f:
            f.write("# Field map (Stage B)\n\n")
            f.write(f"- Version: {VERSION}\n")
            f.write(f"- Date (UTC): {utc_now()}\n")
            f.write(f"- Kept DOI: **{len(rows_doi_sorted[:target_n])}** / target {target_n}\n")
            f.write(f"- Requests used: **{client.request_count}** (cap {int(args.request_cap)})\n")
            f.write(f"- Domain gate: `{dom_gate or 'OFF'}` (seed share={dom_share:.2f})\n")
            f.write(f"- Topic gate count: {len(tgate[:12]) if tgate else 0}\n")
            f.write(f"- Anchor hit-rate top100 (score>=2): **{hit_rate:.2f}**\n")

        with open(os.path.join(out_dir,"prisma_lite.md"),"w",encoding="utf-8") as f:
            f.write("# PRISMA-lite (Stage B)\n\n")
            f.write(f"- Source: OpenAlex API\n")
            f.write(f"- Date (UTC): {utc_now()}\n")
            f.write(f"- Included (DOI): {len(rows_doi_sorted[:target_n])}\n")
            f.write(f"- Requests used: {client.request_count}\n")
            f.write(f"- Quality anchor_hit_rate_top100: {hit_rate:.2f}\n")

        ckpt={"version":VERSION,"input_hash":ih,"scope":args.scope,"target_n":target_n,
              "rows_doi":strip(rows_doi_sorted[:target_n]),
              "rows_all":strip(rows_all_sorted[:max(target_n,500)]),
              "search_log":search_log}
        with open(os.path.join(out_dir,"_moduleB_checkpoint.json"),"w",encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)

        if len(rows_doi_sorted) < int(args.min_keep):
            log(L, f"[ERR] Too few DOI works: {len(rows_doi_sorted)} < min_keep={int(args.min_keep)}")
            return 3
        if len(top100) >= 50 and hit_rate < float(args.min_anchor_hit):
            log(L, f"[ERR] Drift/low relevance: anchor_hit_rate_top100={hit_rate:.2f} < {float(args.min_anchor_hit):.2f}")
            return 4

        log(L, "[OK] Stage B done.")
        return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(f"[FATAL] {type(e).__name__}: {e}\n")
        raise SystemExit(1)