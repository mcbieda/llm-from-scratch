#!/usr/bin/env python3
"""
get_pubmed_abstracts_v7.py
Pulls any-size PubMed result set despite the 10 000-record limit
by automatically slicing the date range and chunking the fetches.

IMPORTANT: there will be a prompt for an API key. Make sure to have an NCBI account and an API key
Note that search params etc are in main block
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, Generator, List, Sequence, Tuple

from Bio import Entrez, Medline

# ──────────────────────────────────────────────────────────────────────
#  0.  PARAMETER BLOCK - ADJUST THESE FOR YOUR QUERY AND LIMITS
# ──────────────────────────────────────────────────────────────────────
# BASE is the prompt for getting abstracts from pubmed  
BASE = "(cancer[Title/Abstract]) AND english[lang] AND" \
           "(ERBB2[Title/Abstract] OR HER2[Title/Abstract] OR EGFR[Title/Abstract])"
#  set year range and the filename root for the resulting files
year_start, year_end= 2024,2025
thisfileroot="pubmed_abstracts_2024to2025ONLY_ERBB2_ABSTRACTS_getv7_english"
# Some run params
DEFAULT_ENTREZ_EMAIL = "" # fill this in  to make easier for getting records
API_DELAY_NO_KEY  = 0.34                       # 3 req/s
API_DELAY_WITH_KEY = 0.12                      # 10 req/s


# ──────────────────────────────────────────────────────────────────────
#  1.  Runtime API-key prompt
# ──────────────────────────────────────────────────────────────────────
def prompt_api_key() -> Tuple[str, float]:
    raw_email = input(f"Enter your NCBI email (blank for default: {DEFAULT_ENTREZ_EMAIL}): ").strip()
    email = raw_email or DEFAULT_ENTREZ_EMAIL
    Entrez.email = email
    print(f"[INFO] Using email: {Entrez.email}")

    raw = input("Enter your NCBI API key (blank for none): ").strip()
    key = raw.strip("\"'").strip()
    if key:
        Entrez.api_key = key
        print("[INFO] Using supplied API key (10 req/s limit).")
        return key, API_DELAY_WITH_KEY
    else:
        print("[INFO] No API key; using 3 req/s limit.")
        return "", API_DELAY_NO_KEY


# ──────────────────────────────────────────────────────────────────────
# 2.  Helpers to beat the 10 000-record ESearch/EFetch wall
# ──────────────────────────────────────────────────────────────────────
def esearch_count(term: str) -> int:
    """Return hit count for *term* (no PMIDs)."""
    h = Entrez.esearch(db="pubmed", term=term, rettype="count")
    return int(Entrez.read(h)["Count"])


def slice_by_year(base_query: str,
                  start_year: int,
                  end_year: int) -> Generator[Tuple[str, str], None, None]:
    """
    Yield date windows (mindate,maxdate) such that each window
    returns ≤ 10 000 records. Recurse to half-years → months if needed.
    """
    def _need_smaller(q: str, mind: str, maxd: str) -> bool:
        slice_term = f"({q}) AND {mind}:{maxd}[dp]"
        return esearch_count(slice_term) > 9_999

    for year in range(start_year, end_year + 1):
        y_start, y_end = f"{year}/01/01", f"{year}/12/31"
        if not _need_smaller(base_query, y_start, y_end):
            yield y_start, y_end
            continue

        # try half-years
        for half in [(1, 6), (7, 12)]:
            h_start = f"{year}/{half[0]:02d}/01"
            h_end   = f"{year}/{half[1]:02d}/{31 if half[1]==12 else 30:02d}"
            if not _need_smaller(base_query, h_start, h_end):
                yield h_start, h_end
                continue

            # fall back to month-level slices
            for month in range(1, 13):
                m_start = f"{year}/{month:02d}/01"
                # crude month-end (28-31 is fine for E-utils)
                m_end   = f"{year}/{month:02d}/31"
                yield m_start, m_end


def esearch_pmids(term: str, retmax: int = 10_000) -> List[str]:
    """
    Return ALL PMIDs matching *term* (guaranteed ≤ 10 000 by caller).
    """
    handle = Entrez.esearch(db="pubmed",
                            term=term,
                            retmax=retmax,
                            usehistory="n")
    rec = Entrez.read(handle)
    return rec["IdList"]


def efetch_chunked(pmids: Sequence[str],
                   delay: float,
                   chunk: int = 1000) -> Generator[Dict[str, str], None, None]:
    """
    Yield MEDLINE records for *pmids* in chunks of ≤ *chunk* IDs.
    """
    for i in range(0, len(pmids), chunk):
        sublist = pmids[i:i + chunk]
        handle = Entrez.efetch(db="pubmed",
                               id=",".join(sublist),
                               rettype="medline",
                               retmode="text")
        for record in Medline.parse(handle):
            yield record
        handle.close()
        time.sleep(delay)


# ──────────────────────────────────────────────────────────────────────
# 3.  Convenience I/O
# ──────────────────────────────────────────────────────────────────────
def save_articles(arts: List[Dict[str, str]],
                  root: str) -> None:
    csv_path = f"{root}_FULL.csv"
    txt_path = f"{root}_ABSTRACTS.txt"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh,
                           fieldnames=["RECNUM","PMID","Year", "Title", "Authors",
                                       "Journal", "Abstract"])
        w.writeheader()
        w.writerows(arts)

    with open(txt_path, "w", encoding="utf-8") as fh:
        for a in arts:
            fh.write(a["Abstract"].replace("\n", " ").strip() + "\n")

    print(f"[DONE] {len(arts):,} records saved → {csv_path}, {txt_path}")


def find_project_root(start: Path) -> Path:
    """
    Resolve repo root by walking upward until both `src/` and `notebooks/` exist.
    """
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "notebooks").exists():
            return p
    return start


# ──────────────────────────────────────────────────────────────────────
# 4.  Main high-level driver
# ──────────────────────────────────────────────────────────────────────
def download_formatted_pubmed_abstracts(base_query: str,
                                        year_start: int,
                                        year_end: int,
                                        fileroot: str = "pubmed",
                                        id_chunk: int = 1000) -> None:
    project_root = find_project_root(Path(__file__).resolve())
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fileroot_path = Path(fileroot)
    if fileroot_path.parent == Path("."):
        fileroot = str(data_dir / fileroot_path.name)

    key, delay = prompt_api_key()

    all_articles: List[Dict[str, str]] = []
    articlenum = 0
    for (mindate, maxdate) in slice_by_year(base_query,
                                            year_start,
                                            year_end):
        slice_term = f"({base_query}) AND {mindate}:{maxdate}[dp]"
        pmids = esearch_pmids(slice_term)
        print(f"[INFO] {mindate}–{maxdate}: {len(pmids):,} PMIDs")

        for rec in efetch_chunked(pmids, delay, chunk=id_chunk):
            art = {
                "RECNUM": articlenum, # added in v7
                "PMID": rec.get("PMID", ""),
                "Year": rec.get("DP","").split(" ")[0], # added year in v6
                "Title": rec.get("TI", ""),
                "Authors": "; ".join(rec.get("AU", [])),
                "Journal": rec.get("JT", ""),
                "Abstract": rec.get("AB", ""),
            }
            articlenum += 1
            all_articles.append(art)

    save_articles(all_articles, fileroot)


# ──────────────────────────────────────────────────────────────────────
# 5.  Example usage
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PARAMS HERE
    
    # main function here
    download_formatted_pubmed_abstracts(BASE,
                                        year_start=year_start,
                                        year_end=year_end,
                                        fileroot=thisfileroot)
