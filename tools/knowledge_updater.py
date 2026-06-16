"""
Research paper crawler for the Dog Behavior Agent.
Sources: ArXiv cs.SD + cs.CV + cs.LG, Semantic Scholar.
Updates SECOND-KNOWLEDGE-BRAIN.md weekly.
"""

import os
import hashlib
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import logging

BRAIN_PATH = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"
DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"

ARXIV_CATEGORIES = ["cs.SD", "cs.CV", "cs.LG"]
ARXIV_QUERIES = [
    "dog vocalization classification",
    "canine behavior audio recognition",
    "animal sound classification deep learning",
    "bioacoustics machine learning",
    "MFCC animal vocalization",
]
S2_QUERIES = [
    "dog bark classification neural network",
    "canine acoustic behavior analysis",
    "animal vocalization intent recognition",
    "wav2vec audio animal sounds",
    "dog behavior machine learning",
]
DOMAIN_KEYWORDS = [
    "dog", "canine", "bark", "vocalization", "animal sound",
    "bioacoustics", "behavior classification", "MFCC", "wav2vec",
    "audio classification", "ethology", "dog behavior", "animal audio",
]
RECENCY_WINDOW_DAYS = 90
TOP_N_PER_RUN = 10

logger = logging.getLogger(__name__)


@dataclass
class PaperEntry:
    title: str
    authors: str
    year: str
    venue: str
    doi_url: str
    abstract: str
    key_finding: str
    relevance: str
    score: float = 0.0


def _sha256(title: str, doi: str = "") -> str:
    return hashlib.sha256((title + doi).encode()).hexdigest()


class KnowledgeUpdater:
    def __init__(self):
        self._memory = None

    def _get_memory(self):
        if self._memory is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager()
        return self._memory

    def run(self) -> Dict[str, int]:
        """Run a full knowledge update cycle. Returns {"added": N, "skipped": M}."""
        all_papers: List[PaperEntry] = []

        for query in ARXIV_QUERIES:
            try:
                papers = self._crawl_arxiv(query)
                all_papers.extend(papers)
            except Exception as exc:
                logger.warning("ArXiv crawl failed for query %r: %s", query, exc)

        for query in S2_QUERIES:
            try:
                papers = self._crawl_semantic_scholar(query)
                all_papers.extend(papers)
            except Exception as exc:
                logger.warning("Semantic Scholar crawl failed for query %r: %s", query, exc)

        unique = self._deduplicate(all_papers)
        scored = self._score_papers(unique)
        top = sorted(scored, key=lambda p: p.score, reverse=True)[:TOP_N_PER_RUN]

        added = 0
        skipped = 0
        for paper in top:
            mem = self._get_memory()
            if mem.is_known_paper(paper.title, paper.doi_url):
                skipped += 1
                continue
            self._append_to_brain(paper)
            mem.mark_paper_known(paper.title, paper.doi_url)
            added += 1

        if added > 0:
            self._log_update(added)

        return {"added": added, "skipped": skipped, "total_candidates": len(all_papers)}

    def _crawl_arxiv(self, query: str, max_results: int = 20) -> List[PaperEntry]:
        """Fetch papers from ArXiv API."""
        base = "http://export.arxiv.org/api/query"
        cat_filter = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
        full_query = f"({query}) AND ({cat_filter})"
        params = urllib.parse.urlencode({
            "search_query": full_query,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"{base}?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)
            authors_els = entry.findall("atom:author/atom:name", ns)

            if title_el is None or id_el is None:
                continue

            title = (title_el.text or "").strip().replace("\n", " ")
            abstract = (summary_el.text or "").strip().replace("\n", " ")[:500] if summary_el is not None else ""
            arxiv_id = (id_el.text or "").strip()
            year = (published_el.text or "")[:4] if published_el is not None else ""
            authors = ", ".join(a.text or "" for a in authors_els[:3])
            if len(authors_els) > 3:
                authors += " et al."

            papers.append(PaperEntry(
                title=title,
                authors=authors,
                year=year,
                venue="ArXiv",
                doi_url=arxiv_id,
                abstract=abstract,
                key_finding=abstract[:200] if abstract else title,
                relevance="Crawled from ArXiv",
            ))
        return papers

    def _crawl_semantic_scholar(self, query: str, limit: int = 10) -> List[PaperEntry]:
        """Fetch papers from Semantic Scholar Graph API."""
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={urllib.parse.quote(query)}&limit={limit}"
            f"&fields=title,authors,year,venue,externalIds,abstract"
        )
        headers = {"User-Agent": "dog-behavior-agent/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        papers = []
        for item in data.get("data", []):
            title = item.get("title", "")
            if not title:
                continue
            abstract = (item.get("abstract") or "")[:500]
            year = str(item.get("year", ""))
            venue = item.get("venue", "Semantic Scholar")
            authors_list = item.get("authors", [])
            authors = ", ".join(a.get("name", "") for a in authors_list[:3])
            if len(authors_list) > 3:
                authors += " et al."
            ext_ids = item.get("externalIds", {})
            doi = ext_ids.get("DOI", "") or ext_ids.get("ArXiv", "")

            papers.append(PaperEntry(
                title=title,
                authors=authors,
                year=year,
                venue=venue or "Semantic Scholar",
                doi_url=f"https://doi.org/{doi}" if doi and "/" in doi else doi,
                abstract=abstract,
                key_finding=abstract[:200] if abstract else title,
                relevance="Crawled from Semantic Scholar",
            ))
        return papers

    def _deduplicate(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        seen = set()
        unique = []
        for p in papers:
            h = _sha256(p.title, p.doi_url)
            if h not in seen:
                seen.add(h)
                unique.append(p)
        return unique

    def _score_papers(self, papers: List[PaperEntry]) -> List[PaperEntry]:
        now = datetime.utcnow()
        for paper in papers:
            # Recency score (0–1)
            try:
                year = int(paper.year) if paper.year else 2000
                delta_days = (now - datetime(year, 1, 1)).days
                recency = max(0.0, 1.0 - delta_days / RECENCY_WINDOW_DAYS)
            except Exception:
                recency = 0.0

            # Relevance score: keyword match in title + abstract
            text = (paper.title + " " + paper.abstract).lower()
            keyword_hits = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in text)
            relevance = min(1.0, keyword_hits / max(1, len(DOMAIN_KEYWORDS) // 2))

            paper.score = 0.6 * recency + 0.4 * relevance
        return papers

    def _append_to_brain(self, paper: PaperEntry):
        """Append a new paper row to the Key Research Papers table in SECOND-KNOWLEDGE-BRAIN.md."""
        if not BRAIN_PATH.exists():
            return
        content = BRAIN_PATH.read_text(encoding="utf-8")

        # Find the table end (first blank line after the table header)
        table_marker = "| # | Title |"
        if table_marker not in content:
            return

        # Count existing rows to get next row number
        lines = content.splitlines()
        row_num = sum(1 for l in lines if l.startswith("| ") and not l.startswith("| # |") and not l.startswith("|---"))
        next_num = row_num + 1

        key_finding = (paper.abstract or paper.title)[:200].replace("|", "-")
        new_row = (
            f"| {next_num} | {paper.title[:80]} | {paper.authors[:40]} | {paper.year} "
            f"| {paper.venue[:30]} | {paper.doi_url[:60]} | {key_finding} | Auto-crawled |\n"
        )

        # Insert before the blank line that ends the table
        insert_pos = content.find("\n\n", content.find(table_marker))
        if insert_pos == -1:
            content += "\n" + new_row
        else:
            content = content[:insert_pos] + "\n" + new_row.rstrip() + content[insert_pos:]

        BRAIN_PATH.write_text(content, encoding="utf-8")

    def _log_update(self, added: int):
        """Append an entry to the Knowledge Update Log."""
        if not BRAIN_PATH.exists():
            return
        content = BRAIN_PATH.read_text(encoding="utf-8")
        log_marker = "## Knowledge Update Log"
        if log_marker not in content:
            return
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        new_entry = f"| {date_str} | {added} | ArXiv, Semantic Scholar | Automated weekly crawl |\n"
        insert_pos = content.find("\n", content.find(log_marker) + len(log_marker))
        # Find the table header line
        table_start = content.find("| Date |", insert_pos)
        sep_line = content.find("|\n", table_start)
        if sep_line != -1:
            content = content[:sep_line + 2] + new_entry + content[sep_line + 2:]
        else:
            content += "\n" + new_entry
        BRAIN_PATH.write_text(content, encoding="utf-8")

    def start_scheduled(self):
        """Start weekly APScheduler job."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            self.run,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="dog_weekly_knowledge_update",
            replace_existing=True,
        )
        scheduler.start()
        return scheduler


if __name__ == "__main__":
    updater = KnowledgeUpdater()
    result = updater.run()
    print(f"Knowledge update complete: {result}")
