"""
FAISS-based dense retrieval index for the SECOND-KNOWLEDGE-BRAIN knowledge base.

Builds and searches a cosine-similarity index over BGE embeddings of research papers.
The index is persisted to disk so retrieval is fast at runtime and does not require
re-encoding the entire knowledge base on every query.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from tools.hf_model_manager import HFModelManager


BRAIN_PATH = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_PATH = DATA_DIR / "knowledge_faiss.index"
MAPPING_PATH = DATA_DIR / "knowledge_faiss_mapping.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KnowledgeIndexManager:
    """
    Manage a FAISS index over the research-paper knowledge base.

    - `build()` reads SECOND-KNOWLEDGE-BRAIN.md, embeds each paper with BGE,
      normalizes vectors, and writes a FAISS InnerProduct index to disk.
    - `search(query, top_k)` loads the index (building it if missing), encodes
      the query, and returns the top-k matching papers with cosine scores.
    """

    def __init__(self, brain_path: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.brain_path = brain_path or BRAIN_PATH
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / INDEX_PATH.name
        self._mapping_path = self.data_dir / MAPPING_PATH.name
        self._hf = HFModelManager.instance()

    # ------------------------------------------------------------------
    # Paper parsing
    # ------------------------------------------------------------------

    def _load_papers(self) -> List[Dict[str, str]]:
        """Parse the Key Research Papers table from SECOND-KNOWLEDGE-BRAIN.md."""
        papers: List[Dict[str, str]] = []
        if not self.brain_path.exists():
            return papers

        content = self.brain_path.read_text(encoding="utf-8")
        in_table = False
        header_passed = False
        for line in content.splitlines():
            if "| # | Title |" in line:
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                header_passed = True
                continue
            if in_table and header_passed and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                # Expected columns: #, Title, Authors, Year, Venue, DOI/Link, Key Finding, Relevance
                if len(parts) >= 7:
                    papers.append({
                        "id": _sha256(parts[1] + parts[5]),
                        "title": parts[1],
                        "authors": parts[2],
                        "year": parts[3],
                        "venue": parts[4],
                        "doi_url": parts[5],
                        "key_finding": parts[6],
                        "relevance": parts[7] if len(parts) > 7 else "",
                    })
            elif in_table and header_passed and not line.startswith("|"):
                break
        return papers

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """
        Build the FAISS index from the current SECOND-KNOWLEDGE-BRAIN.md.
        Returns a status dict with `indexed_count` and `index_path`.
        """
        papers = self._load_papers()
        if not papers:
            return {"indexed_count": 0, "index_path": str(self._index_path), "status": "no_papers"}

        texts = [f"{p['title']} {p['key_finding']}" for p in papers]
        embeddings = self._hf.encode_batch(texts)

        # Normalize vectors so that inner product equals cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        import faiss

        dim = embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base_index)

        ids = np.array([i for i in range(len(papers))], dtype="int64")
        index.add_with_ids(embeddings.astype("float32"), ids)
        faiss.write_index(index, str(self._index_path))

        mapping = {str(i): paper for i, paper in enumerate(papers)}
        self._mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "indexed_count": len(papers),
            "index_path": str(self._index_path),
            "mapping_path": str(self._mapping_path),
            "status": "ok",
        }

    def _load_index(self) -> Optional[Any]:
        """Load the FAISS index from disk if it exists."""
        if not self._index_path.exists():
            return None
        import faiss
        try:
            return faiss.read_index(str(self._index_path))
        except Exception:
            return None

    def _load_mapping(self) -> Dict[str, Dict[str, str]]:
        if not self._mapping_path.exists():
            return {}
        try:
            return json.loads(self._mapping_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant papers for `query`.
        Builds the index on first call if it is not present.
        """
        index = self._load_index()
        if index is None:
            status = self.build()
            if status["indexed_count"] == 0:
                return []
            index = self._load_index()
            if index is None:
                return []

        mapping = self._load_mapping()
        if not mapping:
            return []

        query_embedding = self._hf.encode(query)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            norm = 1.0
        query_embedding = (query_embedding / norm).astype("float32").reshape(1, -1)

        scores, ids = index.search(query_embedding, top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            paper = mapping.get(str(int(idx)))
            if paper is None:
                continue
            results.append({**paper, "score": float(score)})
        return results


def build_knowledge_index(brain_path: Optional[Path] = None, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to rebuild the knowledge FAISS index."""
    return KnowledgeIndexManager(brain_path=brain_path, data_dir=data_dir).build()


if __name__ == "__main__":
    result = build_knowledge_index()
    print(f"Knowledge index build result: {result}")
