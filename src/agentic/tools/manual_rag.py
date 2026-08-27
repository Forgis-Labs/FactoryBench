"""manual_rag - retrieval over vendor manuals.

Loads a FAISS index built by ``src.agentic.build_manual_index`` from
``data/manuals/index.faiss`` + ``data/manuals/chunks.jsonl``. If the
index does not exist, the tool returns a stub telling the agent no
manuals are available (which is a valid signal - the agent then falls
back to its own priors).

The index is per-machine at the chunk level (each chunk carries a
``machine`` tag), so an optional ``machine`` filter narrows retrieval.
Embeddings use the same OpenAI SDK that the runner uses for chat, so no
new provider needs to be configured.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class ManualRAGTool:
    NAME = "retrieve_manual"

    def __init__(
        self,
        index_root: Optional[Path] = None,
        embed_model: str = "text-embedding-3-large",
        embed_dim: int = 3072,
    ):
        self.index_root = Path(index_root or "data/manuals")
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._loaded_err: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if faiss is None:
            self._loaded_err = "faiss not installed; run pip install faiss-cpu"
            return
        idx_path = self.index_root / "index.faiss"
        chunks_path = self.index_root / "chunks.jsonl"
        if not idx_path.exists() or not chunks_path.exists():
            self._loaded_err = (
                f"no manual index at {self.index_root}. Run "
                f"`python -m src.agentic.build_manual_index --pdf-dir <dir>` first."
            )
            return
        try:
            self._index = faiss.read_index(str(idx_path))
            with open(chunks_path, encoding="utf-8") as fh:
                self._chunks = [json.loads(line) for line in fh if line.strip()]
        except Exception as exc:
            self._loaded_err = f"index load failed: {type(exc).__name__}: {exc}"

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": (
                    "Retrieve the top-k most relevant chunks from indexed "
                    "vendor manuals (UR runtime error handbook, KUKA reference, "
                    "voraus-AD supplement, and any generic industrial-controls "
                    "documentation). Use for L4 troubleshooting/optimisation or "
                    "any level where a machine-specific concept (channel name, "
                    "error code, physical unit) needs a definition."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "free-text search query"},
                        "machine": {
                            "type": "string",
                            "description": "optional filter (e.g. 'ur3', 'kuka_kr10', 'yu_cobot'). Omit to search all manuals.",
                        },
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(self, query: str, machine: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
        if self._loaded_err:
            return {"error": self._loaded_err, "hits": []}
        if not self._index or not self._chunks:
            return {"error": "index empty", "hits": []}
        # Embed the query with the same OpenAI SDK used for chat, so we
        # inherit the runner's auth path.
        try:
            vec = _embed_one(query, model=self.embed_model)
        except Exception as exc:
            return {"error": f"embedding failed: {type(exc).__name__}: {exc}", "hits": []}
        v = np.asarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(v)
        # Over-fetch, then post-filter by machine so we still return k hits
        # when a machine filter is passed.
        fetch = k * 8 if machine else k
        D, I = self._index.search(v, min(fetch, len(self._chunks)))
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if machine and chunk.get("machine") != machine:
                continue
            hits.append({
                "score":   float(score),
                "machine": chunk.get("machine"),
                "source":  chunk.get("source"),
                "section": chunk.get("section"),
                "text":    chunk["text"],
            })
            if len(hits) >= k:
                break
        return {"query": query, "machine": machine, "hits": hits}


def _embed_one(text: str, model: str) -> List[float]:
    """Fetch one embedding via OpenAI direct.

    Uses the OpenAI direct API (openai.com) with OPENAI_API_KEY, matching
    the agentic driver route (some managed endpoints do not expose an
    embeddings model, so we use the direct API here).
    """
    # Late import + late load so this works whether called from a runner
    # (which loads .env early) or standalone (which may not).
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set - cannot embed. Add it to .env or export it."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding
