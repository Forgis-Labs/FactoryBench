"""Build the FAISS index of vendor-manual chunks used by ``manual_rag``.

Reads PDFs from ``--pdf-dir`` (or a directory listed in ``--pdf-dir`` per
machine), splits into overlapping chunks, embeds with
``text-embedding-3-large``, and writes:

  * ``data/manuals/index.faiss``   - inner-product index (L2-normalised)
  * ``data/manuals/chunks.jsonl``  - one JSON per chunk with
                                     {machine, source, section, text}

Usage::

    python -m src.agentic.build_manual_index \\
        --pdf-dir data/manuals/pdfs \\
        --out-dir data/manuals

If ``--pdf-dir`` contains subdirectories named after machines
(``ur3/``, ``kuka_kr10/``, ``yu_cobot/``, ``generic/``), the subdir name
becomes the chunk's ``machine`` tag. PDFs at the top level are tagged
``generic``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf missing - pip install pypdf", file=sys.stderr); raise
try:
    import faiss
except ImportError:
    print("faiss missing - pip install faiss-cpu", file=sys.stderr); raise


def _iter_pdfs(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (machine_tag, pdf_path) pairs. Subdir name becomes the tag."""
    for p in sorted(root.rglob("*.pdf")):
        rel = p.relative_to(root)
        tag = rel.parts[0] if len(rel.parts) > 1 else "generic"
        yield tag, p


def _pdf_to_pages(pdf: Path) -> List[str]:
    r = PdfReader(str(pdf))
    return [p.extract_text() or "" for p in r.pages]


def _split_pages(pages: List[str], target_chars: int = 1500, overlap: int = 200) -> Iterable[Dict[str, Any]]:
    """Chunk by page, splitting long pages into ~1500-char windows."""
    for i, page in enumerate(pages):
        text = re.sub(r"\s+", " ", page).strip()
        if not text:
            continue
        if len(text) <= target_chars:
            yield {"section": f"p.{i+1}", "text": text}
            continue
        step = target_chars - overlap
        for j, start in enumerate(range(0, len(text), step)):
            chunk = text[start:start + target_chars]
            if len(chunk) < 200:
                continue
            yield {"section": f"p.{i+1}#{j+1}", "text": chunk}


def _embed_batch(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    from src.agentic.tools.manual_rag import _embed_one
    return [_embed_one(t, model=model) for t in texts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, required=True,
                    help="directory containing PDFs; optional per-machine subdirs")
    ap.add_argument("--out-dir", type=Path, default=Path("data/manuals"))
    ap.add_argument("--embed-model", default="text-embedding-3-large")
    ap.add_argument("--dim", type=int, default=3072,
                    help="embedding dimension (3072 for text-embedding-3-large)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.pdf_dir.exists():
        raise SystemExit(f"pdf-dir does not exist: {args.pdf_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    for tag, pdf in _iter_pdfs(args.pdf_dir):
        pages = _pdf_to_pages(pdf)
        for chunk in _split_pages(pages):
            chunk["machine"] = tag
            chunk["source"] = pdf.name
            all_chunks.append(chunk)
    print(f"extracted {len(all_chunks)} chunks from {args.pdf_dir}")

    if not all_chunks:
        raise SystemExit("no chunks extracted; check --pdf-dir has readable PDFs")

    # embed in batches to keep memory reasonable
    vectors = np.empty((len(all_chunks), args.dim), dtype="float32")
    t0 = time.time()
    for i in range(0, len(all_chunks), args.batch_size):
        batch = [c["text"] for c in all_chunks[i:i + args.batch_size]]
        embs = _embed_batch(batch, model=args.embed_model)
        vectors[i:i + len(embs)] = np.asarray(embs, dtype="float32")
        print(f"  embedded {min(i + args.batch_size, len(all_chunks))}/{len(all_chunks)}"
              f"  ({time.time() - t0:.1f}s)")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(args.dim)
    index.add(vectors)
    faiss.write_index(index, str(args.out_dir / "index.faiss"))
    with open(args.out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_dir}/index.faiss + chunks.jsonl "
          f"({len(all_chunks)} chunks, {args.dim}-dim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
