# dj_inbox/rag/embed.py
"""Step 2 of RAG: turn text into vectors with Gemini Embedding 2.

Documents and questions are embedded differently on purpose (task_type): RETRIEVAL_DOCUMENT for chunks,
RETRIEVAL_QUERY for questions. A question ("What did we quote the Brennans?") and the email that answers it
("Premier package: $2,395") look nothing alike as text; the model is trained to put them close together anyway.

On Google Cloud, Gemini Embedding 2 takes ONE text per request: a list of texts is merged into a single input
and comes back as a single vector (found Nov 5, 2026, when 100 emails returned 1 vector). So each text gets its
own request, run 8 at a time in parallel, with retries on rate limits.

Vectors are saved to disk keyed by a hash of their text, so re-running costs nothing unless the chunks change.
Embedding the whole synthetic inbox costs about 2 cents (Gemini Embedding 2, ~$0.20 / 1M tokens, checked
Nov 5, 2026).
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MODEL = "gemini-embedding-2"
DIM = 768          # Gemini Embedding 2 allows smaller vectors (Matryoshka); 768 keeps files and pgvector small
WORKERS = 8        # requests in flight at once
SAVE_EVERY = 100   # write the cache to disk this often, so a crash keeps the work done so far


def _key(text: str, task: str) -> str:
    return hashlib.sha256(f"{MODEL}|{DIM}|{task}|{text}".encode()).hexdigest()


class Embedder:
    def __init__(self, client, cache_dir: Path | None = None, sleep=time.sleep):
        self.client = client
        self.cache_dir = cache_dir
        self.sleep = sleep
        self.cache: dict[str, list[float]] = {}
        self.calls = 0                                   # API calls made (for the cost/latency log)
        if cache_dir and (cache_dir / "vectors.json").exists():
            self.cache = json.loads((cache_dir / "vectors.json").read_text())

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """Vectors for `texts` (one row each), using the disk cache where possible."""
        missing = [t for t in dict.fromkeys(texts) if _key(t, task) not in self.cache]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            vectors = pool.map(lambda t: self._one(t, task), missing)        # results come back in order
            for n, (text, vec) in enumerate(zip(missing, vectors, strict=True), 1):
                self.cache[_key(text, task)] = vec
                if n % SAVE_EVERY == 0:
                    self._save()
                    print(f"   embedded {n}/{len(missing)}")
        if missing:
            self._save()
        return np.array([self.cache[_key(t, task)] for t in texts], dtype=np.float32)

    def _one(self, text: str, task: str, tries: int = 6) -> list[float]:
        """Embed one text; wait and retry on rate limits (HTTP 429)."""
        from google.genai import types
        for attempt in range(1, tries + 1):
            try:
                self.calls += 1
                resp = self.client.models.embed_content(
                    model=MODEL, contents=text,
                    config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIM))
                if len(resp.embeddings) != 1:              # the check that caught the merged-batch bug
                    raise ValueError(f"expected 1 vector, got {len(resp.embeddings)}")
                return resp.embeddings[0].values
            except Exception as e:
                if getattr(e, "code", None) != 429 or attempt == tries:
                    raise
                self.sleep(5 * attempt)
        raise AssertionError("unreachable")

    def _save(self) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "vectors.json").write_text(json.dumps(self.cache))
