"""Leakage-safe retrieval-augmented generation (RAG) for the Heimdall LLM society.

This module gives society agents a real retrieval capability over a corpus of
prior knowledge — historical market regime statistics, lessons from our own past
society runs, and timeless methodology cards. It is built to answer one research
question honestly: *does retrieval over old results and available information
improve the LLM's market-making and/or its interrogability, relative to a
no-RAG control?*

The single most important property is the **temporal cutoff**. Every document
carries a ``market_as_of`` timestamp on the *simulated* market clock. At decision
time ``t`` the retriever only ever returns documents with ``market_as_of <= t``
(timeless methodology cards carry ``market_as_of = None`` and are always allowed,
because they contain no window-specific realised outcomes). The cutoff is supplied
by the runner from the current tick — the LLM cannot widen it — so a bid decision
can never see the realised outcome of the window it is bidding on. Without this,
the capture metric would be faked.

Two retrieval backends are supported, both real:
  * ``dense``  — sentence-transformers bi-encoder (default ``BAAI/bge-small-en-v1.5``),
                 cosine similarity over L2-normalised embeddings. This is the
                 "proper RAG" path.
  * ``tfidf``  — scikit-learn TF-IDF cosine. Deterministic, dependency-light
                 fallback that needs no model download; used if the embedding
                 model is unavailable.

Embeddings default to CPU so retrieval never contends with the GPUs (the LLM
generation runs on vLLM; only the tiny query/corpus embedding happens here, and
it is cheap enough for CPU at this corpus size).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge-* retrieval models expect this instruction on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DOC_KINDS = ("historical_stats", "prior_run_lesson", "methodology")


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class RagDocument:
    """One retrievable knowledge item.

    ``market_as_of`` is the simulated-clock instant after which this document's
    information became available. ``None`` means timeless (methodology). The
    retriever filters on this to guarantee no future leak.
    """

    doc_id: str
    text: str
    source: str
    kind: str
    market_as_of: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RagDocument:
        return cls(
            doc_id=str(payload["doc_id"]),
            text=str(payload["text"]),
            source=str(payload.get("source", "")),
            kind=str(payload.get("kind", "methodology")),
            market_as_of=_parse_dt(payload.get("market_as_of")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "kind": self.kind,
            "market_as_of": self.market_as_of.isoformat() if self.market_as_of else None,
            "metadata": self.metadata,
        }


def load_corpus(path: str | Path) -> list[RagDocument]:
    docs: list[RagDocument] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        docs.append(RagDocument.from_json(json.loads(line)))
    return docs


def _corpus_fingerprint(docs: list[RagDocument]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        h.update(doc.doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(doc.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


class RAGRetriever:
    """Temporal-cutoff retriever over a fixed corpus.

    Build once per run (cheap), then call :meth:`retrieve` per agent/tick with the
    tick timestamp as ``as_of``.
    """

    def __init__(
        self,
        documents: list[RagDocument],
        *,
        backend: str,
        model_name: str,
        device: str,
        doc_matrix: np.ndarray | None = None,
        dense_model: Any | None = None,
        tfidf_vectorizer: Any | None = None,
    ) -> None:
        self.documents = documents
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self._doc_matrix = doc_matrix  # (n_docs, dim) L2-normalised, for dense
        self._dense_model = dense_model
        self._tfidf = tfidf_vectorizer
        self._as_of_list = [d.market_as_of for d in documents]
        self.query_count = 0

    # ----- construction -------------------------------------------------
    @classmethod
    def build(
        cls,
        corpus_path: str | Path,
        *,
        backend: str = "dense",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> RAGRetriever:
        docs = load_corpus(corpus_path)
        if not docs:
            raise ValueError(f"RAG corpus at {corpus_path} is empty")
        return cls.build_from_documents(
            docs, backend=backend, model_name=model_name, device=device,
            cache_dir=cache_dir, batch_size=batch_size,
        )

    @classmethod
    def build_from_documents(
        cls,
        documents: list[RagDocument],
        *,
        backend: str = "dense",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> RAGRetriever:
        """Build a retriever from an in-memory document list (e.g. a per-run corpus)."""
        if not documents:
            raise ValueError("RAG document list is empty")
        if backend == "dense":
            try:
                return cls._build_dense(documents, model_name, device, cache_dir, batch_size)
            except Exception as exc:  # noqa: BLE001 - degrade to a real fallback, never to fakery
                print(f"[rag] dense backend unavailable ({exc}); falling back to tfidf")
                backend = "tfidf"
        if backend == "tfidf":
            return cls._build_tfidf(documents, model_name="tfidf")
        raise ValueError(f"unknown RAG backend: {backend}")

    @classmethod
    def _build_dense(
        cls,
        docs: list[RagDocument],
        model_name: str,
        device: str,
        cache_dir: str | Path | None,
        batch_size: int,
    ) -> RAGRetriever:
        from sentence_transformers import SentenceTransformer

        fp = _corpus_fingerprint(docs)
        cache_file: Path | None = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_model = model_name.replace("/", "__")
            cache_file = cache_dir / f"emb_{safe_model}_{fp}.npy"

        model = SentenceTransformer(model_name, device=device)
        if cache_file is not None and cache_file.exists():
            doc_matrix = np.load(cache_file)
            if doc_matrix.shape[0] != len(docs):
                doc_matrix = None  # corpus changed under a stale cache name; recompute
            else:
                return cls(
                    docs, backend="dense", model_name=model_name, device=device,
                    doc_matrix=doc_matrix, dense_model=model,
                )
        doc_matrix = model.encode(
            [d.text for d in docs],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if cache_file is not None:
            np.save(cache_file, doc_matrix)
        return cls(
            docs, backend="dense", model_name=model_name, device=device,
            doc_matrix=doc_matrix, dense_model=model,
        )

    @classmethod
    def _build_tfidf(cls, docs: list[RagDocument], *, model_name: str) -> RAGRetriever:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform([d.text for d in docs])  # sparse, l2-normalised by default
        doc_matrix = np.asarray(matrix.todense(), dtype=np.float32)
        return cls(
            docs, backend="tfidf", model_name=model_name, device="cpu",
            doc_matrix=doc_matrix, tfidf_vectorizer=vec,
        )

    # ----- query --------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "dense":
            vec = self._dense_model.encode(
                [BGE_QUERY_INSTRUCTION + query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            return vec.astype(np.float32)
        # tfidf
        sparse = self._tfidf.transform([query])
        return np.asarray(sparse.todense(), dtype=np.float32)[0]

    def _allowed_indices(self, as_of: datetime | None, kinds: tuple[str, ...] | None) -> list[int]:
        idxs: list[int] = []
        for i, doc in enumerate(self.documents):
            if kinds is not None and doc.kind not in kinds:
                continue
            doc_as_of = self._as_of_list[i]
            if doc_as_of is None:
                idxs.append(i)  # timeless methodology
                continue
            if as_of is None or doc_as_of <= as_of:
                idxs.append(i)
        return idxs

    def retrieve(
        self,
        query: str,
        *,
        as_of: datetime | None,
        k: int = 4,
        max_chars: int = 700,
        kinds: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``k`` documents relevant to ``query`` that were available
        at ``as_of`` (simulated-clock cutoff). Documents dated after ``as_of`` are
        never returned.
        """
        self.query_count += 1
        as_of = _parse_dt(as_of)
        candidates = self._allowed_indices(as_of, kinds)
        if not candidates:
            return []
        qvec = self._embed_query(query)
        sub = self._doc_matrix[candidates]  # (m, dim)
        scores = sub @ qvec  # cosine (both L2-normalised)
        order = np.argsort(-scores)[: max(1, k)]
        results: list[dict[str, Any]] = []
        for rank, local_idx in enumerate(order, start=1):
            gi = candidates[int(local_idx)]
            doc = self.documents[gi]
            text = doc.text if len(doc.text) <= max_chars else doc.text[:max_chars].rstrip() + " …"
            results.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(local_idx)]), 4),
                    "kind": doc.kind,
                    "source": doc.source,
                    "as_of": doc.market_as_of.isoformat() if doc.market_as_of else None,
                    "text": text,
                }
            )
        return results

    def stats(self) -> dict[str, Any]:
        kind_counts: dict[str, int] = {}
        for d in self.documents:
            kind_counts[d.kind] = kind_counts.get(d.kind, 0) + 1
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "device": self.device,
            "document_count": len(self.documents),
            "kind_counts": kind_counts,
            "query_count": self.query_count,
        }
