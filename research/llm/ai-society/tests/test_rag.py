"""Tests for the leak-safe RAG retriever (heimdall_ai_society.rag).

The critical property is the temporal cutoff: a query at time ``t`` must never
return a document dated after ``t``. These use the TF-IDF backend so they need no
model download and run in CI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from heimdall_ai_society.rag import RAGRetriever, RagDocument


def _write_corpus(tmp_path: Path) -> Path:
    docs = [
        RagDocument(
            doc_id="past-mar",
            text="DK1 up-heavy activation regime, high settlement price, bid up side",
            source="t", kind="historical_stats",
            market_as_of=datetime(2026, 3, 15, 23, 59, 59, tzinfo=UTC),
        ),
        RagDocument(
            doc_id="future-apr10",
            text="DK1 up-heavy activation regime settlement price up side",
            source="t", kind="historical_stats",
            market_as_of=datetime(2026, 4, 10, 23, 59, 59, tzinfo=UTC),
        ),
        RagDocument(
            doc_id="timeless-method",
            text="The verifier rejects bids whose worst-case profit falls below tau",
            source="t", kind="methodology", market_as_of=None,
        ),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(d.to_json()) for d in docs) + "\n", encoding="utf-8")
    return path


def test_cutoff_excludes_future_documents(tmp_path: Path) -> None:
    r = RAGRetriever.build(_write_corpus(tmp_path), backend="tfidf")
    cutoff = datetime(2026, 4, 2, 5, 30, tzinfo=UTC)
    res = r.retrieve("up-heavy activation settlement price", as_of=cutoff, k=10)
    returned = {d["source"] or d["text"] for d in res}  # source is "t"; use doc identity via text
    ids = {d["text"][:20] for d in res}
    # the future (apr10) doc must never appear
    assert all(d["as_of"] != "2026-04-10T23:59:59+00:00" for d in res), res
    # the past (mar15) doc and the timeless methodology card are allowed
    kinds = {d["kind"] for d in res}
    assert "historical_stats" in kinds
    assert "methodology" in kinds
    assert ids  # non-empty


def test_timeless_methodology_always_allowed(tmp_path: Path) -> None:
    r = RAGRetriever.build(_write_corpus(tmp_path), backend="tfidf")
    # cutoff before every dated doc — only the timeless card may return
    cutoff = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    res = r.retrieve("verifier worst-case profit tau", as_of=cutoff, k=10)
    assert res, "timeless methodology should still be retrievable before any dated doc"
    assert all(d["as_of"] is None for d in res)


def test_no_cutoff_returns_everything(tmp_path: Path) -> None:
    r = RAGRetriever.build(_write_corpus(tmp_path), backend="tfidf")
    res = r.retrieve("activation regime", as_of=None, k=10)
    # as_of=None disables the dated filter (used only for offline corpus inspection)
    assert len(res) == 3


def test_kinds_filter(tmp_path: Path) -> None:
    r = RAGRetriever.build(_write_corpus(tmp_path), backend="tfidf")
    cutoff = datetime(2026, 4, 2, 5, 30, tzinfo=UTC)
    res = r.retrieve("regime", as_of=cutoff, k=10, kinds=("methodology",))
    assert res and all(d["kind"] == "methodology" for d in res)


def test_empty_corpus_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        RAGRetriever.build(empty, backend="tfidf")
