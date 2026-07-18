"""Pruebas offline de recuperación semántica."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from bimbam_assistant.application import rag_service
from bimbam_assistant.application.rag_service import (
    RetrievalError,
    build_context,
    build_retrieved_chunks,
    normalize_query,
    retrieve_documents,
)
from bimbam_assistant.domain.models import RetrievedChunk
from bimbam_assistant.infrastructure.faiss_store import (
    FaissStoreError,
    SearchResult,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    GeminiEmbeddingError,
)


def build_search_result(
    *,
    vector_id: int = 0,
    score: float = 0.88,
    document_name: str = "Manual de Garantía",
    page_number: int = 3,
    category: str = "garantias",
) -> SearchResult:
    return SearchResult(
        vector_id=vector_id,
        score=score,
        document=Document(
            page_content=(
                "La garantía cubre fallas de fabricación."
            ),
            metadata={
                "document_name": document_name,
                "page_number": page_number,
                "category": category,
                "chunk_id": (
                    f"manual-page-{page_number}-chunk-0"
                ),
            },
        ),
    )


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query(
        "  ¿Cuánto   tarda\n el reembolso?  "
    ) == "¿Cuánto tarda el reembolso?"


@pytest.mark.parametrize("query", ["", "   "])
def test_normalize_query_rejects_empty_text(
    query: str,
) -> None:
    with pytest.raises(
        RetrievalError,
        match="no puede estar vacía",
    ):
        normalize_query(query)


def test_normalize_query_rejects_non_string() -> None:
    with pytest.raises(
        RetrievalError,
        match="cadena de texto",
    ):
        normalize_query(
            123  # type: ignore[arg-type]
        )


def test_build_retrieved_chunks_assigns_rank_and_metadata() -> None:
    chunks = build_retrieved_chunks(
        [
            build_search_result(
                vector_id=7,
                score=0.91,
            ),
            build_search_result(
                vector_id=3,
                score=0.82,
                page_number=4,
            ),
        ]
    )

    assert [chunk.rank for chunk in chunks] == [1, 2]
    assert chunks[0].vector_id == 7
    assert chunks[0].score == pytest.approx(0.91)
    assert chunks[1].metadata["page_number"] == 4


def test_build_context_includes_sources_and_traceability() -> None:
    context = build_context(
        [
            RetrievedChunk(
                rank=1,
                vector_id=0,
                score=0.8765,
                page_content="Contenido respaldado.",
                metadata={
                    "document_name": "Política de Reembolsos",
                    "page_number": 4,
                    "category": "reembolsos_devoluciones",
                    "chunk_id": "refund-page-4-chunk-0",
                },
            )
        ]
    )

    assert "[Fuente 1]" in context
    assert "Documento: Política de Reembolsos" in context
    assert "Página: 4" in context
    assert "Categoría: reembolsos_devoluciones" in context
    assert "Chunk: refund-page-4-chunk-0" in context
    assert "Similitud: 0.8765" in context
    assert "Contenido respaldado." in context


def test_retrieve_documents_uses_configuration_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_k=4,
            retrieval_score_threshold=0.30,
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "load_vector_store",
        lambda: fake_store,
    )

    def fake_embed_query(query: str) -> list[float]:
        captured["embedded_query"] = query
        return [1.0, 0.0]

    def fake_search_by_vector(
        store: object,
        query_vector: list[float],
        *,
        k: int,
        score_threshold: float,
        filters: dict[str, object],
    ) -> list[SearchResult]:
        captured["store"] = store
        captured["query_vector"] = query_vector
        captured["k"] = k
        captured["score_threshold"] = score_threshold
        captured["filters"] = filters

        return [build_search_result()]

    monkeypatch.setattr(
        rag_service,
        "embed_query",
        fake_embed_query,
    )

    monkeypatch.setattr(
        rag_service,
        "search_by_vector",
        fake_search_by_vector,
    )

    response = retrieve_documents(
        "  ¿Qué cubre   la garantía?  ",
        filters={
            "category": "garantias"
        },
    )

    assert response.query == "¿Qué cubre la garantía?"
    assert response.has_results is True
    assert len(response.results) == 1
    assert "[Fuente 1]" in response.context
    assert response.filters == {
        "category": "garantias"
    }

    assert captured["embedded_query"] == "¿Qué cubre la garantía?"
    assert captured["store"] is fake_store
    assert captured["query_vector"] == [1.0, 0.0]
    assert captured["k"] == 4
    assert captured["score_threshold"] == 0.30
    assert captured["filters"] == {
        "category": "garantias"
    }


def test_retrieve_documents_allows_explicit_k_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_k=4,
            retrieval_score_threshold=0.30,
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "load_vector_store",
        lambda: object(),
    )

    monkeypatch.setattr(
        rag_service,
        "embed_query",
        lambda query: [1.0, 0.0],
    )

    def fake_search(
        store: object,
        vector: list[float],
        *,
        k: int,
        score_threshold: float,
        filters: dict[str, object],
    ) -> list[SearchResult]:
        captured["k"] = k
        captured["threshold"] = score_threshold
        return []

    monkeypatch.setattr(
        rag_service,
        "search_by_vector",
        fake_search,
    )

    response = retrieve_documents(
        "Consulta válida",
        k=2,
        score_threshold=0.75,
    )

    assert response.has_results is False
    assert response.context == ""
    assert captured["k"] == 2
    assert captured["threshold"] == 0.75


@pytest.mark.parametrize(
    "k, threshold, expected_message",
    [
        (0, 0.30, "k debe ser mayor que cero"),
        (4, -0.01, "score_threshold debe estar entre 0 y 1"),
        (4, 1.01, "score_threshold debe estar entre 0 y 1"),
    ],
)
def test_retrieve_documents_rejects_invalid_parameters(
    monkeypatch: pytest.MonkeyPatch,
    k: int,
    threshold: float,
    expected_message: str,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_k=4,
            retrieval_score_threshold=0.30,
        ),
    )

    with pytest.raises(
        RetrievalError,
        match=expected_message,
    ):
        retrieve_documents(
            "Consulta válida",
            k=k,
            score_threshold=threshold,
        )


@pytest.mark.parametrize(
    "error",
    [
        GeminiEmbeddingError("embedding failure"),
        FaissStoreError("faiss failure"),
    ],
)
def test_retrieve_documents_wraps_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_k=4,
            retrieval_score_threshold=0.30,
        ),
    )

    def raise_error() -> object:
        raise error

    monkeypatch.setattr(
        rag_service,
        "load_vector_store",
        raise_error,
    )

    with pytest.raises(
        RetrievalError,
        match="No fue posible completar la búsqueda semántica",
    ):
        retrieve_documents("Consulta válida")
