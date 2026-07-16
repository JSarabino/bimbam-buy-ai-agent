"""Servicio de recuperación semántica de BimBam Assistant.

Este módulo se encarga de:

1. Validar y normalizar la pregunta.
2. Generar su embedding con Gemini.
3. Consultar el índice FAISS.
4. Aplicar top-k, umbral y filtros.
5. Convertir los resultados en modelos de dominio.
6. Ensamblar el contexto para la futura cadena RAG.

La generación de la respuesta con un modelo de chat se incorporará
posteriormente.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from bimbam_assistant.core.config import get_settings
from bimbam_assistant.domain.models import (
    RetrievedChunk,
    RetrievalResponse,
)
from bimbam_assistant.infrastructure.faiss_store import (
    FaissStoreError,
    SearchResult,
    load_vector_store,
    search_by_vector,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    GeminiEmbeddingError,
    embed_query,
)


logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Error producido durante la recuperación semántica."""


def normalize_query(query: str) -> str:
    """Limpia y valida una pregunta antes de procesarla."""

    if not isinstance(query, str):
        raise RetrievalError(
            "La consulta debe ser una cadena de texto."
        )

    normalized_query = " ".join(
        query.split()
    )

    if not normalized_query:
        raise RetrievalError(
            "La consulta no puede estar vacía."
        )

    return normalized_query


def build_retrieved_chunks(
    search_results: Sequence[SearchResult],
) -> list[RetrievedChunk]:
    """Convierte resultados de infraestructura en modelos de dominio."""

    retrieved_chunks: list[RetrievedChunk] = []

    for rank, result in enumerate(
        search_results,
        start=1,
    ):
        retrieved_chunks.append(
            RetrievedChunk(
                rank=rank,
                vector_id=result.vector_id,
                score=result.score,
                page_content=result.document.page_content,
                metadata=dict(result.document.metadata),
            )
        )

    return retrieved_chunks


def build_context(
    results: Sequence[RetrievedChunk],
) -> str:
    """Construye el bloque de contexto que recibirá el LLM.

    Cada fragmento conserva el documento, la página, la categoría
    y su identificador para facilitar la trazabilidad.
    """

    if not results:
        return ""

    context_sections: list[str] = []

    for result in results:
        metadata = result.metadata

        document_name = str(
            metadata.get(
                "document_name",
                "Documento desconocido",
            )
        )

        page_number = metadata.get(
            "page_number",
            "No disponible",
        )

        category = str(
            metadata.get(
                "category",
                "sin_clasificar",
            )
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
                f"vector-{result.vector_id}",
            )
        )

        section = (
            f"[Fuente {result.rank}]\n"
            f"Documento: {document_name}\n"
            f"Página: {page_number}\n"
            f"Categoría: {category}\n"
            f"Chunk: {chunk_id}\n"
            f"Similitud: {result.score:.4f}\n\n"
            f"{result.page_content}"
        )

        context_sections.append(
            section
        )

    return "\n\n---\n\n".join(
        context_sections
    )


def retrieve_documents(
    query: str,
    *,
    k: int | None = None,
    score_threshold: float | None = None,
    filters: Mapping[str, object] | None = None,
) -> RetrievalResponse:
    """Recupera los chunks más relevantes para una pregunta.

    Cuando ``k`` o ``score_threshold`` no se proporcionan, se utilizan
    los valores definidos en la configuración central.
    """

    settings = get_settings()

    normalized_query = normalize_query(
        query
    )

    selected_k = (
        k
        if k is not None
        else settings.retrieval_k
    )

    selected_threshold = (
        score_threshold
        if score_threshold is not None
        else settings.retrieval_score_threshold
    )

    if selected_k <= 0:
        raise RetrievalError(
            "k debe ser mayor que cero."
        )

    if not 0 <= selected_threshold <= 1:
        raise RetrievalError(
            "score_threshold debe estar entre 0 y 1."
        )

    normalized_filters = (
        dict(filters)
        if filters
        else {}
    )

    logger.info(
        "Iniciando recuperación: query=%r, k=%s, umbral=%s, filtros=%s.",
        normalized_query,
        selected_k,
        selected_threshold,
        normalized_filters,
    )

    try:
        store = load_vector_store()

        query_vector = embed_query(
            normalized_query
        )

        search_results = search_by_vector(
            store,
            query_vector,
            k=selected_k,
            score_threshold=selected_threshold,
            filters=normalized_filters,
        )

    except (
        GeminiEmbeddingError,
        FaissStoreError,
    ) as error:
        raise RetrievalError(
            "No fue posible completar la búsqueda semántica: "
            f"{error}"
        ) from error

    retrieved_chunks = build_retrieved_chunks(
        search_results
    )

    context = build_context(
        retrieved_chunks
    )

    logger.info(
        "Recuperación finalizada: %s resultados encontrados.",
        len(retrieved_chunks),
    )

    return RetrievalResponse(
        query=normalized_query,
        results=retrieved_chunks,
        context=context,
        filters=normalized_filters,
    )