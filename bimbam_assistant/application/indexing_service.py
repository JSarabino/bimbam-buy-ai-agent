"""Servicio de preparación e indexación documental.

Este módulo coordina el proceso que transforma las páginas extraídas
del corpus en un índice vectorial consultable y se encarga de:

1. Construir y validar la configuración de fragmentación.
2. Dividir cada página en chunks sin mezclar contenido entre páginas.
3. Conservar y ampliar los metadatos necesarios para la trazabilidad.
4. Omitir páginas vacías y descartar fragmentos sin contenido.
5. Generar identificadores únicos para cada chunk.
6. Validar que los chunks estén listos para la indexación.
7. Generar embeddings mediante Google Gemini.
8. Verificar la correspondencia entre chunks y vectores.
9. Crear y persistir el índice FAISS y su manifiesto.
10. Devolver un resumen estructurado del proceso de indexación.

La carga de documentos se implementa en pdf_loader.py, la generación de
embeddings en gemini_provider.py y la persistencia vectorial en
faiss_store.py.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bimbam_assistant.core.config import get_settings
from bimbam_assistant.infrastructure.faiss_store import (
    FaissStoreError,
    create_and_save_vector_store,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    GeminiEmbeddingError,
    embed_documents,
)


logger = logging.getLogger(__name__)


class ChunkingError(ValueError):
    """Error producido por una configuración de chunking inválida."""


class IndexingError(RuntimeError):
    """Error producido durante la indexación vectorial."""


@dataclass(frozen=True)
class IndexingResult:
    """Resultado de la generación del índice vectorial."""

    chunk_count: int
    embedding_dimension: int
    output_path: Path
    manifest: dict[str, Any]


def build_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Construye el separador con la configuración del proyecto."""

    settings = get_settings()

    selected_chunk_size = (
        chunk_size
        if chunk_size is not None
        else settings.chunk_size
    )

    selected_chunk_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else settings.chunk_overlap
    )

    if selected_chunk_size <= 0:
        raise ChunkingError(
            "chunk_size debe ser mayor que cero."
        )

    if selected_chunk_overlap < 0:
        raise ChunkingError(
            "chunk_overlap no puede ser negativo."
        )

    if selected_chunk_overlap >= selected_chunk_size:
        raise ChunkingError(
            "chunk_overlap debe ser menor que chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=selected_chunk_size,
        chunk_overlap=selected_chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


def create_chunks(
    documents: Sequence[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Divide las páginas en chunks con metadatos trazables."""

    if not documents:
        logger.warning(
            "No se recibieron documentos para fragmentar."
        )
        return []

    settings = get_settings()

    selected_chunk_size = (
        chunk_size
        if chunk_size is not None
        else settings.chunk_size
    )

    selected_chunk_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else settings.chunk_overlap
    )

    splitter = build_text_splitter(
        chunk_size=selected_chunk_size,
        chunk_overlap=selected_chunk_overlap,
    )

    chunks: list[Document] = []
    skipped_empty_pages = 0

    for page_document in documents:
        page_content = page_document.page_content.strip()
        page_metadata = dict(page_document.metadata)

        is_empty = bool(
            page_metadata.get(
                "is_empty",
                False,
            )
        )

        if not page_content or is_empty:
            skipped_empty_pages += 1
            continue

        source_document = Document(
            page_content=page_content,
            metadata=page_metadata,
        )

        # Se procesa cada página por separado para que un chunk
        # nunca mezcle contenido de páginas diferentes.
        page_chunks = splitter.split_documents(
            [source_document]
        )

        document_id = str(
            page_metadata.get(
                "document_id",
                page_metadata.get(
                    "file_name",
                    "documento",
                ),
            )
        )

        page_number = page_metadata.get(
            "page_number",
            "sin_pagina",
        )

        for chunk_index, chunk in enumerate(page_chunks):
            chunk_content = chunk.page_content.strip()

            if not chunk_content:
                continue

            chunk_metadata = dict(
                chunk.metadata
            )

            chunk_id = (
                f"{document_id}"
                f"-page-{page_number}"
                f"-chunk-{chunk_index}"
            )

            chunk_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_number": chunk_index + 1,
                    "chunk_character_count": len(
                        chunk_content
                    ),
                    "configured_chunk_size": (
                        selected_chunk_size
                    ),
                    "configured_chunk_overlap": (
                        selected_chunk_overlap
                    ),
                }
            )

            chunks.append(
                Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata,
                )
            )

    logger.info(
        "Fragmentación finalizada: %s páginas recibidas, "
        "%s páginas vacías omitidas y %s chunks generados.",
        len(documents),
        skipped_empty_pages,
        len(chunks),
    )

    return chunks


def validate_chunks_for_indexing(
    chunks: Sequence[Document],
) -> None:
    """Valida que los chunks estén listos para generar embeddings."""

    if not chunks:
        raise IndexingError(
            "No existen chunks para indexar."
        )

    chunk_ids: list[str] = []

    for position, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            raise IndexingError(
                "Se encontró un chunk vacío en la posición "
                f"{position}."
            )

        chunk_id = str(
            chunk.metadata.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            raise IndexingError(
                "Se encontró un chunk sin chunk_id en la posición "
                f"{position}."
            )

        if not chunk.metadata.get("source"):
            raise IndexingError(
                "El chunk no contiene la fuente original: "
                f"{chunk_id}."
            )

        if "page_number" not in chunk.metadata:
            raise IndexingError(
                "El chunk no contiene número de página: "
                f"{chunk_id}."
            )

        category = str(
            chunk.metadata.get(
                "category",
                "sin_clasificar",
            )
        )

        if category == "sin_clasificar":
            raise IndexingError(
                "El chunk no tiene una categoría válida: "
                f"{chunk_id}."
            )

        chunk_ids.append(chunk_id)

    if len(chunk_ids) != len(set(chunk_ids)):
        raise IndexingError(
            "Existen identificadores de chunk duplicados."
        )


def create_vector_index(
    chunks: Sequence[Document],
    *,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> IndexingResult:
    """Genera embeddings y persiste el índice FAISS.

    Cada posición de la lista de chunks se mantiene alineada con la
    posición del embedding correspondiente.
    """

    validate_chunks_for_indexing(
        chunks
    )

    if batch_size <= 0:
        raise IndexingError(
            "batch_size debe ser mayor que cero."
        )

    settings = get_settings()

    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else settings.faiss_index_path
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    logger.info(
        "Generando embeddings para %s chunks.",
        len(texts),
    )

    try:
        vectors = embed_documents(
            texts,
            batch_size=batch_size,
        )

        if len(vectors) != len(chunks):
            raise IndexingError(
                "La cantidad de embeddings no coincide con "
                "la cantidad de chunks. "
                f"Chunks: {len(chunks)}. "
                f"Embeddings: {len(vectors)}."
            )

        if not vectors:
            raise IndexingError(
                "El proveedor no devolvió embeddings."
            )

        embedding_dimension = len(
            vectors[0]
        )

        manifest = create_and_save_vector_store(
            chunks,
            vectors,
            destination,
        )

    except IndexingError:
        raise

    except (
        GeminiEmbeddingError,
        FaissStoreError,
    ) as error:
        raise IndexingError(
            "No fue posible crear el índice vectorial: "
            f"{error}"
        ) from error

    logger.info(
        "Índice vectorial creado: %s chunks, dimensión %s, ruta %s.",
        len(chunks),
        embedding_dimension,
        destination,
    )

    return IndexingResult(
        chunk_count=len(chunks),
        embedding_dimension=embedding_dimension,
        output_path=destination,
        manifest=manifest,
    )


def process_and_index_documents(
    pages: Sequence[Document],
    *,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IndexingResult:
    """Ejecuta chunking, embeddings y persistencia de FAISS."""

    chunks = create_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return create_vector_index(
        chunks,
        output_path=output_path,
        batch_size=batch_size,
    )