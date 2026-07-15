"""Servicio de preparación de documentos para la indexación.

Este módulo se encarga de convertir las páginas extraídas
de los PDF en fragmentos más pequeños.

En una etapa posterior también coordinará:

1. La generación de embeddings.
2. La creación del índice FAISS.
3. La persistencia del índice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bimbam_assistant.core.config import get_settings


logger = logging.getLogger(__name__)


class ChunkingError(ValueError):
    """Error producido por parámetros de fragmentación inválidos."""


def build_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Construye el separador de texto con la configuración del proyecto.

    Los parámetros recibidos directamente tienen prioridad sobre los
    valores definidos en el archivo .env.
    """

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
    """Divide documentos de página en chunks con metadatos trazables.

    Cada elemento recibido representa una página extraída previamente
    por pdf_loader.py.

    Las páginas vacías se omiten porque no contienen información útil
    para generar embeddings.
    """

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
            page_metadata.get("is_empty", False)
        )

        if not page_content or is_empty:
            skipped_empty_pages += 1
            continue

        # Creamos una copia para no modificar el Document original.
        source_document = Document(
            page_content=page_content,
            metadata=page_metadata,
        )

        # Como se envía una sola página, ningún chunk puede mezclar
        # contenido perteneciente a páginas diferentes.
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

            chunk_metadata = dict(chunk.metadata)

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