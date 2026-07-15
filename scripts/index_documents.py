"""Ejecuta el procesamiento y la indexación documental.

Este script coordina:

1. La búsqueda de los documentos PDF.
2. La extracción y limpieza del texto.
3. La fragmentación en chunks.
4. La validación de metadatos.
5. La generación de embeddings con Gemini.
6. La creación y persistencia del índice FAISS.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document


# ==========================================================
# Preparación de la ruta del proyecto
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from bimbam_assistant.application.indexing_service import (
    ChunkingError,
    IndexingError,
    IndexingResult,
    create_chunks,
    create_vector_index,
    validate_chunks_for_indexing,
)
from bimbam_assistant.core.config import (
    ConfigurationError,
    get_settings,
)
from bimbam_assistant.infrastructure.pdf_loader import (
    PdfLoadingError,
    find_pdf_files,
    load_pdf_documents,
)


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configura los mensajes del proceso."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def count_pages_by_document(
    pages: list[Document],
) -> Counter[str]:
    """Cuenta las páginas procesadas por documento."""

    return Counter(
        str(
            page.metadata.get(
                "document_name",
                "Documento desconocido",
            )
        )
        for page in pages
    )


def count_chunks_by_document(
    chunks: list[Document],
) -> Counter[str]:
    """Cuenta los chunks generados por documento."""

    return Counter(
        str(
            chunk.metadata.get(
                "document_name",
                "Documento desconocido",
            )
        )
        for chunk in chunks
    )


def get_categories_by_document(
    pages: list[Document],
) -> dict[str, str]:
    """Relaciona cada documento con su categoría."""

    categories: dict[str, str] = {}

    for page in pages:
        document_name = str(
            page.metadata.get(
                "document_name",
                "Documento desconocido",
            )
        )

        category = str(
            page.metadata.get(
                "category",
                "sin_clasificar",
            )
        )

        categories[document_name] = category

    return categories


def print_document_processing_summary(
    *,
    pdf_files: list[Path],
    pages: list[Document],
    chunks: list[Document],
) -> None:
    """Muestra el resumen de extracción y chunking."""

    settings = get_settings()

    empty_pages = sum(
        bool(page.metadata.get("is_empty", False))
        for page in pages
    )

    ocr_candidates = sum(
        bool(page.metadata.get("ocr_candidate", False))
        for page in pages
    )

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    minimum_chunk_size = (
        min(chunk_lengths)
        if chunk_lengths
        else 0
    )

    maximum_chunk_size = (
        max(chunk_lengths)
        if chunk_lengths
        else 0
    )

    average_chunk_size = (
        round(mean(chunk_lengths), 2)
        if chunk_lengths
        else 0
    )

    page_counts = count_pages_by_document(pages)
    chunk_counts = count_chunks_by_document(chunks)
    categories = get_categories_by_document(pages)

    print()
    print("=" * 72)
    print("PROCESAMIENTO DOCUMENTAL DE BIMBAM ASSISTANT")
    print("=" * 72)

    print(f"Ruta de documentos : {settings.documents_path}")
    print(f"Archivos PDF       : {len(pdf_files)}")
    print(f"Páginas procesadas : {len(pages)}")
    print(f"Páginas vacías     : {empty_pages}")
    print(f"Candidatas a OCR   : {ocr_candidates}")
    print(f"Chunks generados   : {len(chunks)}")

    print()
    print("CONFIGURACIÓN DE CHUNKING")
    print("-" * 72)

    print(f"Chunk size         : {settings.chunk_size}")
    print(f"Chunk overlap      : {settings.chunk_overlap}")
    print(f"Tamaño mínimo      : {minimum_chunk_size}")
    print(f"Tamaño promedio    : {average_chunk_size}")
    print(f"Tamaño máximo      : {maximum_chunk_size}")

    print()
    print("DETALLE POR DOCUMENTO")
    print("-" * 72)

    for document_name in sorted(page_counts):
        print(
            f"- {document_name}\n"
            f"  Categoría: {categories[document_name]} | "
            f"Páginas: {page_counts[document_name]} | "
            f"Chunks: {chunk_counts[document_name]}"
        )


def print_indexing_summary(
    result: IndexingResult,
) -> None:
    """Muestra el resumen del índice vectorial generado."""

    manifest = result.manifest

    print()
    print("=" * 72)
    print("INDEXACIÓN VECTORIAL")
    print("=" * 72)

    print(
        f"Modelo de embeddings : "
        f"{manifest.get('embedding_model')}"
    )
    print(
        f"Dimensión             : "
        f"{result.embedding_dimension}"
    )
    print(
        f"Vectores almacenados  : "
        f"{manifest.get('vector_count')}"
    )
    print(
        f"Tipo de índice        : "
        f"{manifest.get('index_type')}"
    )
    print(
        f"Métrica               : "
        f"{manifest.get('distance_metric')}"
    )
    print(
        f"Documentos            : "
        f"{manifest.get('document_count')}"
    )
    print(
        f"Páginas representadas : "
        f"{manifest.get('page_count')}"
    )
    print(
        f"Categorías            : "
        f"{len(manifest.get('categories', []))}"
    )
    print(
        f"Ruta del índice       : "
        f"{result.output_path}"
    )

    print()
    print("ARCHIVOS GENERADOS")
    print("-" * 72)

    for file_name in (
        "index.faiss",
        "documents.json",
        "manifest.json",
    ):
        file_path = result.output_path / file_name
        status = "OK" if file_path.is_file() else "FALTANTE"

        print(
            f"{status:<8} | {file_name}"
        )

    print("=" * 72)


def main() -> int:
    """Ejecuta el pipeline de indexación completo."""

    configure_logging()

    try:
        settings = get_settings()

        logger.info(
            "Iniciando procesamiento documental."
        )

        pdf_files = find_pdf_files(
            settings.require_documents_path()
        )

        logger.info(
            "Se encontraron %s archivos PDF.",
            len(pdf_files),
        )

        pages = load_pdf_documents(
            settings.documents_path
        )

        logger.info(
            "Se extrajeron %s páginas.",
            len(pages),
        )

        chunks = create_chunks(
            pages
        )

        logger.info(
            "Se generaron %s chunks.",
            len(chunks),
        )

        # Valida antes de consumir la API de embeddings.
        validate_chunks_for_indexing(
            chunks
        )

        print_document_processing_summary(
            pdf_files=pdf_files,
            pages=pages,
            chunks=chunks,
        )

        logger.info(
            "Iniciando generación de embeddings."
        )

        indexing_result = create_vector_index(
            chunks,
            output_path=settings.faiss_index_path,
        )

        print_indexing_summary(
            indexing_result
        )

        logger.info(
            "Procesamiento e indexación finalizados correctamente."
        )

        return 0

    except (
        ConfigurationError,
        PdfLoadingError,
        ChunkingError,
        IndexingError,
    ) as error:
        logger.error(
            "No fue posible completar la indexación: %s",
            error,
        )

        return 1

    except KeyboardInterrupt:
        logger.warning(
            "La indexación fue interrumpida por el usuario."
        )

        return 130

    except Exception:
        logger.exception(
            "Se produjo un error inesperado durante la indexación."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())