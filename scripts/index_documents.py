"""Ejecuta el procesamiento documental de BimBam Assistant.

En la etapa actual, este script:

1. Localiza los archivos PDF.
2. Extrae y limpia el texto página por página.
3. Divide las páginas en chunks.
4. Verifica la trazabilidad y calidad básica de los resultados.
5. Muestra un resumen del procesamiento.

La generación de embeddings y la persistencia del índice FAISS se
incorporarán posteriormente.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


# ==========================================================
# Preparación de la ruta del proyecto
# ==========================================================

# Archivo actual:
# bimbam-buy-ai-agent/scripts/index_documents.py
#
# parents[0] -> scripts
# parents[1] -> raíz del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Permite ejecutar:
# python scripts/index_documents.py
#
# sin que Python pierda el acceso al paquete bimbam_assistant.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_core.documents import Document

from bimbam_assistant.application.indexing_service import (
    ChunkingError,
    create_chunks,
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
    """Configura los mensajes mostrados durante el procesamiento."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def _count_pages_by_document(
    pages: list[Document],
) -> Counter[str]:
    """Cuenta las páginas extraídas de cada documento."""

    return Counter(
        str(page.metadata.get("document_name", "Documento desconocido"))
        for page in pages
    )


def _count_chunks_by_document(
    chunks: list[Document],
) -> Counter[str]:
    """Cuenta los chunks generados por cada documento."""

    return Counter(
        str(chunk.metadata.get("document_name", "Documento desconocido"))
        for chunk in chunks
    )


def _get_category_by_document(
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


def validate_chunks(
    chunks: list[Document],
    *,
    configured_chunk_size: int,
) -> dict[str, object]:
    """Ejecuta comprobaciones básicas sobre los chunks."""

    chunk_ids = [
        str(chunk.metadata.get("chunk_id", ""))
        for chunk in chunks
    ]

    chunks_without_id = sum(
        not chunk_id
        for chunk_id in chunk_ids
    )

    unique_ids = len(
        {
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id
        }
    )

    duplicated_ids = (
        len(chunk_ids)
        - chunks_without_id
        - unique_ids
    )

    chunks_without_source = sum(
        not chunk.metadata.get("source")
        for chunk in chunks
    )

    chunks_without_page = sum(
        "page_number" not in chunk.metadata
        for chunk in chunks
    )

    unclassified_chunks = sum(
        chunk.metadata.get("category") == "sin_clasificar"
        for chunk in chunks
    )

    oversized_chunks = sum(
        len(chunk.page_content) > configured_chunk_size
        for chunk in chunks
    )

    return {
        "chunks_without_id": chunks_without_id,
        "duplicated_ids": duplicated_ids,
        "chunks_without_source": chunks_without_source,
        "chunks_without_page": chunks_without_page,
        "unclassified_chunks": unclassified_chunks,
        "oversized_chunks": oversized_chunks,
    }


def print_processing_summary(
    *,
    pdf_files: list[Path],
    pages: list[Document],
    chunks: list[Document],
) -> None:
    """Muestra un resumen legible del procesamiento documental."""

    settings = get_settings()

    empty_pages = sum(
        bool(page.metadata.get("is_empty", False))
        for page in pages
    )

    ocr_candidates = sum(
        bool(page.metadata.get("ocr_candidate", False))
        for page in pages
    )

    native_text_pages = len(pages) - empty_pages

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

    page_counts = _count_pages_by_document(pages)
    chunk_counts = _count_chunks_by_document(chunks)
    categories = _get_category_by_document(pages)

    validations = validate_chunks(
        chunks,
        configured_chunk_size=settings.chunk_size,
    )

    recognized_categories = sorted(
        {
            str(chunk.metadata.get("category"))
            for chunk in chunks
            if chunk.metadata.get("category")
        }
    )

    print()
    print("=" * 68)
    print("PROCESAMIENTO DOCUMENTAL DE BIMBAM ASSISTANT")
    print("=" * 68)

    print(f"Ruta de documentos : {settings.documents_path}")
    print(f"Archivos PDF       : {len(pdf_files)}")
    print(f"Páginas procesadas : {len(pages)}")
    print(f"Páginas con texto  : {native_text_pages}")
    print(f"Páginas vacías     : {empty_pages}")
    print(f"Candidatas a OCR   : {ocr_candidates}")
    print(f"Chunks generados   : {len(chunks)}")

    print()
    print("CONFIGURACIÓN DE CHUNKING")
    print("-" * 68)
    print(f"Chunk size         : {settings.chunk_size}")
    print(f"Chunk overlap      : {settings.chunk_overlap}")
    print(f"Tamaño mínimo      : {minimum_chunk_size}")
    print(f"Tamaño promedio    : {average_chunk_size}")
    print(f"Tamaño máximo      : {maximum_chunk_size}")

    print()
    print("CATEGORÍAS")
    print("-" * 68)

    for category in recognized_categories:
        print(f"- {category}")

    print()
    print("DETALLE POR DOCUMENTO")
    print("-" * 68)

    for document_name in sorted(page_counts):
        category = categories.get(
            document_name,
            "sin_clasificar",
        )

        print(
            f"- {document_name}\n"
            f"  Categoría: {category} | "
            f"Páginas: {page_counts[document_name]} | "
            f"Chunks: {chunk_counts[document_name]}"
        )

    print()
    print("VALIDACIONES")
    print("-" * 68)

    validation_rows = [
        (
            "Chunks sin identificador",
            int(validations["chunks_without_id"]),
        ),
        (
            "Identificadores duplicados",
            int(validations["duplicated_ids"]),
        ),
        (
            "Chunks sin fuente",
            int(validations["chunks_without_source"]),
        ),
        (
            "Chunks sin página",
            int(validations["chunks_without_page"]),
        ),
        (
            "Chunks sin clasificar",
            int(validations["unclassified_chunks"]),
        ),
        (
            "Chunks que superan el tamaño",
            int(validations["oversized_chunks"]),
        ),
    ]

    for label, error_count in validation_rows:
        status = "OK" if error_count == 0 else "REVISAR"

        print(
            f"{status:<8} | {label}: {error_count}"
        )

    print()
    print(
        "Nota: en esta etapa todavía no se generan embeddings "
        "ni el índice FAISS."
    )
    print("=" * 68)


def has_processing_errors(
    chunks: list[Document],
) -> bool:
    """Indica si existen errores que impidan continuar con embeddings."""

    settings = get_settings()

    if not chunks:
        return True

    validations = validate_chunks(
        chunks,
        configured_chunk_size=settings.chunk_size,
    )

    critical_fields = (
        "chunks_without_id",
        "duplicated_ids",
        "chunks_without_source",
        "chunks_without_page",
        "unclassified_chunks",
        "oversized_chunks",
    )

    return any(
        int(validations[field]) > 0
        for field in critical_fields
    )


def main() -> int:
    """Ejecuta el flujo completo de preparación documental."""

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

        chunks = create_chunks(pages)

        logger.info(
            "Se generaron %s chunks.",
            len(chunks),
        )

        print_processing_summary(
            pdf_files=pdf_files,
            pages=pages,
            chunks=chunks,
        )

        if has_processing_errors(chunks):
            logger.error(
                "El procesamiento terminó con errores de validación."
            )
            return 1

        logger.info(
            "Procesamiento finalizado correctamente."
        )

        return 0

    except (
        ConfigurationError,
        PdfLoadingError,
        ChunkingError,
    ) as error:
        logger.error(
            "No fue posible completar el procesamiento: %s",
            error,
        )
        return 1

    except Exception:
        logger.exception(
            "Se produjo un error inesperado durante el procesamiento."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())