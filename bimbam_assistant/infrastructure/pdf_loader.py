"""Carga y extracción de documentos PDF de BimBam Assistant.

Este módulo se encarga de:

1. Localizar los archivos PDF configurados.
2. Extraer el texto página por página con PyMuPDF.
3. Aplicar una limpieza conservadora al texto.
4. Crear objetos Document de LangChain.
5. Conservar metadatos para la citación de fuentes.
6. Detectar páginas vacías o potencialmente escaneadas.

El chunking se implementa en indexing_service.py.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf
import unicodedata
from langchain_core.documents import Document

from bimbam_assistant.core.config import get_settings


logger = logging.getLogger(__name__)


# Una página con menos caracteres puede estar vacía, contener únicamente
# una imagen o necesitar OCR. Es solo una regla heurística.
MIN_NATIVE_TEXT_LENGTH = 30


def _normalize_identifier(value: str) -> str:
    """Convierte un nombre en un identificador estable.

    Ejemplo:
        Política de Reembolsos y Devoluciones
        ->
        politica_de_reembolsos_y_devoluciones
    """

    normalized_value = unicodedata.normalize(
        "NFKD",
        value,
    )

    without_accents = "".join(
        character
        for character in normalized_value
        if not unicodedata.combining(character)
    )

    identifier = re.sub(
        r"[^a-z0-9]+",
        "_",
        without_accents.lower(),
    )

    return identifier.strip("_")

# Información conocida del corpus inicial.
#
# Permite mostrar nombres y categorías comprensibles sin depender
# únicamente de los metadatos internos de cada PDF.
DOCUMENT_CATALOG: dict[str, dict[str, str]] = {
    "guia_de_tiempos_y_costos_de_envio_de_bimbam_buy": {
        "document_name": (
            "Guía de Tiempos y Costos de Envío de BimBam Buy"
        ),
        "category": "envios",
    },
    "manual_de_garantia_de_productos_de_bimbam_buy": {
        "document_name": (
            "Manual de Garantía de Productos de BimBam Buy"
        ),
        "category": "garantias",
    },
    "politica_de_reembolsos_y_devoluciones_de_bimbam": {
        "document_name": (
            "Política de Reembolsos y Devoluciones de BimBam"
        ),
        "category": "reembolsos_devoluciones",
    },
    "politica_de_reembolsos_y_devoluciones_de_bimbam_buy": {
        "document_name": (
            "Política de Reembolsos y Devoluciones de BimBam Buy"
        ),
        "category": "reembolsos_devoluciones",
    },
    "preguntas_frecuentes_sobre_metodos_de_pago_de_bimbam_buy": {
        "document_name": (
            "Preguntas Frecuentes sobre Métodos de Pago de BimBam Buy"
        ),
        "category": "metodos_pago",
    },
    "programa_de_afiliados_de_bimbam_buy": {
        "document_name": (
            "Programa de Afiliados de BimBam Buy"
        ),
        "category": "afiliados",
    },
}


class PdfLoadingError(RuntimeError):
    """Error producido durante la búsqueda o lectura de un PDF."""


def clean_text(text: str) -> str:
    """Aplica una limpieza conservadora al texto extraído.

    La función elimina caracteres invisibles, normaliza espacios
    y reduce saltos de línea repetidos.

    No elimina automáticamente encabezados, pies de página ni números,
    porque podrían contener información relevante.
    """

    if not text:
        return ""

    # Elimina caracteres invisibles frecuentes en documentos PDF.
    cleaned_text = (
        text.replace("\x00", "")
        .replace("\u00ad", "")
        .replace("\u200b", "")
    )

    # Normaliza los diferentes tipos de salto de línea.
    cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []

    for line in cleaned_text.split("\n"):
        # Convierte espacios, tabulaciones y otros espacios horizontales
        # repetidos en un único espacio.
        normalized_line = re.sub(
            r"[ \t\f\v]+",
            " ",
            line,
        ).strip()

        cleaned_lines.append(normalized_line)

    cleaned_text = "\n".join(cleaned_lines)

    # Conserva la separación entre párrafos, pero elimina acumulaciones
    # innecesarias de tres o más saltos de línea.
    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    )

    return cleaned_text.strip()


def find_pdf_files(
    documents_path: Path | None = None,
) -> list[Path]:
    """Busca los archivos PDF disponibles en la carpeta documental.

    Cuando no se proporciona una ruta, utiliza DOCUMENTS_PATH desde
    la configuración central del proyecto.
    """

    if documents_path is None:
        settings = get_settings()
        documents_directory = settings.require_documents_path()
    else:
        documents_directory = documents_path.expanduser().resolve()

        if not documents_directory.exists():
            raise PdfLoadingError(
                "No se encontró la carpeta de documentos: "
                f"{documents_directory}"
            )

        if not documents_directory.is_dir():
            raise PdfLoadingError(
                "La ruta proporcionada no corresponde a una carpeta: "
                f"{documents_directory}"
            )

    pdf_files = sorted(
        (
            path
            for path in documents_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )

    if not pdf_files:
        raise PdfLoadingError(
            "No se encontraron archivos PDF en: "
            f"{documents_directory}"
        )

    return pdf_files


def _clean_metadata_value(value: object) -> str | None:
    """Normaliza un valor obtenido de los metadatos internos del PDF."""

    if value is None:
        return None

    cleaned_value = str(value).strip()

    if not cleaned_value:
        return None

    if cleaned_value.lower() == "none":
        return None

    return cleaned_value


def _humanize_file_name(pdf_path: Path) -> str:
    """Genera un nombre legible a partir del nombre del archivo."""

    return (
        pdf_path.stem
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def _resolve_source_path(
    pdf_path: Path,
    project_root: Path,
) -> str:
    """Genera una ruta de fuente adecuada para los metadatos.

    Cuando el documento se encuentra dentro del proyecto, devuelve una
    ruta relativa portable, como:

        data/documents/manual_garantia_productos.pdf

    Si está fuera del proyecto, conserva su ruta absoluta.
    """

    try:
        return pdf_path.relative_to(project_root).as_posix()
    except ValueError:
        return str(pdf_path)


def _is_ocr_candidate(
    page: pymupdf.Page,
    cleaned_text: str,
) -> bool:
    """Identifica páginas que podrían necesitar OCR.

    Una página se marca como candidata cuando:

    - Tiene muy poco texto extraído.
    - Contiene al menos una imagen.

    Esta regla no confirma que el OCR sea obligatorio; únicamente
    identifica páginas que deberían revisarse.
    """

    if len(cleaned_text) >= MIN_NATIVE_TEXT_LENGTH:
        return False

    try:
        return bool(page.get_images(full=True))
    except Exception:
        # La detección de imágenes no debe impedir la extracción del PDF.
        logger.warning(
            "No fue posible inspeccionar las imágenes de la página %s.",
            page.number + 1,
        )
        return False


def load_pdf(
    pdf_path: Path,
    *,
    project_root: Path | None = None,
) -> list[Document]:
    """Carga un PDF y devuelve un Document de LangChain por página."""

    resolved_pdf_path = pdf_path.expanduser().resolve()

    if not resolved_pdf_path.exists():
        raise PdfLoadingError(
            f"No se encontró el archivo PDF: {resolved_pdf_path}"
        )

    if not resolved_pdf_path.is_file():
        raise PdfLoadingError(
            f"La ruta no corresponde a un archivo: {resolved_pdf_path}"
        )

    if resolved_pdf_path.suffix.lower() != ".pdf":
        raise PdfLoadingError(
            f"El archivo no tiene extensión PDF: {resolved_pdf_path}"
        )

    settings = get_settings()
    root_path = project_root or settings.project_root

    source = _resolve_source_path(
        resolved_pdf_path,
        root_path,
    )

    document_id = _normalize_identifier(
        resolved_pdf_path.stem
    )

    catalog_entry = DOCUMENT_CATALOG.get(
        document_id,
        {},
    )

    documents: list[Document] = []

    try:
        with pymupdf.open(resolved_pdf_path) as pdf:
            if pdf.needs_pass:
                raise PdfLoadingError(
                    "El documento está protegido con contraseña: "
                    f"{resolved_pdf_path.name}"
                )

            pdf_metadata = pdf.metadata or {}

            internal_title = _clean_metadata_value(
                pdf_metadata.get("title")
            )

            document_name = (
                catalog_entry.get("document_name")
                or internal_title
                or resolved_pdf_path.stem
            )

            category = catalog_entry.get(
                "category",
                "sin_clasificar",
            )

            author = _clean_metadata_value(
                pdf_metadata.get("author")
            )

            creation_date = _clean_metadata_value(
                pdf_metadata.get("creationDate")
            )

            modification_date = _clean_metadata_value(
                pdf_metadata.get("modDate")
            )

            total_pages = pdf.page_count

            for page_index in range(total_pages):
                page = pdf.load_page(page_index)

                raw_text = page.get_text(
                    "text",
                    sort=True,
                )

                cleaned_text = clean_text(raw_text)

                page_number = page_index + 1
                page_label = page.get_label() or str(page_number)

                is_empty = not bool(cleaned_text)

                ocr_candidate = _is_ocr_candidate(
                    page,
                    cleaned_text,
                )

                metadata: dict[str, object] = {
                    "company": "BimBam Buy",
                    "document_id": document_id,
                    "document_name": document_name,
                    "category": category,
                    "source": source,
                    "file_name": resolved_pdf_path.name,
                    "page_index": page_index,
                    "page_number": page_number,
                    "page_label": page_label,
                    "total_pages": total_pages,
                    "extraction_method": "native_text",
                    "character_count": len(cleaned_text),
                    "is_empty": is_empty,
                    "ocr_candidate": ocr_candidate,
                }

                # Solo agregamos los metadatos opcionales cuando existen.
                if author:
                    metadata["author"] = author

                if creation_date:
                    metadata["creation_date"] = creation_date

                if modification_date:
                    metadata["modification_date"] = modification_date

                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata=metadata,
                    )
                )

                if is_empty:
                    logger.warning(
                        "Página sin texto: %s, página %s.",
                        resolved_pdf_path.name,
                        page_number,
                    )
                elif ocr_candidate:
                    logger.warning(
                        "Posible página escaneada: %s, página %s.",
                        resolved_pdf_path.name,
                        page_number,
                    )

    except PdfLoadingError:
        raise
    except Exception as error:
        raise PdfLoadingError(
            "No fue posible procesar el archivo "
            f"{resolved_pdf_path.name}: {error}"
        ) from error

    logger.info(
        "Archivo cargado: %s (%s páginas).",
        resolved_pdf_path.name,
        len(documents),
    )

    return documents


def load_pdf_documents(
    documents_path: Path | None = None,
) -> list[Document]:
    """Carga todos los PDF y devuelve sus páginas como documentos."""

    settings = get_settings()

    pdf_files = find_pdf_files(
        documents_path=documents_path,
    )

    documents: list[Document] = []

    for pdf_path in pdf_files:
        pdf_documents = load_pdf(
            pdf_path,
            project_root=settings.project_root,
        )

        documents.extend(pdf_documents)

    logger.info(
        "Carga terminada: %s archivos PDF y %s páginas.",
        len(pdf_files),
        len(documents),
    )

    return documents