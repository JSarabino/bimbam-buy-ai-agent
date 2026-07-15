"""Punto de entrada de la aplicación Streamlit de BimBam Assistant."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st

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


@st.cache_data(show_spinner=False)
def build_processing_snapshot(
    documents_path: str,
    document_signature: tuple[tuple[str, int, int], ...],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    """Procesa el corpus y devuelve datos simples para la interfaz.

    La firma de archivos forma parte de la clave de caché. Cuando un PDF
    cambia de nombre, tamaño o fecha de modificación, Streamlit vuelve a
    procesar el corpus automáticamente.
    """

    del document_signature

    resolved_documents_path = Path(documents_path)

    pages = load_pdf_documents(resolved_documents_path)
    chunks = create_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    pages_by_document = Counter(
        str(page.metadata["document_name"])
        for page in pages
    )
    chunks_by_document = Counter(
        str(chunk.metadata["document_name"])
        for chunk in chunks
    )
    category_by_document = {
        str(page.metadata["document_name"]): str(page.metadata["category"])
        for page in pages
    }

    details = [
        {
            "Documento": document_name,
            "Categoría": category_by_document[document_name],
            "Páginas": pages_by_document[document_name],
            "Chunks": chunks_by_document[document_name],
        }
        for document_name in sorted(pages_by_document)
    ]

    chunk_ids = [
        str(chunk.metadata.get("chunk_id", ""))
        for chunk in chunks
    ]

    maximum_chunk_size = max(
        (len(chunk.page_content) for chunk in chunks),
        default=0,
    )

    validation_errors = {
        "chunks_without_id": sum(not chunk_id for chunk_id in chunk_ids),
        "duplicated_ids": len(chunk_ids) - len(set(chunk_ids)),
        "chunks_without_source": sum(
            not chunk.metadata.get("source")
            for chunk in chunks
        ),
        "chunks_without_page": sum(
            "page_number" not in chunk.metadata
            for chunk in chunks
        ),
        "unclassified_chunks": sum(
            chunk.metadata.get("category") == "sin_clasificar"
            for chunk in chunks
        ),
        "oversized_chunks": sum(
            len(chunk.page_content) > chunk_size
            for chunk in chunks
        ),
    }

    return {
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "empty_pages": sum(
            bool(page.metadata.get("is_empty", False))
            for page in pages
        ),
        "ocr_candidates": sum(
            bool(page.metadata.get("ocr_candidate", False))
            for page in pages
        ),
        "category_count": len(
            {
                str(chunk.metadata.get("category"))
                for chunk in chunks
            }
        ),
        "maximum_chunk_size": maximum_chunk_size,
        "details": details,
        "validation_errors": validation_errors,
        "processing_ready": (
            bool(chunks)
            and not any(validation_errors.values())
        ),
    }


def build_document_signature(
    pdf_files: list[Path],
) -> tuple[tuple[str, int, int], ...]:
    """Construye una firma que permite invalidar la caché al cambiar un PDF."""

    return tuple(
        (
            pdf_file.name,
            pdf_file.stat().st_size,
            pdf_file.stat().st_mtime_ns,
        )
        for pdf_file in pdf_files
    )


def render_sidebar() -> None:
    """Muestra la configuración pública del proyecto."""

    settings = get_settings()

    with st.sidebar:
        st.header("Configuración")

        st.write(f"**Aplicación:** {settings.app_name}")
        st.write(f"**Versión:** {settings.app_version}")
        st.write(f"**Entorno:** {settings.app_environment}")

        st.divider()

        st.write(f"**Chunk size:** {settings.chunk_size}")
        st.write(f"**Chunk overlap:** {settings.chunk_overlap}")
        st.write(f"**Modelo:** {settings.gemini_chat_model}")
        st.write(
            f"**Embeddings:** {settings.gemini_embedding_model}"
        )

        st.divider()

        st.caption(
            "La clave de Gemini nunca se muestra en la interfaz."
        )


def main() -> None:
    """Renderiza la pantalla inicial de la aplicación."""

    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.error(f"Error de configuración: {error}")
        return

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🛍️ {settings.app_name}")
    st.caption(
        "Agente inteligente para consultar las políticas y los "
        "documentos corporativos de BimBam Buy."
    )

    st.markdown(
        """
        La preparación documental ya está implementada: los PDF se leen
        página por página, se limpian, se clasifican y se dividen en
        fragmentos trazables. La siguiente etapa incorporará embeddings
        con Gemini y la persistencia del índice vectorial en FAISS.
        """
    )

    render_sidebar()

    try:
        pdf_files = find_pdf_files(
            settings.require_documents_path()
        )

        document_signature = build_document_signature(pdf_files)

        with st.spinner(
            "Validando la lectura y fragmentación del corpus..."
        ):
            processing = build_processing_snapshot(
                str(settings.documents_path),
                document_signature,
                settings.chunk_size,
                settings.chunk_overlap,
            )

    except (
        ConfigurationError,
        PdfLoadingError,
        ChunkingError,
    ) as error:
        st.error(
            "No fue posible procesar el corpus documental: "
            f"{error}"
        )
        return

    st.subheader("Estado del proyecto")

    document_column, page_column, chunk_column, index_column = st.columns(4)

    with document_column:
        st.metric(
            label="Documentos PDF",
            value=len(pdf_files),
        )

    with page_column:
        st.metric(
            label="Páginas procesadas",
            value=int(processing["page_count"]),
        )

    with chunk_column:
        st.metric(
            label="Chunks generados",
            value=int(processing["chunk_count"]),
        )

    with index_column:
        st.metric(
            label="Índice FAISS",
            value=(
                "Disponible"
                if settings.faiss_index_exists
                else "Pendiente"
            ),
        )

    if bool(processing["processing_ready"]):
        st.success(
            "La lectura, limpieza, clasificación y fragmentación "
            "del corpus finalizaron correctamente."
        )
    else:
        st.warning(
            "El procesamiento terminó, pero existen validaciones "
            "que deben revisarse antes de generar embeddings."
        )

    if not settings.google_api_key_configured:
        st.warning(
            "GOOGLE_API_KEY no está configurada. La preparación "
            "documental funciona sin ella, pero será necesaria para "
            "generar embeddings y respuestas."
        )
    else:
        st.info(
            "La clave de Gemini está configurada. Todavía no se utiliza "
            "durante la lectura y fragmentación de los PDF."
        )

    if not settings.faiss_index_exists:
        st.info(
            "El índice FAISS aún no ha sido generado. La siguiente "
            "etapa convertirá los chunks en embeddings y persistirá "
            "el índice en storage/faiss_index/."
        )

    st.subheader("Resumen del procesamiento")

    empty_column, ocr_column, category_column, size_column = st.columns(4)

    with empty_column:
        st.metric(
            label="Páginas vacías",
            value=int(processing["empty_pages"]),
        )

    with ocr_column:
        st.metric(
            label="Candidatas a OCR",
            value=int(processing["ocr_candidates"]),
        )

    with category_column:
        st.metric(
            label="Categorías",
            value=int(processing["category_count"]),
        )

    with size_column:
        st.metric(
            label="Chunk máximo",
            value=f"{int(processing['maximum_chunk_size'])} caracteres",
        )

    with st.expander("Ver detalle por documento", expanded=True):
        st.dataframe(
            processing["details"],
            use_container_width=True,
            hide_index=True,
        )

    validation_errors = processing["validation_errors"]

    with st.expander("Ver validaciones técnicas"):
        st.write(
            {
                "Chunks sin identificador": validation_errors[
                    "chunks_without_id"
                ],
                "Identificadores duplicados": validation_errors[
                    "duplicated_ids"
                ],
                "Chunks sin fuente": validation_errors[
                    "chunks_without_source"
                ],
                "Chunks sin página": validation_errors[
                    "chunks_without_page"
                ],
                "Chunks sin clasificar": validation_errors[
                    "unclassified_chunks"
                ],
                "Chunks que superan el tamaño": validation_errors[
                    "oversized_chunks"
                ],
            }
        )

    with st.expander("Ver documentos detectados"):
        for pdf_file in pdf_files:
            st.write(f"• {pdf_file.name}")

    st.subheader("Consulta documental")

    st.text_input(
        label="Escribe una pregunta sobre BimBam Buy",
        placeholder="Ejemplo: ¿Cuánto tarda un reembolso?",
        disabled=True,
        help=(
            "El campo se habilitará cuando estén implementados "
            "los embeddings, FAISS y el servicio RAG."
        ),
    )

    st.caption(
        "Siguiente hito: generar embeddings con Gemini y construir "
        "el índice vectorial FAISS."
    )


if __name__ == "__main__":
    main()
