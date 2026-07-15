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
from bimbam_assistant.infrastructure.faiss_store import (
    FaissStoreError,
    load_vector_store,
)
from bimbam_assistant.infrastructure.pdf_loader import (
    PdfLoadingError,
    find_pdf_files,
    load_pdf_documents,
)


INDEX_FILE_NAMES = (
    "index.faiss",
    "documents.json",
    "manifest.json",
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

    pages = load_pdf_documents(
        Path(documents_path)
    )

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

    validation_errors = {
        "chunks_without_id": sum(
            not chunk_id
            for chunk_id in chunk_ids
        ),
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
        "maximum_chunk_size": max(
            (len(chunk.page_content) for chunk in chunks),
            default=0,
        ),
        "details": details,
        "validation_errors": validation_errors,
        "processing_ready": (
            bool(chunks)
            and not any(validation_errors.values())
        ),
    }


@st.cache_data(show_spinner=False)
def build_index_snapshot(
    index_path: str,
    index_signature: tuple[tuple[str, int, int], ...],
) -> dict[str, object]:
    """Carga y valida el índice persistido para mostrar su estado."""

    del index_signature

    store = load_vector_store(
        Path(index_path)
    )

    manifest = dict(
        store.manifest
    )

    return {
        "vector_count": int(store.index.ntotal),
        "embedding_dimension": int(store.index.d),
        "embedding_model": str(
            manifest.get("embedding_model", "Desconocido")
        ),
        "index_type": str(
            manifest.get("index_type", "Desconocido")
        ),
        "distance_metric": str(
            manifest.get("distance_metric", "Desconocida")
        ),
        "document_count": int(
            manifest.get("document_count", 0)
        ),
        "page_count": int(
            manifest.get("page_count", 0)
        ),
        "categories": list(
            manifest.get("categories", [])
        ),
        "created_at_utc": str(
            manifest.get("created_at_utc", "")
        ),
        "manifest": manifest,
    }


def build_file_signature(
    files: list[Path],
) -> tuple[tuple[str, int, int], ...]:
    """Crea una firma para invalidar la caché cuando cambia un archivo."""

    return tuple(
        (
            file.name,
            file.stat().st_size,
            file.stat().st_mtime_ns,
        )
        for file in files
    )


def get_index_files(
    index_path: Path,
) -> list[Path]:
    """Devuelve los archivos esperados del almacén vectorial."""

    return [
        index_path / file_name
        for file_name in INDEX_FILE_NAMES
    ]


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
        st.write(f"**Top k:** {settings.retrieval_k}")
        st.write(
            f"**Umbral:** {settings.retrieval_score_threshold}"
        )

        st.divider()

        st.write(f"**Modelo de chat:** {settings.gemini_chat_model}")
        st.write(
            f"**Modelo de embeddings:** "
            f"{settings.gemini_embedding_model}"
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
        La preparación e indexación documental ya están implementadas.
        Los PDF se procesan en chunks trazables, se convierten en
        embeddings con Gemini y se almacenan en un índice FAISS local.
        El siguiente hito es exponer la recuperación semántica y usar
        los fragmentos recuperados para construir respuestas con fuentes.
        """
    )

    render_sidebar()

    try:
        pdf_files = find_pdf_files(
            settings.require_documents_path()
        )

        with st.spinner(
            "Validando la lectura y fragmentación del corpus..."
        ):
            processing = build_processing_snapshot(
                str(settings.documents_path),
                build_file_signature(pdf_files),
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

    index_snapshot: dict[str, object] | None = None
    index_error: str | None = None

    if settings.faiss_index_exists:
        try:
            index_files = get_index_files(
                settings.faiss_index_path
            )

            with st.spinner(
                "Validando el índice vectorial..."
            ):
                index_snapshot = build_index_snapshot(
                    str(settings.faiss_index_path),
                    build_file_signature(index_files),
                )

        except FaissStoreError as error:
            index_error = str(error)

    st.subheader("Estado del proyecto")

    (
        document_column,
        page_column,
        chunk_column,
        vector_column,
        index_column,
    ) = st.columns(5)

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

    with vector_column:
        st.metric(
            label="Vectores",
            value=(
                int(index_snapshot["vector_count"])
                if index_snapshot
                else 0
            ),
        )

    with index_column:
        st.metric(
            label="Índice FAISS",
            value=(
                "Disponible"
                if index_snapshot
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
            "El procesamiento documental contiene validaciones "
            "que deben revisarse."
        )

    if not settings.google_api_key_configured:
        st.warning(
            "GOOGLE_API_KEY no está configurada. El índice existente "
            "puede cargarse localmente, pero la clave será necesaria "
            "para nuevas consultas y para regenerar embeddings."
        )
    else:
        st.info(
            "La clave de Gemini está configurada para generar "
            "embeddings de documentos y consultas."
        )

    if index_snapshot:
        st.success(
            "El índice FAISS está disponible y fue validado: "
            f"{index_snapshot['vector_count']} vectores de "
            f"{index_snapshot['embedding_dimension']} dimensiones."
        )
    elif index_error:
        st.error(
            "Se encontraron archivos del índice, pero no superaron "
            f"la validación: {index_error}"
        )
    else:
        st.warning(
            "El índice vectorial no está disponible. Ejecuta "
            "`python scripts/index_documents.py` para generarlo."
        )

    st.subheader("Resumen del procesamiento")

    (
        empty_column,
        ocr_column,
        category_column,
        size_column,
    ) = st.columns(4)

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

    with st.expander(
        "Ver detalle por documento",
        expanded=True,
    ):
        st.dataframe(
            processing["details"],
            use_container_width=True,
            hide_index=True,
        )

    validation_errors = processing["validation_errors"]

    with st.expander("Ver validaciones del procesamiento"):
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

    st.subheader("Índice vectorial")

    if index_snapshot:
        (
            model_column,
            dimension_column,
            type_column,
            metric_column,
        ) = st.columns(4)

        with model_column:
            st.metric(
                label="Modelo de embeddings",
                value=str(index_snapshot["embedding_model"]),
            )

        with dimension_column:
            st.metric(
                label="Dimensión",
                value=int(index_snapshot["embedding_dimension"]),
            )

        with type_column:
            st.metric(
                label="Tipo de índice",
                value=str(index_snapshot["index_type"]),
            )

        with metric_column:
            st.metric(
                label="Métrica",
                value=str(index_snapshot["distance_metric"]),
            )

        with st.expander("Ver manifiesto del índice"):
            st.json(
                index_snapshot["manifest"]
            )
    else:
        st.info(
            "El resumen vectorial aparecerá después de generar "
            "y validar el índice."
        )

    st.subheader("Consulta documental")

    st.text_input(
        label="Escribe una pregunta sobre BimBam Buy",
        placeholder="Ejemplo: ¿Cuánto tarda un reembolso?",
        disabled=True,
        help=(
            "El campo se habilitará cuando se implemente el servicio "
            "de recuperación semántica y la cadena RAG."
        ),
    )

    st.caption(
        "Siguiente hito: recuperar los fragmentos más relevantes "
        f"con k={settings.retrieval_k} y un umbral de "
        f"{settings.retrieval_score_threshold}."
    )


if __name__ == "__main__":
    main()
