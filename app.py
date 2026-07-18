"""Punto de entrada de la aplicación Streamlit de BimBam Assistant."""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st

from bimbam_assistant.application.indexing_service import (
    ChunkingError,
    create_chunks,
)
from bimbam_assistant.application.rag_service import (
    RagGenerationError,
    RetrievalError,
    answer_question,
)
from bimbam_assistant.core.config import (
    ConfigurationError,
    get_settings,
)
from bimbam_assistant.domain.models import RagResponse
from bimbam_assistant.infrastructure.document_change_detector import (
    DocumentChangeDetectionError,
    default_corpus_manifest_path,
    inspect_corpus_changes,
    load_corpus_manifest,
)
from bimbam_assistant.infrastructure.faiss_store import (
    FaissStoreError,
    load_vector_store,
)
from bimbam_assistant.infrastructure.monitoring_repository import (
    InteractionRecord,
    MonitoringRepositoryError,
    get_content_gap_questions,
    get_monitoring_database_path,
    get_quality_summary,
    get_recent_interactions,
    initialize_monitoring_database,
    save_interaction,
    update_interaction_feedback,
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

EVALUATION_BANK_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "evaluation"
    / "questions.json"
)

CATEGORY_LABELS = {
    "envios": "Envíos",
    "garantias": "Garantías",
    "reembolsos_devoluciones": "Reembolsos y devoluciones",
    "metodos_pago": "Métodos de pago",
    "afiliados": "Programa de afiliados",
}


@st.cache_data(show_spinner=False)
def build_processing_snapshot(
    documents_path: str,
    document_signature: tuple[tuple[str, int, int], ...],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    """Procesa el corpus y devuelve datos simples para la interfaz."""

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
    """Crea una firma para invalidar la caché al cambiar un archivo."""

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


def format_category(
    category: str,
) -> str:
    """Devuelve una etiqueta legible para una categoría técnica."""

    return CATEGORY_LABELS.get(
        category,
        category.replace("_", " ").title(),
    )



def render_global_styles() -> None:
    """Aplica ajustes visuales ligeros a la interfaz."""

    st.markdown(
        """
        <style>
        .compact-metric-card {
            padding: 0.55rem 0.7rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.65rem;
            min-height: 4.2rem;
            background: rgba(128, 128, 128, 0.035);
        }

        .compact-metric-label {
            margin: 0;
            font-size: 0.76rem;
            line-height: 1.15;
            opacity: 0.72;
        }

        .compact-metric-value {
            margin: 0.2rem 0 0;
            font-size: 1.35rem;
            line-height: 1.15;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        .assistant-welcome {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.8rem;
            background: rgba(128, 128, 128, 0.04);
        }

        .answer-focus-label {
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
            font-weight: 600;
            opacity: 0.75;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .process-card {
            height: 100%;
            min-height: 10.2rem;
            padding: 0.95rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.8rem;
            background: rgba(128, 128, 128, 0.035);
        }

        .process-step-number {
            width: 2rem;
            height: 2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.7rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            background: rgba(55, 125, 255, 0.13);
        }

        .process-step-title {
            margin: 0 0 0.4rem;
            font-size: 1rem;
            font-weight: 650;
        }

        .process-step-description {
            margin: 0;
            font-size: 0.84rem;
            line-height: 1.4;
            opacity: 0.78;
        }

        .process-step-status {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 650;
        }

        .process-step-ready {
            background: rgba(35, 160, 90, 0.14);
        }

        .process-step-pending {
            background: rgba(230, 150, 20, 0.16);
        }

        .section-introduction {
            margin-top: -0.35rem;
            margin-bottom: 1rem;
            opacity: 0.78;
        }

        div[data-testid="stChatMessage"] {
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compact_metric(
    *,
    label: str,
    value: object,
) -> None:
    """Muestra una métrica compacta para los paneles técnicos."""

    st.markdown(
        (
            '<div class="compact-metric-card">'
            f'<p class="compact-metric-label">{escape(label)}</p>'
            f'<p class="compact-metric-value">{escape(str(value))}</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )



def render_process_step(
    *,
    number: int,
    title: str,
    description: str,
    ready: bool,
) -> None:
    """Muestra una etapa del recorrido documental y conversacional."""

    status_label = (
        "Listo"
        if ready
        else "Pendiente"
    )

    status_class = (
        "process-step-ready"
        if ready
        else "process-step-pending"
    )

    st.markdown(
        (
            '<div class="process-card">'
            f'<div class="process-step-number">{number}</div>'
            f'<p class="process-step-title">{escape(title)}</p>'
            f'<p class="process-step-description">'
            f'{escape(description)}</p>'
            f'<span class="process-step-status {status_class}">'
            f'{status_label}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_process_overview(
    *,
    document_count: int,
    page_count: int,
    chunk_count: int,
    vector_count: int,
    processing_ready: bool,
    index_ready: bool,
    corpus_is_current: bool,
    api_key_configured: bool,
) -> None:
    """Explica de forma visual cómo se prepara y responde el asistente."""

    st.subheader(
        "Cómo funciona BimBam Assistant"
    )

    st.markdown(
        (
            '<p class="section-introduction">'
            "El sistema completa estas etapas antes de habilitar la "
            "conversación y repite la recuperación, generación y "
            "verificación para cada pregunta."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    chat_ready = bool(
        processing_ready
        and index_ready
        and corpus_is_current
        and api_key_configured
    )

    steps = [
        {
            "title": "Carga documental",
            "description": (
                f"Localiza {document_count} documentos PDF y extrae "
                f"{page_count} páginas con sus metadatos."
            ),
            "ready": document_count > 0,
        },
        {
            "title": "Limpieza y fragmentación",
            "description": (
                f"Limpia el texto y genera {chunk_count} fragmentos "
                "trazables sin mezclar páginas."
            ),
            "ready": processing_ready,
        },
        {
            "title": "Embeddings e índice",
            "description": (
                f"Representa los fragmentos como vectores y mantiene "
                f"{vector_count} registros en FAISS."
            ),
            "ready": index_ready,
        },
        {
            "title": "Control de vigencia",
            "description": (
                "Compara firmas SHA-256 y evita responder cuando el "
                "corpus cambió después de la indexación."
            ),
            "ready": corpus_is_current,
        },
        {
            "title": "Respuesta RAG segura",
            "description": (
                "Recupera las fuentes más relevantes, genera la "
                "respuesta y verifica citas y respaldo documental."
            ),
            "ready": bool(
                index_ready
                and corpus_is_current
                and api_key_configured
            ),
        },
        {
            "title": "Conversación trazable",
            "description": (
                "Presenta la respuesta, sus fuentes, la verificación "
                "y el feedback de manera independiente por pregunta."
            ),
            "ready": chat_ready,
        },
    ]

    first_row = st.columns(
        3
    )

    for index, column in enumerate(
        first_row,
        start=0,
    ):
        with column:
            render_process_step(
                number=index + 1,
                **steps[
                    index
                ],
            )

    second_row = st.columns(
        3
    )

    for index, column in enumerate(
        second_row,
        start=3,
    ):
        with column:
            render_process_step(
                number=index + 1,
                **steps[
                    index
                ],
            )


def build_corpus_sync_snapshot(
    *,
    documents_path: Path,
    faiss_index_path: Path,
    index_exists: bool,
) -> dict[str, object]:
    """Compara el corpus actual con el manifiesto de la última indexación."""

    corpus_manifest_path = default_corpus_manifest_path(
        faiss_index_path
    )

    stored_manifest = load_corpus_manifest(
        corpus_manifest_path
    )

    (
        current_manifest,
        changes,
    ) = inspect_corpus_changes(
        documents_path=documents_path,
        manifest_path=corpus_manifest_path,
    )

    synchronized = bool(
        index_exists
        and changes.previous_manifest_exists
        and not changes.has_changes
    )

    return {
        "synchronized": synchronized,
        "manifest_path": str(
            corpus_manifest_path
        ),
        "manifest_exists": changes.previous_manifest_exists,
        "indexed_at_utc": (
            stored_manifest.get(
                "created_at_utc"
            )
            if stored_manifest
            else None
        ),
        "document_count": int(
            current_manifest.get(
                "document_count",
                0,
            )
        ),
        "added": list(
            changes.added
        ),
        "modified": list(
            changes.modified
        ),
        "deleted": list(
            changes.deleted
        ),
        "unchanged": list(
            changes.unchanged
        ),
        "changed_count": changes.changed_count,
    }


def render_corpus_sync_status(
    sync_snapshot: dict[str, object] | None,
    *,
    sync_error: str | None,
) -> None:
    """Muestra si FAISS representa el corpus documental actual."""

    st.subheader(
        "Sincronización documental"
    )

    if sync_error:
        st.error(
            "No fue posible comparar el corpus con el manifiesto "
            f"de indexación: {sync_error}"
        )
        return

    if sync_snapshot is None:
        st.warning(
            "No fue posible determinar el estado de sincronización."
        )
        return

    (
        status_column,
        added_column,
        modified_column,
        deleted_column,
        unchanged_column,
    ) = st.columns(5)

    with status_column:
        render_compact_metric(
            label="Estado",
            value=(
                "Actualizado"
                if sync_snapshot["synchronized"]
                else "Requiere indexación"
            ),
        )

    with added_column:
        render_compact_metric(
            label="Agregados",
            value=len(
                sync_snapshot["added"]
            ),
        )

    with modified_column:
        render_compact_metric(
            label="Modificados",
            value=len(
                sync_snapshot["modified"]
            ),
        )

    with deleted_column:
        render_compact_metric(
            label="Eliminados",
            value=len(
                sync_snapshot["deleted"]
            ),
        )

    with unchanged_column:
        render_compact_metric(
            label="Sin cambios",
            value=len(
                sync_snapshot["unchanged"]
            ),
        )

    if sync_snapshot["synchronized"]:
        st.success(
            "El índice FAISS representa el estado actual de los "
            "documentos. El chat está habilitado."
        )
    else:
        if not sync_snapshot["manifest_exists"]:
            st.warning(
                "No existe un manifiesto del corpus. Ejecuta "
                "`python scripts/index_documents.py` para sincronizar "
                "el índice antes de consultar el asistente."
            )
        else:
            st.warning(
                "El corpus cambió después de la última indexación. "
                "El chat permanecerá deshabilitado para evitar respuestas "
                "basadas en documentos desactualizados. Ejecuta "
                "`python scripts/index_documents.py`."
            )

    indexed_at_utc = sync_snapshot.get(
        "indexed_at_utc"
    )

    if indexed_at_utc:
        st.caption(
            "Última firma guardada del corpus: "
            f"`{indexed_at_utc}`"
        )

    changed_documents = [
        (
            "Agregado",
            document,
        )
        for document in sync_snapshot["added"]
    ] + [
        (
            "Modificado",
            document,
        )
        for document in sync_snapshot["modified"]
    ] + [
        (
            "Eliminado",
            document,
        )
        for document in sync_snapshot["deleted"]
    ]

    if changed_documents:
        with st.expander(
            "Ver cambios detectados",
            expanded=True,
        ):
            st.dataframe(
                [
                    {
                        "Cambio": change_type,
                        "Documento": document,
                    }
                    for change_type, document in changed_documents
                ],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        "Ver manifiesto de sincronización"
    ):
        st.write(
            {
                "Ruta": sync_snapshot[
                    "manifest_path"
                ],
                "Existe": sync_snapshot[
                    "manifest_exists"
                ],
                "Documentos actuales": sync_snapshot[
                    "document_count"
                ],
                "Cambios totales": sync_snapshot[
                    "changed_count"
                ],
            }
        )


@st.cache_data(show_spinner=False)
def build_evaluation_bank_snapshot(
    bank_path: str,
    modified_at_ns: int,
) -> dict[str, object]:
    """Resume el banco de evaluación sin realizar llamadas a Gemini."""

    del modified_at_ns

    path = Path(
        bank_path
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    questions = payload.get(
        "questions",
        [],
    )

    if not isinstance(
        questions,
        list,
    ):
        raise ValueError(
            "El campo 'questions' debe ser una lista."
        )

    category_counts = Counter(
        str(
            question.get(
                "category",
                "sin_categoria",
            )
        )
        for question in questions
        if isinstance(
            question,
            dict,
        )
    )

    tier_counts = Counter(
        str(
            question.get(
                "evaluation_tier",
                "sin_nivel",
            )
        )
        for question in questions
        if isinstance(
            question,
            dict,
        )
    )

    batch_counts = Counter(
        str(
            question.get(
                "budget_batch",
                "sin_lote",
            )
        )
        for question in questions
        if isinstance(
            question,
            dict,
        )
    )

    fallback_count = sum(
        isinstance(
            question,
            dict,
        )
        and question.get(
            "expected_behavior"
        )
        == "fallback"
        for question in questions
    )

    policy = payload.get(
        "evaluation_policy",
        {},
    )

    return {
        "name": payload.get(
            "name",
            "Banco de evaluación RAG",
        ),
        "question_count": len(
            questions
        ),
        "category_counts": dict(
            category_counts
        ),
        "tier_counts": dict(
            tier_counts
        ),
        "batch_counts": dict(
            batch_counts
        ),
        "batch_count": len(
            batch_counts
        ),
        "fallback_count": fallback_count,
        "daily_limit": policy.get(
            "daily_gemini_call_limit",
            20,
        ),
        "recommended_questions_per_day": policy.get(
            "recommended_max_generated_questions_per_day",
            4,
        ),
    }


def render_evaluation_bank_status() -> None:
    """Muestra el estado del banco y del evaluador sin consumir API."""

    st.subheader(
        "Evaluación RAG"
    )

    if not EVALUATION_BANK_PATH.is_file():
        st.warning(
            "El banco de evaluación todavía no está disponible en "
            "`data/evaluation/questions.json`."
        )
        return

    try:
        snapshot = build_evaluation_bank_snapshot(
            str(
                EVALUATION_BANK_PATH
            ),
            EVALUATION_BANK_PATH.stat().st_mtime_ns,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        st.error(
            "No fue posible leer el banco de evaluación: "
            f"{error}"
        )
        return

    (
        questions_column,
        batches_column,
        fallback_column,
        budget_column,
    ) = st.columns(
        4
    )

    with questions_column:
        render_compact_metric(
            label="Preguntas",
            value=snapshot[
                "question_count"
            ],
        )

    with batches_column:
        render_compact_metric(
            label="Lotes",
            value=snapshot[
                "batch_count"
            ],
        )

    with fallback_column:
        render_compact_metric(
            label="Casos fallback",
            value=snapshot[
                "fallback_count"
            ],
        )

    with budget_column:
        render_compact_metric(
            label="Máximo diario",
            value=(
                f"{snapshot['recommended_questions_per_day']} "
                "preguntas"
            ),
        )

    st.success(
        "El banco y el ejecutor controlado están preparados. "
        "Las evaluaciones reales con Gemini no se han ejecutado "
        "para conservar la cuota diaria."
    )

    st.caption(
        "La validación del JSON, las pruebas offline y las simulaciones "
        "del presupuesto no consumen llamadas de Gemini."
    )

    with st.expander(
        "Ver cobertura del banco"
    ):
        st.write(
            {
                "Nombre": snapshot[
                    "name"
                ],
                "Categorías": snapshot[
                    "category_counts"
                ],
                "Niveles": snapshot[
                    "tier_counts"
                ],
                "Lotes": snapshot[
                    "batch_counts"
                ],
                "Límite diario configurado": snapshot[
                    "daily_limit"
                ],
            }
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
        st.write(f"**Top k:** {settings.retrieval_k}")
        st.write(
            f"**Umbral:** {settings.retrieval_score_threshold}"
        )

        st.divider()

        st.write(
            f"**Modelo de chat:** {settings.gemini_chat_model}"
        )
        st.write(
            f"**Modelo de embeddings:** "
            f"{settings.gemini_embedding_model}"
        )

        st.divider()

        st.caption(
            "La clave de Gemini nunca se muestra en la interfaz."
        )


def build_source_rows(
    response: RagResponse,
) -> list[dict[str, object]]:
    """Construye el resumen tabular de las fuentes recuperadas."""

    rows: list[dict[str, object]] = []

    for source in response.sources:
        metadata = source.metadata

        rows.append(
            {
                "Fuente": source.rank,
                "Documento": metadata.get(
                    "document_name",
                    "Documento desconocido",
                ),
                "Página": metadata.get(
                    "page_number",
                    "N/D",
                ),
                "Categoría": format_category(
                    str(
                        metadata.get(
                            "category",
                            "sin_clasificar",
                        )
                    )
                ),
                "Similitud": round(
                    source.score,
                    4,
                ),
            }
        )

    return rows



MAX_CONVERSATION_CONTEXT_TURNS = 3


def initialize_conversation_state() -> None:
    """Inicializa el historial y elimina estados incompatibles anteriores."""

    st.session_state.setdefault(
        "conversation_turns",
        [],
    )

    st.session_state.setdefault(
        "monitoring_session_id",
        uuid4().hex,
    )

    st.session_state.pop(
        "verified_rag_response",
        None,
    )


def clear_conversation() -> None:
    """Elimina el historial de la sesión actual."""

    st.session_state["conversation_turns"] = []


def build_contextual_query(
    current_query: str,
    turns: list[dict[str, object]],
) -> str:
    """Añade las preguntas recientes para resolver seguimientos breves.

    Los mensajes anteriores ayudan a recuperar documentos para preguntas
    como "¿y qué documentos necesito?". Las respuestas previas no se usan
    como fuente: la evidencia autorizada continúa siendo el corpus.
    """

    previous_questions = [
        str(turn.get("question", "")).strip()
        for turn in turns[-MAX_CONVERSATION_CONTEXT_TURNS:]
        if str(turn.get("question", "")).strip()
    ]

    if not previous_questions:
        return current_query

    history = "\n".join(
        f"- {question}"
        for question in previous_questions
    )

    return (
        "Pregunta actual:\n"
        f"{current_query}\n\n"
        "Preguntas recientes de esta conversación, utilizadas solo "
        "para interpretar referencias o seguimientos:\n"
        f"{history}"
    )


def update_turn_feedback(
    turn_id: str,
    rating: str,
) -> None:
    """Registra feedback en la sesión y en SQLite."""

    turns = list(
        st.session_state.get(
            "conversation_turns",
            [],
        )
    )

    for turn in turns:
        if turn.get("id") == turn_id:
            turn["feedback"] = rating
            break

    st.session_state["conversation_turns"] = turns

    try:
        update_interaction_feedback(
            turn_id,
            rating,
        )

        st.session_state.pop(
            "monitoring_warning",
            None,
        )

    except MonitoringRepositoryError as error:
        st.session_state[
            "monitoring_warning"
        ] = str(error)


def get_feedback_counts() -> tuple[int, int]:
    """Cuenta las valoraciones registradas en la sesión."""

    turns = st.session_state.get(
        "conversation_turns",
        [],
    )

    positive = sum(
        turn.get("feedback") == "positive"
        for turn in turns
    )

    negative = sum(
        turn.get("feedback") == "negative"
        for turn in turns
    )

    return positive, negative


def render_feedback_controls(
    *,
    turn_id: str,
    current_feedback: str | None,
) -> None:
    """Muestra los botones de valoración asociados a una respuesta."""

    st.markdown("#### ¿Esta respuesta fue útil?")

    positive_column, negative_column, status_column = st.columns(
        [1, 1, 3]
    )

    with positive_column:
        positive_clicked = st.button(
            "👍 Útil",
            key=f"positive_feedback_{turn_id}",
            type=(
                "primary"
                if current_feedback == "positive"
                else "secondary"
            ),
            use_container_width=True,
        )

    with negative_column:
        negative_clicked = st.button(
            "👎 No útil",
            key=f"negative_feedback_{turn_id}",
            type=(
                "primary"
                if current_feedback == "negative"
                else "secondary"
            ),
            use_container_width=True,
        )

    with status_column:
        if current_feedback == "positive":
            st.success(
                "Retroalimentación positiva registrada en esta sesión."
            )
        elif current_feedback == "negative":
            st.warning(
                "Retroalimentación negativa registrada en esta sesión."
            )
        else:
            st.caption(
                "La valoración se conserva mientras esta sesión siga activa."
            )

    if positive_clicked:
        update_turn_feedback(
            turn_id,
            "positive",
        )
        st.rerun()

    if negative_clicked:
        update_turn_feedback(
            turn_id,
            "negative",
        )
        st.rerun()


def render_conversation_history() -> None:
    """Presenta todas las preguntas y respuestas de la sesión."""

    turns = st.session_state.get(
        "conversation_turns",
        [],
    )

    if not turns:
        with st.chat_message("assistant"):
            st.markdown(
                """
                <div class="assistant-welcome">
                <strong>¡Hola! Soy BimBam Assistant.</strong><br><br>
                Puedo ayudarte a consultar las políticas y documentos
                corporativos de BimBam Buy. Por ejemplo:
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                - ¿Cuánto tarda un reembolso?
                - ¿Qué evidencia necesito para solicitar una garantía?
                - ¿Cuáles son los tiempos estimados de envío?
                - ¿Qué hago si mi pago fue rechazado?
                - ¿Cómo funciona el programa de afiliados?
                """
            )

            st.caption(
                "Las respuestas son generadas por inteligencia artificial "
                "y se acompañan de sus fuentes documentales."
            )

        return

    for turn in turns:
        question = str(
            turn.get(
                "question",
                "",
            )
        )

        category = str(
            turn.get(
                "category",
                "todas",
            )
        )

        with st.chat_message("user"):
            st.markdown(
                question
            )

            st.caption(
                "Categoría: "
                + (
                    "Todas las categorías"
                    if category == "todas"
                    else format_category(category)
                )
            )

        response_data = turn.get(
            "response"
        )

        if not isinstance(response_data, dict):
            continue

        response = RagResponse.model_validate(
            response_data
        )

        with st.chat_message("assistant"):
            render_rag_response(
                response,
                turn_id=str(turn["id"]),
                current_feedback=(
                    str(turn["feedback"])
                    if turn.get("feedback")
                    else None
                ),
            )


def render_rag_response(
    response: RagResponse,
    *,
    turn_id: str,
    current_feedback: str | None,
) -> None:
    """Muestra la respuesta, sus fuentes y los detalles de verificación."""

    verification = response.verification

    status_messages = {
        "verified": (
            "✅ Respuesta verificada automáticamente contra "
            "el contexto documental."
        ),
        "rejected": (
            "⚠️ La respuesta generada fue rechazada y sustituida "
            "por un mensaje seguro."
        ),
        "not_applicable": (
            "ℹ️ No se generó una respuesta con el modelo porque "
            "no se encontró evidencia suficiente."
        ),
    }

    with st.container(
        border=True
    ):
        st.markdown(
            (
                '<div class="answer-focus-label">'
                "Respuesta de BimBam Assistant"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            response.answer
        )

        st.caption(
            status_messages.get(
                verification.status,
                verification.status,
            )
        )

    st.markdown(
        "#### Fuentes utilizadas en esta respuesta"
    )

    if response.has_sources:
        cited_sources = (
            ", ".join(
                f"[Fuente {source_number}]"
                for source_number
                in verification.cited_sources
            )
            if verification.cited_sources
            else "Ninguna cita explícita"
        )

        st.caption(
            "Las referencias incluidas en la respuesta corresponden "
            f"a esta tabla. Fuentes citadas: {cited_sources}."
        )

        st.dataframe(
            build_source_rows(
                response
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Consultar los fragmentos recuperados"
        ):
            for source_position, source in enumerate(
                response.sources,
                start=1,
            ):
                metadata = source.metadata

                document_name = str(
                    metadata.get(
                        "document_name",
                        "Documento desconocido",
                    )
                )

                page_number = metadata.get(
                    "page_number",
                    "N/D",
                )

                category = format_category(
                    str(
                        metadata.get(
                            "category",
                            "sin_clasificar",
                        )
                    )
                )

                with st.container(
                    border=True
                ):
                    st.markdown(
                        (
                            f"**Fuente {source.rank}: "
                            f"{document_name}**"
                        )
                    )

                    st.caption(
                        f"Página {page_number} · "
                        f"Categoría: {category} · "
                        f"Similitud: {source.score:.4f}"
                    )

                    st.markdown(
                        source.page_content
                    )

                    chunk_id = str(
                        metadata.get(
                            "chunk_id",
                            f"vector-{source.vector_id}",
                        )
                    )

                    st.caption(
                        f"Chunk: {chunk_id}"
                    )

                if source_position < len(
                    response.sources
                ):
                    st.write("")
    else:
        st.info(
            "Esta respuesta no utilizó fuentes porque no se encontró "
            "evidencia documental suficiente."
        )

    render_feedback_controls(
        turn_id=turn_id,
        current_feedback=current_feedback,
    )

    with st.expander(
        "Verificación y detalles técnicos",
        expanded=verification.status == "rejected",
    ):
        (
            source_column,
            status_column,
            confidence_column,
            context_column,
        ) = st.columns(
            4
        )

        with source_column:
            st.metric(
                label="Fuentes recuperadas",
                value=len(
                    response.sources
                ),
            )

        with status_column:
            status_labels = {
                "verified": "Verificada",
                "rejected": "Rechazada",
                "not_applicable": "No aplicable",
            }

            st.metric(
                label="Verificación",
                value=status_labels.get(
                    verification.status,
                    verification.status,
                ),
            )

        with confidence_column:
            st.metric(
                label="Confianza",
                value=f"{verification.confidence:.0%}",
            )

        with context_column:
            st.metric(
                label="Usó contexto",
                value=(
                    "Sí"
                    if response.used_context
                    else "No"
                ),
            )

        st.caption(
            f"Modelo generativo: {response.model_name}"
        )

        if response.support_contact is not None:
            st.info(
                "**Contacto alternativo de demostración:** "
                f"{response.support_contact.area} — "
                f"`{response.support_contact.email}`\n\n"
                "Este contacto es ficticio y no proviene del corpus "
                "documental."
            )

        st.write(
            {
                "Estado": verification.status,
                "Superó la verificación": verification.passed,
                "Contenido respaldado": (
                    verification.semantic_supported
                ),
                "Confianza": verification.confidence,
                "Citas presentes": verification.citations_present,
                "Fuentes citadas": verification.cited_sources,
                "Citas inválidas": verification.invalid_citations,
            }
        )

        st.write(
            f"**Explicación:** {verification.explanation}"
        )

        if verification.unsupported_claims:
            st.write(
                "**Afirmaciones no respaldadas:**"
            )

            for claim in verification.unsupported_claims:
                st.write(
                    f"• {claim}"
                )
        else:
            st.caption(
                "No se detectaron afirmaciones sin respaldo."
            )

        with st.expander(
            "Contexto completo enviado a generación y verificación"
        ):
            st.code(
                response.retrieval.context,
                language="text",
            )



def determine_interaction_outcome(
    response: RagResponse,
) -> str:
    """Clasifica el resultado para las métricas de calidad."""

    if not response.used_context:
        return "no_evidence"

    if response.verification.status == "rejected":
        return "rejected"

    return "answered"


def build_persisted_sources(
    response: RagResponse,
) -> list[dict[str, object]]:
    """Reduce las fuentes a los campos útiles para auditoría."""

    return [
        {
            "rank": source.rank,
            "document_name": source.metadata.get(
                "document_name",
                "Documento desconocido",
            ),
            "page_number": source.metadata.get(
                "page_number",
            ),
            "category": source.metadata.get(
                "category",
            ),
            "score": source.score,
        }
        for source in response.sources
    ]


def persist_successful_interaction(
    *,
    interaction_id: str,
    question: str,
    contextual_query: str,
    category: str,
    response: RagResponse,
    latency_ms: int,
) -> None:
    """Guarda una respuesta final y sus métricas."""

    save_interaction(
        InteractionRecord(
            interaction_id=interaction_id,
            session_id=str(
                st.session_state["monitoring_session_id"]
            ),
            question=question,
            contextual_query=contextual_query,
            category=category,
            answer=response.answer,
            outcome=determine_interaction_outcome(
                response
            ),
            verification_status=(
                response.verification.status
            ),
            verification_confidence=(
                response.verification.confidence
            ),
            used_context=response.used_context,
            source_count=len(
                response.sources
            ),
            model_name=response.model_name,
            latency_ms=latency_ms,
            sources=build_persisted_sources(
                response
            ),
        )
    )


def persist_failed_interaction(
    *,
    interaction_id: str,
    question: str,
    contextual_query: str,
    category: str,
    latency_ms: int,
    error: Exception,
) -> None:
    """Registra un fallo para incorporarlo al monitoreo."""

    settings = get_settings()

    save_interaction(
        InteractionRecord(
            interaction_id=interaction_id,
            session_id=str(
                st.session_state["monitoring_session_id"]
            ),
            question=question,
            contextual_query=contextual_query,
            category=category,
            answer="",
            outcome="error",
            verification_status="error",
            verification_confidence=0.0,
            used_context=False,
            source_count=0,
            model_name=settings.gemini_chat_model,
            latency_ms=latency_ms,
            sources=[],
            error_message=str(error),
        )
    )


def render_quality_monitoring() -> None:
    """Muestra métricas persistentes sin competir con el chat."""

    with st.expander(
        "Monitoreo de calidad",
        expanded=False,
    ):
        try:
            summary = get_quality_summary()

            (
                total_column,
                unanswered_column,
                rejected_column,
                latency_column,
            ) = st.columns(4)

            with total_column:
                render_compact_metric(
                    label="Consultas registradas",
                    value=summary[
                        "total_interactions"
                    ],
                )

            with unanswered_column:
                render_compact_metric(
                    label="Sin evidencia",
                    value=summary[
                        "no_evidence"
                    ],
                )

            with rejected_column:
                render_compact_metric(
                    label="Respuestas rechazadas",
                    value=summary[
                        "rejected"
                    ],
                )

            with latency_column:
                render_compact_metric(
                    label="Latencia promedio",
                    value=(
                        f"{summary['average_latency_ms'] / 1000:.2f} s"
                    ),
                )

            (
                positive_column,
                negative_column,
                feedback_rate_column,
                error_column,
            ) = st.columns(4)

            with positive_column:
                render_compact_metric(
                    label="Feedback positivo",
                    value=summary[
                        "positive_feedback"
                    ],
                )

            with negative_column:
                render_compact_metric(
                    label="Feedback negativo",
                    value=summary[
                        "negative_feedback"
                    ],
                )

            with feedback_rate_column:
                render_compact_metric(
                    label="Tasa de feedback",
                    value=(
                        f"{summary['feedback_rate']:.0%}"
                    ),
                )

            with error_column:
                render_compact_metric(
                    label="Errores registrados",
                    value=summary[
                        "errors"
                    ],
                )

            st.markdown(
                "#### Interacciones recientes"
            )

            recent_interactions = get_recent_interactions(
                limit=15
            )

            if recent_interactions:
                st.dataframe(
                    recent_interactions,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(
                    "Todavía no hay interacciones persistidas."
                )

            st.markdown(
                "#### Posibles vacíos de conocimiento"
            )

            content_gaps = get_content_gap_questions(
                limit=15
            )

            if content_gaps:
                st.dataframe(
                    content_gaps,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(
                    "No se han registrado preguntas sin evidencia, "
                    "rechazos, errores o feedback negativo."
                )

            st.caption(
                "Base local: "
                f"`{get_monitoring_database_path()}`"
            )

        except MonitoringRepositoryError as error:
            st.warning(
                "El monitoreo persistente no está disponible: "
                f"{error}"
            )

def render_rag_section(
    *,
    index_snapshot: dict[str, object] | None,
    corpus_is_current: bool,
) -> None:
    """Renderiza el chat, el historial y las respuestas RAG."""

    settings = get_settings()

    initialize_conversation_state()

    st.subheader("Conversa con BimBam Assistant")

    st.markdown(
        (
            '<p class="section-introduction">'
            "Formula una pregunta sobre las políticas de BimBam Buy. "
            "Cada respuesta conservará sus propias fuentes y su "
            "resultado de verificación."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    monitoring_warning = st.session_state.pop(
        "monitoring_warning",
        None,
    )

    if monitoring_warning:
        st.warning(
            "La respuesta sigue disponible, pero no se pudo "
            f"persistir una métrica: {monitoring_warning}"
        )

    st.info(
        "Estás conversando con un asistente de inteligencia artificial, "
        "no con una persona. Verifica la respuesta en las fuentes "
        "documentales mostradas."
    )

    query_ready = bool(
        index_snapshot
        and settings.google_api_key_configured
        and corpus_is_current
    )

    available_categories = (
        sorted(
            str(category)
            for category in index_snapshot.get(
                "categories",
                [],
            )
        )
        if index_snapshot
        else []
    )

    category_options = [
        "todas",
        *available_categories,
    ]

    control_column, clear_column = st.columns(
        [4, 1]
    )

    with control_column:
        selected_category = st.selectbox(
            label="Categoría documental",
            options=category_options,
            format_func=(
                lambda value: (
                    "Todas las categorías"
                    if value == "todas"
                    else format_category(value)
                )
            ),
            disabled=not query_ready,
            help=(
                "El filtro restringe la recuperación a una categoría."
            ),
            key="chat_category",
        )

    with clear_column:
        st.write("")
        st.write("")

        clear_clicked = st.button(
            "Nueva conversación",
            use_container_width=True,
            disabled=not bool(
                st.session_state.get(
                    "conversation_turns",
                    [],
                )
            ),
        )

    if clear_clicked:
        clear_conversation()
        st.rerun()

    positive_count, negative_count = get_feedback_counts()

    history_column, positive_column, negative_column = st.columns(3)

    with history_column:
        st.metric(
            label="Turnos en la sesión",
            value=len(
                st.session_state.get(
                    "conversation_turns",
                    [],
                )
            ),
        )

    with positive_column:
        st.metric(
            label="Respuestas útiles",
            value=positive_count,
        )

    with negative_column:
        st.metric(
            label="Respuestas no útiles",
            value=negative_count,
        )

    render_conversation_history()

    if not query_ready:
        if not index_snapshot:
            st.warning(
                "Genera y valida el índice FAISS antes de realizar "
                "consultas."
            )
        elif not corpus_is_current:
            st.warning(
                "El corpus y el índice no están sincronizados. "
                "Ejecuta `python scripts/index_documents.py` antes "
                "de realizar consultas."
            )
        elif not settings.google_api_key_configured:
            st.warning(
                "Configura GOOGLE_API_KEY para generar embeddings, "
                "respuestas y verificaciones."
            )

    prompt = st.chat_input(
        "Escribe una pregunta sobre BimBam Buy",
        disabled=not query_ready,
    )

    if prompt:
        turns = list(
            st.session_state.get(
                "conversation_turns",
                [],
            )
        )

        contextual_query = build_contextual_query(
            prompt,
            turns,
        )

        filters = (
            {}
            if selected_category == "todas"
            else {
                "category": selected_category,
            }
        )

        interaction_id = uuid4().hex
        started_at = perf_counter()

        try:
            with st.spinner(
                "Recuperando evidencia, generando y verificando "
                "la respuesta..."
            ):
                response = answer_question(
                    contextual_query,
                    filters=filters,
                )

            latency_ms = round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000
            )

            try:
                persist_successful_interaction(
                    interaction_id=interaction_id,
                    question=prompt,
                    contextual_query=contextual_query,
                    category=selected_category,
                    response=response,
                    latency_ms=latency_ms,
                )

            except MonitoringRepositoryError as error:
                st.session_state[
                    "monitoring_warning"
                ] = str(error)

            turns.append(
                {
                    "id": interaction_id,
                    "question": prompt,
                    "category": selected_category,
                    "response": response.model_dump(),
                    "feedback": None,
                }
            )

            st.session_state[
                "conversation_turns"
            ] = turns

            st.rerun()

        except (
            RetrievalError,
            RagGenerationError,
        ) as error:
            latency_ms = round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000
            )

            try:
                persist_failed_interaction(
                    interaction_id=interaction_id,
                    question=prompt,
                    contextual_query=contextual_query,
                    category=selected_category,
                    latency_ms=latency_ms,
                    error=error,
                )

            except MonitoringRepositoryError as monitoring_error:
                st.session_state[
                    "monitoring_warning"
                ] = str(
                    monitoring_error
                )

            st.error(
                "No fue posible completar la consulta: "
                f"{error}"
            )

    st.caption(
        "El historial y la retroalimentación se conservan durante "
        "la sesión actual. Las preguntas recientes ayudan a interpretar "
        "seguimientos breves, pero las respuestas continúan sustentándose "
        "exclusivamente en los documentos recuperados."
    )


def main() -> None:
    """Renderiza la aplicación."""

    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.error(
            f"Error de configuración: {error}"
        )
        return

    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_global_styles()

    try:
        initialize_monitoring_database()
    except MonitoringRepositoryError as error:
        st.session_state[
            "monitoring_warning"
        ] = str(error)

    st.title(
        f"🛍️ {settings.app_name}"
    )

    st.caption(
        "Asistente de inteligencia artificial para consultar las "
        "políticas y los documentos corporativos de BimBam Buy."
    )

    st.markdown(
        """
        El asistente prepara y valida el corpus documental antes de
        habilitar la conversación. Cada consulta recupera evidencia,
        genera una respuesta y verifica automáticamente sus citas antes
        de presentarla.
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

    corpus_sync_snapshot: dict[str, object] | None = None
    corpus_sync_error: str | None = None

    try:
        corpus_sync_snapshot = build_corpus_sync_snapshot(
            documents_path=settings.documents_path,
            faiss_index_path=settings.faiss_index_path,
            index_exists=bool(index_snapshot),
        )
    except DocumentChangeDetectionError as error:
        corpus_sync_error = str(error)

    st.subheader(
        "Estado general"
    )

    (
        document_column,
        page_column,
        chunk_column,
        vector_column,
        index_column,
    ) = st.columns(
        5
    )

    with document_column:
        render_compact_metric(
            label="Documentos PDF",
            value=len(
                pdf_files
            ),
        )

    with page_column:
        render_compact_metric(
            label="Páginas procesadas",
            value=int(
                processing[
                    "page_count"
                ]
            ),
        )

    with chunk_column:
        render_compact_metric(
            label="Chunks generados",
            value=int(
                processing[
                    "chunk_count"
                ]
            ),
        )

    with vector_column:
        render_compact_metric(
            label="Vectores",
            value=(
                int(
                    index_snapshot[
                        "vector_count"
                    ]
                )
                if index_snapshot
                else 0
            ),
        )

    with index_column:
        render_compact_metric(
            label="Índice FAISS",
            value=(
                "Disponible"
                if index_snapshot
                else "Pendiente"
            ),
        )

    processing_ready = bool(
        processing[
            "processing_ready"
        ]
    )

    corpus_is_current = bool(
        corpus_sync_snapshot
        and corpus_sync_snapshot[
            "synchronized"
        ]
    )

    if processing_ready:
        st.success(
            "El corpus fue leído, limpiado, clasificado y fragmentado "
            "correctamente."
        )
    else:
        st.warning(
            "El procesamiento documental contiene validaciones "
            "que deben revisarse."
        )

    if settings.google_api_key_configured:
        st.success(
            "La clave de Gemini está configurada para generar embeddings "
            "de consulta, respuestas y verificaciones estructuradas."
        )
    else:
        st.warning(
            "GOOGLE_API_KEY no está configurada. El índice existente "
            "puede cargarse localmente, pero la clave es necesaria "
            "para consultar Gemini."
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

    render_process_overview(
        document_count=len(
            pdf_files
        ),
        page_count=int(
            processing[
                "page_count"
            ]
        ),
        chunk_count=int(
            processing[
                "chunk_count"
            ]
        ),
        vector_count=(
            int(
                index_snapshot[
                    "vector_count"
                ]
            )
            if index_snapshot
            else 0
        ),
        processing_ready=processing_ready,
        index_ready=bool(
            index_snapshot
        ),
        corpus_is_current=corpus_is_current,
        api_key_configured=settings.google_api_key_configured,
    )

    st.subheader(
        "Información técnica y control de calidad"
    )

    st.markdown(
        (
            '<p class="section-introduction">'
            "Los detalles operativos se organizan en pestañas para "
            "mantener la lectura principal compacta."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    (
        synchronization_tab,
        processing_tab,
        index_tab,
        monitoring_tab,
        evaluation_tab,
    ) = st.tabs(
        [
            "Sincronización",
            "Procesamiento",
            "Índice vectorial",
            "Monitoreo",
            "Evaluación RAG",
        ]
    )

    with synchronization_tab:
        render_corpus_sync_status(
            corpus_sync_snapshot,
            sync_error=corpus_sync_error,
        )

    with processing_tab:
        st.markdown(
            "### Resumen del procesamiento"
        )

        (
            empty_column,
            ocr_column,
            category_column,
            size_column,
        ) = st.columns(
            4
        )

        with empty_column:
            render_compact_metric(
                label="Páginas vacías",
                value=int(
                    processing[
                        "empty_pages"
                    ]
                ),
            )

        with ocr_column:
            render_compact_metric(
                label="Candidatas a OCR",
                value=int(
                    processing[
                        "ocr_candidates"
                    ]
                ),
            )

        with category_column:
            render_compact_metric(
                label="Categorías",
                value=int(
                    processing[
                        "category_count"
                    ]
                ),
            )

        with size_column:
            render_compact_metric(
                label="Chunk máximo",
                value=(
                    f"{int(processing['maximum_chunk_size'])} "
                    "caracteres"
                ),
            )

        st.markdown(
            "#### Documentos procesados"
        )

        st.dataframe(
            processing[
                "details"
            ],
            use_container_width=True,
            hide_index=True,
        )

        validation_errors = processing[
            "validation_errors"
        ]

        with st.expander(
            "Ver validaciones del procesamiento"
        ):
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

        with st.expander(
            "Ver archivos PDF detectados"
        ):
            for pdf_file in pdf_files:
                st.write(
                    f"• {pdf_file.name}"
                )

    with index_tab:
        st.markdown(
            "### Índice vectorial"
        )

        if index_snapshot:
            (
                model_column,
                dimension_column,
                type_column,
                metric_column,
            ) = st.columns(
                4
            )

            with model_column:
                render_compact_metric(
                    label="Modelo de embeddings",
                    value=str(
                        index_snapshot[
                            "embedding_model"
                        ]
                    ),
                )

            with dimension_column:
                render_compact_metric(
                    label="Dimensión",
                    value=int(
                        index_snapshot[
                            "embedding_dimension"
                        ]
                    ),
                )

            with type_column:
                render_compact_metric(
                    label="Tipo de índice",
                    value=str(
                        index_snapshot[
                            "index_type"
                        ]
                    ),
                )

            with metric_column:
                render_compact_metric(
                    label="Métrica",
                    value=str(
                        index_snapshot[
                            "distance_metric"
                        ]
                    ),
                )

            with st.expander(
                "Ver manifiesto del índice"
            ):
                st.json(
                    index_snapshot[
                        "manifest"
                    ]
                )
        else:
            st.info(
                "El resumen vectorial aparecerá después de generar "
                "y validar el índice."
            )

    with monitoring_tab:
        render_quality_monitoring()

    with evaluation_tab:
        render_evaluation_bank_status()

    st.divider()

    render_rag_section(
        index_snapshot=index_snapshot,
        corpus_is_current=corpus_is_current,
    )



if __name__ == "__main__":
    main()
