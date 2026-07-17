"""Persistencia local de métricas y retroalimentación con SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bimbam_assistant.core.config import PROJECT_ROOT


MonitoringOutcome = Literal[
    "answered",
    "no_evidence",
    "rejected",
    "error",
]

FeedbackRating = Literal[
    "positive",
    "negative",
]


class MonitoringRepositoryError(RuntimeError):
    """Error relacionado con la persistencia de monitoreo."""


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """Datos mínimos necesarios para auditar una interacción."""

    interaction_id: str
    session_id: str
    question: str
    contextual_query: str
    category: str
    answer: str
    outcome: MonitoringOutcome
    verification_status: str
    verification_confidence: float
    used_context: bool
    source_count: int
    model_name: str
    latency_ms: int
    sources: list[dict[str, object]]
    error_message: str | None = None


def get_monitoring_database_path() -> Path:
    """Devuelve la ruta local de la base de monitoreo."""

    return (
        PROJECT_ROOT
        / "storage"
        / "monitoring"
        / "bimbam_quality.db"
    )


def _connect() -> sqlite3.Connection:
    """Abre una conexión configurada para uso local concurrente."""

    database_path = get_monitoring_database_path()

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        connection = sqlite3.connect(
            database_path,
            timeout=10,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible abrir la base de monitoreo."
        ) from error


def initialize_monitoring_database() -> Path:
    """Crea la estructura de persistencia cuando no existe."""

    schema = """
    CREATE TABLE IF NOT EXISTS interactions (
        interaction_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        created_at_utc TEXT NOT NULL DEFAULT (
            strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        ),
        question TEXT NOT NULL,
        contextual_query TEXT NOT NULL,
        category TEXT NOT NULL,
        answer TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (
            outcome IN (
                'answered',
                'no_evidence',
                'rejected',
                'error'
            )
        ),
        verification_status TEXT NOT NULL,
        verification_confidence REAL NOT NULL,
        used_context INTEGER NOT NULL CHECK (
            used_context IN (0, 1)
        ),
        source_count INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        latency_ms INTEGER NOT NULL,
        sources_json TEXT NOT NULL,
        feedback TEXT CHECK (
            feedback IN ('positive', 'negative')
            OR feedback IS NULL
        ),
        feedback_updated_at_utc TEXT,
        error_message TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_interactions_created_at
        ON interactions(created_at_utc DESC);

    CREATE INDEX IF NOT EXISTS idx_interactions_outcome
        ON interactions(outcome);

    CREATE INDEX IF NOT EXISTS idx_interactions_feedback
        ON interactions(feedback);
    """

    try:
        with _connect() as connection:
            connection.executescript(
                schema
            )

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible inicializar la base de monitoreo."
        ) from error

    return get_monitoring_database_path()


def save_interaction(
    record: InteractionRecord,
) -> None:
    """Guarda una interacción exitosa o fallida."""

    if not record.interaction_id.strip():
        raise MonitoringRepositoryError(
            "interaction_id no puede estar vacío."
        )

    if record.latency_ms < 0:
        raise MonitoringRepositoryError(
            "latency_ms no puede ser negativo."
        )

    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO interactions (
                    interaction_id,
                    session_id,
                    question,
                    contextual_query,
                    category,
                    answer,
                    outcome,
                    verification_status,
                    verification_confidence,
                    used_context,
                    source_count,
                    model_name,
                    latency_ms,
                    sources_json,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.interaction_id,
                    record.session_id,
                    record.question,
                    record.contextual_query,
                    record.category,
                    record.answer,
                    record.outcome,
                    record.verification_status,
                    record.verification_confidence,
                    int(record.used_context),
                    record.source_count,
                    record.model_name,
                    record.latency_ms,
                    json.dumps(
                        record.sources,
                        ensure_ascii=False,
                        default=str,
                    ),
                    record.error_message,
                ),
            )

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible guardar la interacción."
        ) from error


def update_interaction_feedback(
    interaction_id: str,
    rating: FeedbackRating,
) -> None:
    """Persiste la valoración positiva o negativa."""

    if rating not in {
        "positive",
        "negative",
    }:
        raise MonitoringRepositoryError(
            "La valoración debe ser positive o negative."
        )

    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                UPDATE interactions
                SET
                    feedback = ?,
                    feedback_updated_at_utc = (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                WHERE interaction_id = ?
                """,
                (
                    rating,
                    interaction_id,
                ),
            )

            if cursor.rowcount == 0:
                raise MonitoringRepositoryError(
                    "No se encontró la interacción para guardar "
                    "la retroalimentación."
                )

    except MonitoringRepositoryError:
        raise

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible guardar la retroalimentación."
        ) from error


def get_quality_summary() -> dict[str, int | float]:
    """Calcula las métricas generales del historial persistido."""

    query = """
    SELECT
        COUNT(*) AS total_interactions,
        SUM(CASE WHEN outcome = 'answered' THEN 1 ELSE 0 END)
            AS answered,
        SUM(CASE WHEN outcome = 'no_evidence' THEN 1 ELSE 0 END)
            AS no_evidence,
        SUM(CASE WHEN outcome = 'rejected' THEN 1 ELSE 0 END)
            AS rejected,
        SUM(CASE WHEN outcome = 'error' THEN 1 ELSE 0 END)
            AS errors,
        SUM(CASE WHEN feedback = 'positive' THEN 1 ELSE 0 END)
            AS positive_feedback,
        SUM(CASE WHEN feedback = 'negative' THEN 1 ELSE 0 END)
            AS negative_feedback,
        SUM(CASE WHEN feedback IS NOT NULL THEN 1 ELSE 0 END)
            AS feedback_count,
        COALESCE(AVG(latency_ms), 0) AS average_latency_ms
    FROM interactions
    """

    try:
        with _connect() as connection:
            row = connection.execute(
                query
            ).fetchone()

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible consultar las métricas."
        ) from error

    total = int(
        row["total_interactions"] or 0
    )

    feedback_count = int(
        row["feedback_count"] or 0
    )

    negative_feedback = int(
        row["negative_feedback"] or 0
    )

    feedback_rate = (
        feedback_count / total
        if total
        else 0.0
    )

    negative_rate = (
        negative_feedback / feedback_count
        if feedback_count
        else 0.0
    )

    return {
        "total_interactions": total,
        "answered": int(row["answered"] or 0),
        "no_evidence": int(row["no_evidence"] or 0),
        "rejected": int(row["rejected"] or 0),
        "errors": int(row["errors"] or 0),
        "positive_feedback": int(
            row["positive_feedback"] or 0
        ),
        "negative_feedback": negative_feedback,
        "feedback_count": feedback_count,
        "feedback_rate": feedback_rate,
        "negative_rate": negative_rate,
        "average_latency_ms": float(
            row["average_latency_ms"] or 0
        ),
    }


def get_recent_interactions(
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Devuelve las interacciones más recientes para auditoría."""

    if limit <= 0:
        raise MonitoringRepositoryError(
            "limit debe ser mayor que cero."
        )

    query = """
    SELECT
        created_at_utc,
        question,
        category,
        outcome,
        verification_status,
        verification_confidence,
        source_count,
        latency_ms,
        feedback
    FROM interactions
    ORDER BY created_at_utc DESC
    LIMIT ?
    """

    try:
        with _connect() as connection:
            rows = connection.execute(
                query,
                (limit,),
            ).fetchall()

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible consultar las interacciones recientes."
        ) from error

    return [
        {
            "Fecha UTC": row["created_at_utc"],
            "Pregunta": row["question"],
            "Categoría": row["category"],
            "Resultado": row["outcome"],
            "Verificación": row["verification_status"],
            "Confianza": round(
                float(row["verification_confidence"]),
                2,
            ),
            "Fuentes": int(row["source_count"]),
            "Latencia (ms)": int(row["latency_ms"]),
            "Feedback": row["feedback"] or "sin valorar",
        }
        for row in rows
    ]


def get_content_gap_questions(
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Lista preguntas sin evidencia, rechazadas o mal evaluadas."""

    if limit <= 0:
        raise MonitoringRepositoryError(
            "limit debe ser mayor que cero."
        )

    query = """
    SELECT
        created_at_utc,
        question,
        category,
        outcome,
        feedback,
        error_message
    FROM interactions
    WHERE
        outcome IN ('no_evidence', 'rejected', 'error')
        OR feedback = 'negative'
    ORDER BY created_at_utc DESC
    LIMIT ?
    """

    try:
        with _connect() as connection:
            rows = connection.execute(
                query,
                (limit,),
            ).fetchall()

    except sqlite3.Error as error:
        raise MonitoringRepositoryError(
            "No fue posible consultar los posibles vacíos "
            "de conocimiento."
        ) from error

    return [
        {
            "Fecha UTC": row["created_at_utc"],
            "Pregunta": row["question"],
            "Categoría": row["category"],
            "Resultado": row["outcome"],
            "Feedback": row["feedback"] or "sin valorar",
            "Error": row["error_message"] or "",
        }
        for row in rows
    ]
