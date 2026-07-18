"""Ejecuta evaluaciones controladas del sistema RAG.

El script ofrece dos modos:

1. retrieval:
   Evalúa únicamente la recuperación semántica.
   Presupuesto conservador: 1 llamada por pregunta.

2. full:
   Ejecuta recuperación, generación y verificación.
   Presupuesto conservador: 4 llamadas por pregunta:
   - embedding de consulta;
   - generación de respuesta;
   - verificación estructurada;
   - posible reintento de verificación.

Por seguridad, el script funciona como simulación hasta que se agrega
la opción ``--execute``.

Ejemplos:

    python scripts/evaluate_rag.py --batch A
    python scripts/evaluate_rag.py --batch A --mode retrieval --execute
    python scripts/evaluate_rag.py --batch A --mode full --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# ==========================================================
# Preparación de la ruta del proyecto
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from bimbam_assistant.application.rag_service import (
    RagGenerationError,
    RetrievalError,
    answer_question,
    retrieve_documents,
)
from scripts.validate_evaluation_bank import (
    EvaluationBankValidationError,
    load_evaluation_bank,
    validate_evaluation_bank,
)


DEFAULT_BANK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "questions.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "evaluation"
)

DEFAULT_LEDGER_PATH = (
    DEFAULT_OUTPUT_DIRECTORY
    / "gemini_budget.json"
)

DEFAULT_DAILY_LIMIT = 20
DEFAULT_SAFETY_BUFFER = 2

MODE_RETRIEVAL = "retrieval"
MODE_FULL = "full"

MODE_CALL_RESERVATION = {
    MODE_RETRIEVAL: 1,
    MODE_FULL: 4,
}

CITATION_PATTERN = re.compile(
    r"\[Fuente\s+(\d+)\]",
    flags=re.IGNORECASE,
)


class EvaluationError(RuntimeError):
    """Error producido durante la evaluación."""


class BudgetExceededError(EvaluationError):
    """Indica que la ejecución excedería el presupuesto diario."""


@dataclass(frozen=True)
class EvaluationPlan:
    """Describe una ejecución antes de realizar llamadas."""

    mode: str
    question_ids: tuple[str, ...]
    calls_per_question: int
    reserved_calls: int
    daily_limit: int
    safety_buffer: int
    usable_daily_budget: int
    calls_already_reserved: int
    calls_available: int
    can_execute: bool


def utc_now_iso() -> str:
    """Devuelve la fecha y hora UTC en formato ISO."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_text(
    value: object,
) -> str:
    """Normaliza texto para comparaciones simples."""

    text = str(
        value or ""
    ).strip().lower()

    decomposed = unicodedata.normalize(
        "NFKD",
        text,
    )

    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(
            character
        )
    )

    return " ".join(
        without_accents.split()
    )


def serialize_value(
    value: object,
) -> object:
    """Convierte modelos y objetos anidados a tipos serializables."""

    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): serialize_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            serialize_value(
                item
            )
            for item in value
        ]

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        return serialize_value(
            model_dump()
        )

    object_dictionary = getattr(
        value,
        "__dict__",
        None,
    )

    if isinstance(
        object_dictionary,
        dict,
    ):
        return {
            str(key): serialize_value(
                item
            )
            for key, item in object_dictionary.items()
            if not str(key).startswith(
                "_"
            )
        }

    return str(
        value
    )


def load_budget_ledger(
    path: Path,
    *,
    daily_limit: int,
    safety_buffer: int,
) -> dict[str, Any]:
    """Carga el presupuesto diario o crea uno para la fecha actual."""

    today = date.today().isoformat()

    empty_ledger = {
        "date": today,
        "daily_limit": daily_limit,
        "safety_buffer": safety_buffer,
        "reserved_calls": 0,
        "entries": [],
    }

    if not path.is_file():
        return empty_ledger

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return empty_ledger

    if not isinstance(
        payload,
        dict,
    ):
        return empty_ledger

    if payload.get(
        "date"
    ) != today:
        return empty_ledger

    payload[
        "daily_limit"
    ] = daily_limit

    payload[
        "safety_buffer"
    ] = safety_buffer

    payload.setdefault(
        "reserved_calls",
        0,
    )

    payload.setdefault(
        "entries",
        [],
    )

    return payload


def save_budget_ledger(
    path: Path,
    ledger: dict[str, Any],
) -> None:
    """Guarda el presupuesto mediante reemplazo atómico."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            ledger,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def reserve_budget(
    *,
    ledger_path: Path,
    question_id: str,
    mode: str,
    calls: int,
    daily_limit: int,
    safety_buffer: int,
) -> dict[str, Any]:
    """Reserva presupuesto antes de realizar llamadas a Gemini."""

    if calls <= 0:
        raise EvaluationError(
            "La reserva debe ser mayor que cero."
        )

    usable_budget = (
        daily_limit
        - safety_buffer
    )

    if usable_budget <= 0:
        raise EvaluationError(
            "El límite diario debe ser mayor que el margen de seguridad."
        )

    ledger = load_budget_ledger(
        ledger_path,
        daily_limit=daily_limit,
        safety_buffer=safety_buffer,
    )

    already_reserved = int(
        ledger.get(
            "reserved_calls",
            0,
        )
    )

    projected_total = (
        already_reserved
        + calls
    )

    if projected_total > usable_budget:
        raise BudgetExceededError(
            "La pregunta excedería el presupuesto protegido del día. "
            f"Reservadas: {already_reserved}. "
            f"Solicitadas: {calls}. "
            f"Máximo utilizable: {usable_budget}."
        )

    ledger[
        "reserved_calls"
    ] = projected_total

    entries = ledger.setdefault(
        "entries",
        [],
    )

    entries.append(
        {
            "timestamp_utc": utc_now_iso(),
            "question_id": question_id,
            "mode": mode,
            "reserved_calls": calls,
            "status": "reserved",
        }
    )

    save_budget_ledger(
        ledger_path,
        ledger,
    )

    return ledger


def update_last_budget_entry(
    *,
    ledger_path: Path,
    daily_limit: int,
    safety_buffer: int,
    question_id: str,
    status: str,
) -> None:
    """Actualiza el estado informativo de la última reserva."""

    ledger = load_budget_ledger(
        ledger_path,
        daily_limit=daily_limit,
        safety_buffer=safety_buffer,
    )

    entries = ledger.get(
        "entries",
        [],
    )

    for entry in reversed(
        entries
    ):
        if (
            entry.get(
                "question_id"
            )
            == question_id
            and entry.get(
                "status"
            )
            == "reserved"
        ):
            entry[
                "status"
            ] = status

            entry[
                "completed_at_utc"
            ] = utc_now_iso()

            break

    save_budget_ledger(
        ledger_path,
        ledger,
    )


def select_questions(
    questions: Sequence[dict[str, Any]],
    *,
    batch: str | None = None,
    tier: str | None = None,
    identifiers: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Selecciona preguntas por lote, nivel o identificador."""

    selected = list(
        questions
    )

    if batch:
        normalized_batch = batch.strip().upper()

        selected = [
            question
            for question in selected
            if str(
                question.get(
                    "budget_batch",
                    "",
                )
            ).upper()
            == normalized_batch
        ]

    if tier:
        normalized_tier = tier.strip().lower()

        selected = [
            question
            for question in selected
            if str(
                question.get(
                    "evaluation_tier",
                    "",
                )
            ).lower()
            == normalized_tier
        ]

    if identifiers:
        expected_ids = {
            identifier.strip()
            for identifier in identifiers
            if identifier.strip()
        }

        selected = [
            question
            for question in selected
            if str(
                question.get(
                    "id",
                    "",
                )
            )
            in expected_ids
        ]

        found_ids = {
            str(
                question.get(
                    "id"
                )
            )
            for question in selected
        }

        missing_ids = (
            expected_ids
            - found_ids
        )

        if missing_ids:
            raise EvaluationError(
                "No se encontraron estos identificadores: "
                f"{sorted(missing_ids)}"
            )

    return selected


def build_plan(
    *,
    questions: Sequence[dict[str, Any]],
    mode: str,
    ledger_path: Path,
    daily_limit: int,
    safety_buffer: int,
) -> EvaluationPlan:
    """Calcula el costo conservador antes de ejecutar."""

    calls_per_question = (
        MODE_CALL_RESERVATION[
            mode
        ]
    )

    reserved_calls = (
        calls_per_question
        * len(
            questions
        )
    )

    usable_daily_budget = (
        daily_limit
        - safety_buffer
    )

    ledger = load_budget_ledger(
        ledger_path,
        daily_limit=daily_limit,
        safety_buffer=safety_buffer,
    )

    calls_already_reserved = int(
        ledger.get(
            "reserved_calls",
            0,
        )
    )

    calls_available = max(
        usable_daily_budget
        - calls_already_reserved,
        0,
    )

    return EvaluationPlan(
        mode=mode,
        question_ids=tuple(
            str(
                question["id"]
            )
            for question in questions
        ),
        calls_per_question=calls_per_question,
        reserved_calls=reserved_calls,
        daily_limit=daily_limit,
        safety_buffer=safety_buffer,
        usable_daily_budget=usable_daily_budget,
        calls_already_reserved=calls_already_reserved,
        calls_available=calls_available,
        can_execute=(
            reserved_calls
            <= calls_available
        ),
    )


def get_retrieval_results(
    retrieval: object,
) -> list[object]:
    """Obtiene los resultados de recuperación de forma segura."""

    results = getattr(
        retrieval,
        "results",
        [],
    )

    return list(
        results or []
    )


def get_metadata(
    result: object,
) -> dict[str, Any]:
    """Obtiene los metadatos de un resultado."""

    metadata = getattr(
        result,
        "metadata",
        {},
    )

    return (
        dict(
            metadata
        )
        if isinstance(
            metadata,
            dict,
        )
        else {}
    )


def evaluate_retrieval(
    *,
    question: dict[str, Any],
    retrieval: object,
) -> dict[str, Any]:
    """Calcula métricas deterministas de recuperación."""

    results = get_retrieval_results(
        retrieval
    )

    retrieved_documents: list[str] = []
    retrieved_pages: list[int] = []
    retrieved_categories: list[str] = []
    retrieved_scores: list[float] = []

    for result in results:
        metadata = get_metadata(
            result
        )

        document_name = metadata.get(
            "document_name"
        )

        if document_name:
            retrieved_documents.append(
                str(
                    document_name
                )
            )

        page_number = metadata.get(
            "page_number"
        )

        try:
            if page_number is not None:
                retrieved_pages.append(
                    int(
                        page_number
                    )
                )
        except (
            TypeError,
            ValueError,
        ):
            pass

        category = metadata.get(
            "category"
        )

        if category:
            retrieved_categories.append(
                str(
                    category
                )
            )

        score = getattr(
            result,
            "score",
            None,
        )

        try:
            if score is not None:
                retrieved_scores.append(
                    float(
                        score
                    )
                )
        except (
            TypeError,
            ValueError,
        ):
            pass

    normalized_retrieved_documents = {
        normalize_text(
            document_name
        )
        for document_name in retrieved_documents
    }

    normalized_expected_documents = {
        normalize_text(
            document_name
        )
        for document_name in question.get(
            "expected_documents",
            [],
        )
    }

    retrieved_page_set = set(
        retrieved_pages
    )

    expected_page_set = {
        int(
            page
        )
        for page in question.get(
            "expected_pages",
            [],
        )
    }

    expected_category = str(
        question.get(
            "category",
            "",
        )
    )

    category_hit = (
        expected_category
        in {
            "multi_document",
            "fuera_de_alcance",
        }
        or normalize_text(
            expected_category
        )
        in {
            normalize_text(
                category
            )
            for category in retrieved_categories
        }
    )

    document_hit_any = bool(
        normalized_expected_documents
        & normalized_retrieved_documents
    )

    document_hit_all = (
        normalized_expected_documents.issubset(
            normalized_retrieved_documents
        )
        if normalized_expected_documents
        else len(
            results
        )
        == 0
    )

    page_hit_any = bool(
        expected_page_set
        & retrieved_page_set
    )

    page_hit_all = (
        expected_page_set.issubset(
            retrieved_page_set
        )
        if expected_page_set
        else len(
            results
        )
        == 0
    )

    return {
        "result_count": len(
            results
        ),
        "retrieved_documents": retrieved_documents,
        "retrieved_pages": retrieved_pages,
        "retrieved_categories": retrieved_categories,
        "retrieved_scores": retrieved_scores,
        "top_score": (
            max(
                retrieved_scores
            )
            if retrieved_scores
            else None
        ),
        "document_hit_any": document_hit_any,
        "document_hit_all": document_hit_all,
        "page_hit_any": page_hit_any,
        "page_hit_all": page_hit_all,
        "category_hit": category_hit,
    }


def evaluate_full_response(
    *,
    question: dict[str, Any],
    response: object,
) -> dict[str, Any]:
    """Calcula métricas deterministas de generación y verificación."""

    answer = str(
        getattr(
            response,
            "answer",
            "",
        )
        or ""
    )

    retrieval = getattr(
        response,
        "retrieval",
        None,
    )

    verification = getattr(
        response,
        "verification",
        None,
    )

    verification_passed = bool(
        getattr(
            verification,
            "passed",
            False,
        )
    )

    verification_status = str(
        getattr(
            verification,
            "status",
            "",
        )
        or ""
    )

    confidence = getattr(
        verification,
        "confidence",
        None,
    )

    citations = [
        int(
            citation
        )
        for citation in CITATION_PATTERN.findall(
            answer
        )
    ]

    expected_behavior = question.get(
        "expected_behavior"
    )

    used_context = bool(
        getattr(
            response,
            "used_context",
            False,
        )
    )

    fallback_detected = (
        "no encontre informacion suficiente"
        in normalize_text(
            answer
        )
        or "no pude validar automaticamente"
        in normalize_text(
            answer
        )
        or verification_status
        in {
            "not_applicable",
            "rejected",
        }
        or not used_context
    )

    if expected_behavior == "answer":
        behavior_passed = bool(
            used_context
            and verification_passed
            and citations
        )
    else:
        behavior_passed = fallback_detected

    retrieval_metrics = (
        evaluate_retrieval(
            question=question,
            retrieval=retrieval,
        )
        if retrieval is not None
        else {}
    )

    invalid_citations = getattr(
        verification,
        "invalid_citations",
        [],
    )

    unsupported_claims = getattr(
        verification,
        "unsupported_claims",
        [],
    )

    return {
        **retrieval_metrics,
        "answer": answer,
        "used_context": used_context,
        "model_name": getattr(
            response,
            "model_name",
            None,
        ),
        "citation_numbers": citations,
        "citations_present": bool(
            citations
        ),
        "verification_status": verification_status,
        "verification_passed": verification_passed,
        "verification_confidence": confidence,
        "invalid_citations": serialize_value(
            invalid_citations
        ),
        "unsupported_claims": serialize_value(
            unsupported_claims
        ),
        "fallback_detected": fallback_detected,
        "expected_behavior_passed": behavior_passed,
        "expected_facts_for_manual_review": question.get(
            "expected_facts",
            [],
        ),
        "forbidden_facts_for_manual_review": question.get(
            "forbidden_facts",
            [],
        ),
    }


def execute_question(
    *,
    question: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Ejecuta una pregunta en el modo seleccionado."""

    started_at = time.perf_counter()

    if mode == MODE_RETRIEVAL:
        retrieval = retrieve_documents(
            str(
                question["question"]
            )
        )

        metrics = evaluate_retrieval(
            question=question,
            retrieval=retrieval,
        )

        raw_response = serialize_value(
            retrieval
        )

    elif mode == MODE_FULL:
        response = answer_question(
            str(
                question["question"]
            )
        )

        metrics = evaluate_full_response(
            question=question,
            response=response,
        )

        raw_response = serialize_value(
            response
        )

    else:
        raise EvaluationError(
            f"Modo no soportado: {mode}"
        )

    latency_seconds = round(
        time.perf_counter()
        - started_at,
        4,
    )

    return {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "budget_batch": question["budget_batch"],
        "evaluation_tier": question["evaluation_tier"],
        "mode": mode,
        "status": "completed",
        "latency_seconds": latency_seconds,
        "evaluated_at_utc": utc_now_iso(),
        "metrics": metrics,
        "raw_response": raw_response,
    }


def build_error_result(
    *,
    question: dict[str, Any],
    mode: str,
    error: Exception,
    latency_seconds: float,
) -> dict[str, Any]:
    """Construye un resultado persistible para una pregunta fallida."""

    return {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "budget_batch": question["budget_batch"],
        "evaluation_tier": question["evaluation_tier"],
        "mode": mode,
        "status": "error",
        "latency_seconds": round(
            latency_seconds,
            4,
        ),
        "evaluated_at_utc": utc_now_iso(),
        "error_type": type(
            error
        ).__name__,
        "error": str(
            error
        ),
    }


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Escribe resultados en formato JSON Lines."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv_summary(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    """Escribe una tabla compacta de resultados."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question_id",
        "mode",
        "status",
        "category",
        "budget_batch",
        "latency_seconds",
        "result_count",
        "document_hit_any",
        "document_hit_all",
        "page_hit_any",
        "page_hit_all",
        "category_hit",
        "expected_behavior_passed",
        "verification_passed",
        "verification_confidence",
        "error",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            metrics = record.get(
                "metrics",
                {},
            )

            writer.writerow(
                {
                    "question_id": record.get(
                        "question_id"
                    ),
                    "mode": record.get(
                        "mode"
                    ),
                    "status": record.get(
                        "status"
                    ),
                    "category": record.get(
                        "category"
                    ),
                    "budget_batch": record.get(
                        "budget_batch"
                    ),
                    "latency_seconds": record.get(
                        "latency_seconds"
                    ),
                    "result_count": metrics.get(
                        "result_count"
                    ),
                    "document_hit_any": metrics.get(
                        "document_hit_any"
                    ),
                    "document_hit_all": metrics.get(
                        "document_hit_all"
                    ),
                    "page_hit_any": metrics.get(
                        "page_hit_any"
                    ),
                    "page_hit_all": metrics.get(
                        "page_hit_all"
                    ),
                    "category_hit": metrics.get(
                        "category_hit"
                    ),
                    "expected_behavior_passed": metrics.get(
                        "expected_behavior_passed"
                    ),
                    "verification_passed": metrics.get(
                        "verification_passed"
                    ),
                    "verification_confidence": metrics.get(
                        "verification_confidence"
                    ),
                    "error": record.get(
                        "error"
                    ),
                }
            )


def summarize_results(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Resume una ejecución sin usar un juez LLM adicional."""

    completed = [
        record
        for record in records
        if record.get(
            "status"
        )
        == "completed"
    ]

    errors = [
        record
        for record in records
        if record.get(
            "status"
        )
        == "error"
    ]

    retrieval_metrics = [
        record.get(
            "metrics",
            {},
        )
        for record in completed
    ]

    def ratio(
        key: str,
    ) -> float | None:
        values = [
            bool(
                metrics[key]
            )
            for metrics in retrieval_metrics
            if key in metrics
            and metrics[key] is not None
        ]

        if not values:
            return None

        return round(
            sum(
                values
            )
            / len(
                values
            ),
            4,
        )

    latencies = [
        float(
            record.get(
                "latency_seconds",
                0,
            )
        )
        for record in completed
    ]

    return {
        "questions_attempted": len(
            records
        ),
        "questions_completed": len(
            completed
        ),
        "questions_with_error": len(
            errors
        ),
        "document_hit_any_rate": ratio(
            "document_hit_any"
        ),
        "document_hit_all_rate": ratio(
            "document_hit_all"
        ),
        "page_hit_any_rate": ratio(
            "page_hit_any"
        ),
        "category_hit_rate": ratio(
            "category_hit"
        ),
        "expected_behavior_pass_rate": ratio(
            "expected_behavior_passed"
        ),
        "verification_pass_rate": ratio(
            "verification_passed"
        ),
        "average_latency_seconds": (
            round(
                sum(
                    latencies
                )
                / len(
                    latencies
                ),
                4,
            )
            if latencies
            else None
        ),
        "semantic_fact_review": (
            "manual_required"
        ),
        "semantic_fact_review_reason": (
            "No se usa un juez LLM adicional para no consumir llamadas "
            "extra de Gemini."
        ),
    }


def print_plan(
    plan: EvaluationPlan,
) -> None:
    """Muestra el presupuesto previsto."""

    print()
    print("=" * 72)
    print("PLAN DE EVALUACIÓN RAG")
    print("=" * 72)
    print(
        f"Modo                    : {plan.mode}"
    )
    print(
        f"Preguntas seleccionadas : {len(plan.question_ids)}"
    )
    print(
        f"IDs                     : {', '.join(plan.question_ids)}"
    )
    print(
        f"Reserva por pregunta    : {plan.calls_per_question}"
    )
    print(
        f"Reserva de la ejecución : {plan.reserved_calls}"
    )
    print(
        f"Límite diario           : {plan.daily_limit}"
    )
    print(
        f"Margen de seguridad     : {plan.safety_buffer}"
    )
    print(
        f"Presupuesto utilizable  : {plan.usable_daily_budget}"
    )
    print(
        f"Ya reservado hoy        : {plan.calls_already_reserved}"
    )
    print(
        f"Disponible hoy          : {plan.calls_available}"
    )
    print(
        f"Puede ejecutarse        : {'Sí' if plan.can_execute else 'No'}"
    )
    print("=" * 72)


def print_run_summary(
    *,
    summary: dict[str, Any],
    jsonl_path: Path,
    csv_path: Path,
    summary_path: Path,
) -> None:
    """Muestra el resultado final de la evaluación."""

    print()
    print("=" * 72)
    print("RESULTADO DE LA EVALUACIÓN")
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key:<30}: {value}"
        )

    print()
    print(
        f"Detalle JSONL : {jsonl_path}"
    )
    print(
        f"Resumen CSV   : {csv_path}"
    )
    print(
        f"Resumen JSON  : {summary_path}"
    )
    print("=" * 72)


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Interpreta los argumentos del evaluador."""

    parser = argparse.ArgumentParser(
        description=(
            "Evalúa la recuperación o el flujo RAG completo con "
            "un presupuesto conservador de llamadas a Gemini."
        )
    )

    parser.add_argument(
        "--bank",
        type=Path,
        default=DEFAULT_BANK_PATH,
        help="Ruta del banco questions.json.",
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="Lote de presupuesto, por ejemplo A.",
    )

    parser.add_argument(
        "--tier",
        choices=(
            "smoke",
            "standard",
            "full",
        ),
        help="Nivel de evaluación.",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        help="Uno o más identificadores de preguntas.",
    )

    parser.add_argument(
        "--mode",
        choices=(
            MODE_RETRIEVAL,
            MODE_FULL,
        ),
        default=MODE_RETRIEVAL,
        help="Tipo de evaluación. Por defecto: retrieval.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Realiza las llamadas. Sin esta opción solo muestra el plan.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directorio para resultados y presupuesto.",
    )

    parser.add_argument(
        "--daily-limit",
        type=int,
        default=DEFAULT_DAILY_LIMIT,
        help="Límite diario conocido de llamadas.",
    )

    parser.add_argument(
        "--safety-buffer",
        type=int,
        default=DEFAULT_SAFETY_BUFFER,
        help="Llamadas que el evaluador dejará sin utilizar.",
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Punto de entrada del evaluador."""

    arguments = parse_arguments(
        argv
    )

    try:
        bank = load_evaluation_bank(
            arguments.bank
        )

        validate_evaluation_bank(
            bank
        )

        questions = select_questions(
            bank["questions"],
            batch=arguments.batch,
            tier=arguments.tier,
            identifiers=arguments.ids,
        )

        if not questions:
            raise EvaluationError(
                "La selección no contiene preguntas."
            )

        if (
            arguments.execute
            and not any(
                (
                    arguments.batch,
                    arguments.tier,
                    arguments.ids,
                )
            )
        ):
            raise EvaluationError(
                "Para ejecutar llamadas debes seleccionar --batch, "
                "--tier o --ids. Esto evita evaluar todo el banco "
                "por accidente."
            )

        if arguments.daily_limit <= 0:
            raise EvaluationError(
                "--daily-limit debe ser mayor que cero."
            )

        if arguments.safety_buffer < 0:
            raise EvaluationError(
                "--safety-buffer no puede ser negativo."
            )

        ledger_path = (
            arguments.output_directory
            / "gemini_budget.json"
        )

        plan = build_plan(
            questions=questions,
            mode=arguments.mode,
            ledger_path=ledger_path,
            daily_limit=arguments.daily_limit,
            safety_buffer=arguments.safety_buffer,
        )

        print_plan(
            plan
        )

        if not arguments.execute:
            print()
            print(
                "SIMULACIÓN: no se realizaron llamadas a Gemini."
            )
            print(
                "Agrega --execute después de revisar el presupuesto."
            )
            return 0

        if not plan.can_execute:
            raise BudgetExceededError(
                "La ejecución completa no cabe en el presupuesto "
                "protegido disponible."
            )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_name = (
            f"rag-evaluation-{arguments.mode}-{timestamp}"
        )

        run_directory = (
            arguments.output_directory
            / "runs"
            / run_name
        )

        results: list[dict[str, Any]] = []

        calls_per_question = (
            MODE_CALL_RESERVATION[
                arguments.mode
            ]
        )

        for position, question in enumerate(
            questions,
            start=1,
        ):
            question_id = str(
                question["id"]
            )

            print()
            print(
                f"[{position}/{len(questions)}] "
                f"{question_id}: {question['question']}"
            )

            reserve_budget(
                ledger_path=ledger_path,
                question_id=question_id,
                mode=arguments.mode,
                calls=calls_per_question,
                daily_limit=arguments.daily_limit,
                safety_buffer=arguments.safety_buffer,
            )

            started_at = time.perf_counter()

            try:
                result = execute_question(
                    question=question,
                    mode=arguments.mode,
                )

                update_last_budget_entry(
                    ledger_path=ledger_path,
                    daily_limit=arguments.daily_limit,
                    safety_buffer=arguments.safety_buffer,
                    question_id=question_id,
                    status="completed",
                )

                print(
                    "  OK | "
                    f"{result['latency_seconds']} s"
                )

            except (
                RetrievalError,
                RagGenerationError,
                Exception,
            ) as error:
                latency_seconds = (
                    time.perf_counter()
                    - started_at
                )

                result = build_error_result(
                    question=question,
                    mode=arguments.mode,
                    error=error,
                    latency_seconds=latency_seconds,
                )

                update_last_budget_entry(
                    ledger_path=ledger_path,
                    daily_limit=arguments.daily_limit,
                    safety_buffer=arguments.safety_buffer,
                    question_id=question_id,
                    status="error",
                )

                print(
                    f"  ERROR | {error}"
                )

            results.append(
                result
            )

        jsonl_path = (
            run_directory
            / "results.jsonl"
        )

        csv_path = (
            run_directory
            / "summary.csv"
        )

        summary_path = (
            run_directory
            / "summary.json"
        )

        write_jsonl(
            jsonl_path,
            results,
        )

        write_csv_summary(
            csv_path,
            results,
        )

        summary = {
            "run_name": run_name,
            "mode": arguments.mode,
            "question_ids": [
                result["question_id"]
                for result in results
            ],
            "reserved_calls": (
                calls_per_question
                * len(
                    results
                )
            ),
            "daily_limit": arguments.daily_limit,
            "safety_buffer": arguments.safety_buffer,
            "generated_at_utc": utc_now_iso(),
            **summarize_results(
                results
            ),
        }

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print_run_summary(
            summary=summary,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
            summary_path=summary_path,
        )

        return 0

    except (
        EvaluationBankValidationError,
        EvaluationError,
    ) as error:
        print(
            f"ERROR | {error}"
        )

        return 1

    except KeyboardInterrupt:
        print(
            "Evaluación interrumpida por el usuario."
        )

        return 130


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
