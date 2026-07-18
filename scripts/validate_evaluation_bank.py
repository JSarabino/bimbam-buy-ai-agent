"""Valida el banco de evaluación RAG sin consumir Gemini."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "questions.json"
)

ALLOWED_CATEGORIES = {
    "envios",
    "garantias",
    "reembolsos_devoluciones",
    "metodos_pago",
    "afiliados",
    "multi_document",
    "fuera_de_alcance",
}

ALLOWED_TIERS = {
    "smoke",
    "standard",
    "full",
}

ALLOWED_TYPES = {
    "factual",
    "procedural",
    "multi_document",
    "out_of_scope",
}

ALLOWED_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

ALLOWED_BEHAVIORS = {
    "answer",
    "fallback",
}


class EvaluationBankValidationError(ValueError):
    """Indica que el banco de evaluación no cumple el esquema."""


def load_evaluation_bank(
    path: Path,
) -> dict[str, Any]:
    """Carga el archivo JSON del banco."""

    if not path.is_file():
        raise EvaluationBankValidationError(
            f"No se encontró el banco de evaluación: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise EvaluationBankValidationError(
            f"El banco no contiene JSON válido: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise EvaluationBankValidationError(
            "La raíz del banco debe ser un objeto JSON."
        )

    return payload


def _require_non_empty_string(
    value: object,
    *,
    field_name: str,
    question_id: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationBankValidationError(
            f"{question_id}: '{field_name}' debe ser un texto no vacío."
        )


def _require_string_list(
    value: object,
    *,
    field_name: str,
    question_id: str,
) -> list[str]:
    if not isinstance(value, list):
        raise EvaluationBankValidationError(
            f"{question_id}: '{field_name}' debe ser una lista."
        )

    if any(
        not isinstance(item, str)
        or not item.strip()
        for item in value
    ):
        raise EvaluationBankValidationError(
            f"{question_id}: '{field_name}' solo admite textos no vacíos."
        )

    return value


def validate_evaluation_bank(
    bank: dict[str, Any],
) -> dict[str, int]:
    """Valida estructura, coherencia y presupuesto del banco."""

    if bank.get("schema_version") != 1:
        raise EvaluationBankValidationError(
            "schema_version debe ser 1."
        )

    questions = bank.get(
        "questions"
    )

    if not isinstance(questions, list) or not questions:
        raise EvaluationBankValidationError(
            "'questions' debe ser una lista no vacía."
        )

    if bank.get("question_count") != len(questions):
        raise EvaluationBankValidationError(
            "question_count no coincide con la cantidad real."
        )

    identifiers: set[str] = set()
    category_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()
    fallback_count = 0

    for index, item in enumerate(
        questions,
        start=1,
    ):
        if not isinstance(item, dict):
            raise EvaluationBankValidationError(
                f"Pregunta {index}: debe ser un objeto JSON."
            )

        question_id = item.get(
            "id",
            f"pregunta-{index}",
        )

        _require_non_empty_string(
            question_id,
            field_name="id",
            question_id=f"Pregunta {index}",
        )

        if question_id in identifiers:
            raise EvaluationBankValidationError(
                f"ID duplicado: {question_id}"
            )

        identifiers.add(
            question_id
        )

        _require_non_empty_string(
            item.get("question"),
            field_name="question",
            question_id=question_id,
        )

        category = item.get(
            "category"
        )

        if category not in ALLOWED_CATEGORIES:
            raise EvaluationBankValidationError(
                f"{question_id}: categoría no válida: {category}"
            )

        tier = item.get(
            "evaluation_tier"
        )

        if tier not in ALLOWED_TIERS:
            raise EvaluationBankValidationError(
                f"{question_id}: tier no válido: {tier}"
            )

        question_type = item.get(
            "question_type"
        )

        if question_type not in ALLOWED_TYPES:
            raise EvaluationBankValidationError(
                f"{question_id}: tipo no válido: {question_type}"
            )

        difficulty = item.get(
            "difficulty"
        )

        if difficulty not in ALLOWED_DIFFICULTIES:
            raise EvaluationBankValidationError(
                f"{question_id}: dificultad no válida: {difficulty}"
            )

        behavior = item.get(
            "expected_behavior"
        )

        if behavior not in ALLOWED_BEHAVIORS:
            raise EvaluationBankValidationError(
                f"{question_id}: comportamiento no válido: {behavior}"
            )

        batch = item.get(
            "budget_batch"
        )

        _require_non_empty_string(
            batch,
            field_name="budget_batch",
            question_id=question_id,
        )

        expected_documents = _require_string_list(
            item.get("expected_documents"),
            field_name="expected_documents",
            question_id=question_id,
        )

        expected_facts = _require_string_list(
            item.get("expected_facts"),
            field_name="expected_facts",
            question_id=question_id,
        )

        _require_string_list(
            item.get("forbidden_facts"),
            field_name="forbidden_facts",
            question_id=question_id,
        )

        expected_pages = item.get(
            "expected_pages"
        )

        if (
            not isinstance(expected_pages, list)
            or any(
                not isinstance(page, int)
                or page <= 0
                for page in expected_pages
            )
        ):
            raise EvaluationBankValidationError(
                f"{question_id}: expected_pages debe contener enteros positivos."
            )

        should_have_answer = item.get(
            "should_have_answer"
        )

        should_cite_sources = item.get(
            "should_cite_sources"
        )

        if not isinstance(
            should_have_answer,
            bool,
        ):
            raise EvaluationBankValidationError(
                f"{question_id}: should_have_answer debe ser booleano."
            )

        if not isinstance(
            should_cite_sources,
            bool,
        ):
            raise EvaluationBankValidationError(
                f"{question_id}: should_cite_sources debe ser booleano."
            )

        if not expected_facts:
            raise EvaluationBankValidationError(
                f"{question_id}: expected_facts no puede estar vacío."
            )

        if behavior == "answer":
            if not should_have_answer:
                raise EvaluationBankValidationError(
                    f"{question_id}: answer requiere should_have_answer=true."
                )

            if not should_cite_sources:
                raise EvaluationBankValidationError(
                    f"{question_id}: answer requiere citas."
                )

            if not expected_documents or not expected_pages:
                raise EvaluationBankValidationError(
                    f"{question_id}: answer requiere documentos y páginas."
                )

        if behavior == "fallback":
            fallback_count += 1

            if should_have_answer:
                raise EvaluationBankValidationError(
                    f"{question_id}: fallback requiere should_have_answer=false."
                )

            if should_cite_sources:
                raise EvaluationBankValidationError(
                    f"{question_id}: fallback no debe exigir citas."
                )

            if expected_documents or expected_pages:
                raise EvaluationBankValidationError(
                    f"{question_id}: fallback no debe declarar fuentes esperadas."
                )

        category_counts[
            category
        ] += 1

        tier_counts[
            tier
        ] += 1

        batch_counts[
            batch
        ] += 1

    oversized_batches = {
        batch: count
        for batch, count in batch_counts.items()
        if count > 4
    }

    if oversized_batches:
        raise EvaluationBankValidationError(
            "Los lotes exceden cuatro preguntas: "
            f"{oversized_batches}"
        )

    required_corpus_categories = {
        "envios",
        "garantias",
        "reembolsos_devoluciones",
        "metodos_pago",
        "afiliados",
    }

    missing_categories = (
        required_corpus_categories
        - set(category_counts)
    )

    if missing_categories:
        raise EvaluationBankValidationError(
            "Faltan categorías del corpus: "
            f"{sorted(missing_categories)}"
        )

    return {
        "questions": len(questions),
        "categories": len(category_counts),
        "tiers": len(tier_counts),
        "batches": len(batch_counts),
        "fallback_questions": fallback_count,
    }


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida el banco de evaluación RAG sin realizar "
            "llamadas a Gemini."
        )
    )

    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_BANK_PATH,
        help="Ruta del archivo questions.json.",
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = parse_arguments(
        argv
    )

    try:
        bank = load_evaluation_bank(
            arguments.path
        )

        summary = validate_evaluation_bank(
            bank
        )
    except EvaluationBankValidationError as error:
        print(
            f"ERROR | {error}"
        )
        return 1

    print("=" * 64)
    print("BANCO DE EVALUACIÓN RAG VÁLIDO")
    print("=" * 64)

    for key, value in summary.items():
        print(
            f"{key:<20}: {value}"
        )

    print()
    print(
        "Esta validación no realizó llamadas a Gemini."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
