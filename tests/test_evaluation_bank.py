"""Pruebas offline del banco de evaluación RAG."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.validate_evaluation_bank import (
    load_evaluation_bank,
    validate_evaluation_bank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "questions.json"
)


def load_validated_bank() -> dict:
    bank = load_evaluation_bank(
        BANK_PATH
    )

    validate_evaluation_bank(
        bank
    )

    return bank


def test_bank_has_twenty_unique_questions() -> None:
    bank = load_validated_bank()
    questions = bank["questions"]
    identifiers = {
        question["id"]
        for question in questions
    }

    assert len(questions) == 20
    assert len(identifiers) == 20


def test_bank_covers_all_corpus_categories() -> None:
    bank = load_validated_bank()
    categories = {
        question["category"]
        for question in bank["questions"]
    }

    assert {
        "envios",
        "garantias",
        "reembolsos_devoluciones",
        "metodos_pago",
        "afiliados",
    }.issubset(categories)


def test_smoke_suite_covers_all_corpus_categories() -> None:
    bank = load_validated_bank()

    smoke_categories = {
        question["category"]
        for question in bank["questions"]
        if question["evaluation_tier"] == "smoke"
    }

    assert smoke_categories == {
        "envios",
        "garantias",
        "reembolsos_devoluciones",
        "metodos_pago",
        "afiliados",
    }


def test_budget_batches_have_at_most_four_questions() -> None:
    bank = load_validated_bank()

    batch_counts = Counter(
        question["budget_batch"]
        for question in bank["questions"]
    )

    assert batch_counts
    assert max(
        batch_counts.values()
    ) <= 4


def test_fallback_questions_do_not_expect_sources() -> None:
    bank = load_validated_bank()

    fallback_questions = [
        question
        for question in bank["questions"]
        if question["expected_behavior"] == "fallback"
    ]

    assert len(fallback_questions) == 3

    for question in fallback_questions:
        assert question["expected_documents"] == []
        assert question["expected_pages"] == []
        assert question["should_cite_sources"] is False
        assert question["should_have_answer"] is False


def test_answer_questions_have_expected_evidence() -> None:
    bank = load_validated_bank()

    answer_questions = [
        question
        for question in bank["questions"]
        if question["expected_behavior"] == "answer"
    ]

    assert answer_questions

    for question in answer_questions:
        assert question["expected_documents"]
        assert question["expected_pages"]
        assert question["expected_facts"]
        assert question["should_cite_sources"] is True
        assert question["should_have_answer"] is True


def test_daily_budget_policy_recommends_four_questions() -> None:
    bank = load_validated_bank()
    policy = bank["evaluation_policy"]

    assert policy["daily_gemini_call_limit"] == 20
    assert (
        policy["recommended_max_generated_questions_per_day"]
        == 4
    )
    assert policy["offline_validation_uses_gemini"] is False
