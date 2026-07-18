"""Pruebas offline del ejecutor de evaluación RAG."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluate_rag import (
    BudgetExceededError,
    MODE_FULL,
    MODE_RETRIEVAL,
    build_plan,
    evaluate_full_response,
    evaluate_retrieval,
    load_budget_ledger,
    main,
    reserve_budget,
    select_questions,
)


def sample_questions() -> list[dict]:
    return [
        {
            "id": "ENV-001",
            "question": "¿Cuánto tarda?",
            "category": "envios",
            "budget_batch": "A",
            "evaluation_tier": "smoke",
            "expected_behavior": "answer",
            "expected_documents": [
                "Guía de Envíos"
            ],
            "expected_pages": [
                3
            ],
            "expected_facts": [
                "Seis días."
            ],
            "forbidden_facts": [],
        },
        {
            "id": "OUT-001",
            "question": "¿Cuál es el teléfono real?",
            "category": "fuera_de_alcance",
            "budget_batch": "B",
            "evaluation_tier": "full",
            "expected_behavior": "fallback",
            "expected_documents": [],
            "expected_pages": [],
            "expected_facts": [
                "No está disponible."
            ],
            "forbidden_facts": [],
        },
    ]


def test_select_questions_filters_by_batch() -> None:
    selected = select_questions(
        sample_questions(),
        batch="a",
    )

    assert [
        question["id"]
        for question in selected
    ] == [
        "ENV-001"
    ]


def test_retrieval_plan_reserves_one_call_per_question(
    tmp_path: Path,
) -> None:
    plan = build_plan(
        questions=sample_questions(),
        mode=MODE_RETRIEVAL,
        ledger_path=tmp_path / "budget.json",
        daily_limit=20,
        safety_buffer=2,
    )

    assert plan.calls_per_question == 1
    assert plan.reserved_calls == 2
    assert plan.can_execute is True


def test_full_plan_reserves_four_calls_per_question(
    tmp_path: Path,
) -> None:
    plan = build_plan(
        questions=sample_questions(),
        mode=MODE_FULL,
        ledger_path=tmp_path / "budget.json",
        daily_limit=20,
        safety_buffer=2,
    )

    assert plan.calls_per_question == 4
    assert plan.reserved_calls == 8
    assert plan.can_execute is True


def test_budget_blocks_calls_above_protected_limit(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "budget.json"

    for position in range(4):
        reserve_budget(
            ledger_path=ledger_path,
            question_id=f"Q-{position}",
            mode=MODE_FULL,
            calls=4,
            daily_limit=20,
            safety_buffer=2,
        )

    with pytest.raises(
        BudgetExceededError
    ):
        reserve_budget(
            ledger_path=ledger_path,
            question_id="Q-5",
            mode=MODE_FULL,
            calls=4,
            daily_limit=20,
            safety_buffer=2,
        )

    ledger = load_budget_ledger(
        ledger_path,
        daily_limit=20,
        safety_buffer=2,
    )

    assert ledger["reserved_calls"] == 16


def test_evaluate_retrieval_detects_document_and_page() -> None:
    retrieval = SimpleNamespace(
        results=[
            SimpleNamespace(
                score=0.88,
                metadata={
                    "document_name": "Guía de Envíos",
                    "page_number": 3,
                    "category": "envios",
                },
            )
        ]
    )

    metrics = evaluate_retrieval(
        question=sample_questions()[0],
        retrieval=retrieval,
    )

    assert metrics["document_hit_any"] is True
    assert metrics["document_hit_all"] is True
    assert metrics["page_hit_any"] is True
    assert metrics["category_hit"] is True


def test_evaluate_full_response_uses_existing_verification() -> None:
    retrieval = SimpleNamespace(
        results=[
            SimpleNamespace(
                score=0.91,
                metadata={
                    "document_name": "Guía de Envíos",
                    "page_number": 3,
                    "category": "envios",
                },
            )
        ]
    )

    verification = SimpleNamespace(
        passed=True,
        status="verified",
        confidence=0.95,
        invalid_citations=[],
        unsupported_claims=[],
    )

    response = SimpleNamespace(
        answer="La entrega tarda seis días [Fuente 1].",
        retrieval=retrieval,
        verification=verification,
        used_context=True,
        model_name="test-model",
    )

    metrics = evaluate_full_response(
        question=sample_questions()[0],
        response=response,
    )

    assert metrics["citations_present"] is True
    assert metrics["verification_passed"] is True
    assert metrics["expected_behavior_passed"] is True


def test_dry_run_does_not_call_gemini(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path = tmp_path / "questions.json"

    bank = {
        "schema_version": 1,
        "question_count": 2,
        "questions": [],
    }

    # Se usa un validador falso para aislar la prueba del plan.
    bank["questions"] = sample_questions()

    bank_path.write_text(
        json.dumps(
            bank,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.evaluate_rag.validate_evaluation_bank",
        lambda bank: {
            "questions": 2
        },
    )

    monkeypatch.setattr(
        "scripts.evaluate_rag.retrieve_documents",
        lambda query: pytest.fail(
            "El dry-run no debe llamar a Gemini."
        ),
    )

    exit_code = main(
        [
            "--bank",
            str(
                bank_path
            ),
            "--batch",
            "A",
            "--output-directory",
            str(
                tmp_path / "output"
            ),
        ]
    )

    assert exit_code == 0


def test_execute_requires_a_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path = tmp_path / "questions.json"

    bank_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "question_count": 2,
                "questions": sample_questions(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.evaluate_rag.validate_evaluation_bank",
        lambda bank: {
            "questions": 2
        },
    )

    exit_code = main(
        [
            "--bank",
            str(
                bank_path
            ),
            "--execute",
            "--output-directory",
            str(
                tmp_path / "output"
            ),
        ]
    )

    assert exit_code == 1
