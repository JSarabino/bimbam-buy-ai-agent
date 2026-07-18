"""Pruebas offline de generación y fallback RAG."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bimbam_assistant.application import rag_service
from bimbam_assistant.application.rag_service import (
    NO_EVIDENCE_ANSWER,
    UNVERIFIED_ANSWER,
    RagGenerationError,
    append_demo_contact,
    answer_question,
    build_generation_prompt,
)
from bimbam_assistant.application.verification_service import (
    VerificationError,
)
from bimbam_assistant.domain.models import (
    AnswerVerification,
    RetrievedChunk,
    RetrievalResponse,
    SupportContact,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    GeminiChatError,
)


def build_retrieval(
    *,
    with_results: bool,
) -> RetrievalResponse:
    if not with_results:
        return RetrievalResponse(
            query="¿Cuánto tarda un reembolso?",
            results=[],
            context="",
            filters={},
        )

    chunk = RetrievedChunk(
        rank=1,
        vector_id=0,
        score=0.91,
        page_content=(
            "El reembolso tarda entre 5 y 10 días hábiles."
        ),
        metadata={
            "document_name": (
                "Política de Reembolsos y Devoluciones de BimBam"
            ),
            "page_number": 4,
            "category": "reembolsos_devoluciones",
            "chunk_id": "refund-page-4-chunk-0",
        },
    )

    return RetrievalResponse(
        query="¿Cuánto tarda un reembolso?",
        results=[chunk],
        context=(
            "[Fuente 1]\n"
            "Documento: Política de Reembolsos\n"
            "Página: 4\n\n"
            "El reembolso tarda entre 5 y 10 días hábiles."
        ),
        filters={},
    )


def build_verification(
    *,
    passed: bool,
) -> AnswerVerification:
    return AnswerVerification(
        status=(
            "verified"
            if passed
            else "rejected"
        ),
        passed=passed,
        semantic_supported=passed,
        confidence=(
            0.98
            if passed
            else 0.20
        ),
        citations_present=True,
        cited_sources=[1],
        invalid_citations=[],
        unsupported_claims=(
            []
            if passed
            else ["Afirmación no respaldada"]
        ),
        explanation=(
            "La respuesta está respaldada."
            if passed
            else "La respuesta no está respaldada."
        ),
    )


def demo_contact() -> SupportContact:
    return SupportContact(
        area="Soporte de demostración",
        email="soporte@example.com",
    )


def test_build_generation_prompt_requires_context() -> None:
    with pytest.raises(
        RagGenerationError,
        match="No existe contexto documental",
    ):
        build_generation_prompt(
            build_retrieval(
                with_results=False
            )
        )


def test_build_generation_prompt_contains_query_context_and_citation_rules() -> None:
    prompt = build_generation_prompt(
        build_retrieval(
            with_results=True
        )
    )

    assert "¿Cuánto tarda un reembolso?" in prompt
    assert "El reembolso tarda entre 5 y 10 días hábiles." in prompt
    assert "[Fuente 1]" in prompt
    assert "No agregues una bibliografía separada" in prompt


def test_answer_question_returns_verified_grounded_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = build_retrieval(
        with_results=True
    )
    verification = build_verification(
        passed=True
    )

    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_chat_model="test-model"
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "retrieve_documents",
        lambda *args, **kwargs: retrieval,
    )

    monkeypatch.setattr(
        rag_service,
        "generate_text",
        lambda **kwargs: (
            "El reembolso tarda entre 5 y 10 días hábiles "
            "[Fuente 1]."
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "verify_answer",
        lambda **kwargs: verification,
    )

    response = answer_question(
        "¿Cuánto tarda un reembolso?"
    )

    assert response.answer.startswith(
        "El reembolso tarda"
    )
    assert response.used_context is True
    assert response.is_verified is True
    assert response.model_name == "test-model"
    assert response.support_contact is None
    assert response.has_sources is True


def test_answer_question_skips_generation_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = build_retrieval(
        with_results=False
    )

    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_chat_model="test-model"
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "retrieve_documents",
        lambda *args, **kwargs: retrieval,
    )

    monkeypatch.setattr(
        rag_service,
        "resolve_support_contact",
        lambda retrieval: demo_contact(),
    )

    monkeypatch.setattr(
        rag_service,
        "generate_text",
        lambda **kwargs: pytest.fail(
            "No debe invocarse Gemini sin evidencia."
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "verify_answer",
        lambda **kwargs: pytest.fail(
            "No debe verificarse una respuesta no generada."
        ),
    )

    response = answer_question(
        "Pregunta sin cobertura"
    )

    assert response.used_context is False
    assert response.verification.status == "not_applicable"
    assert response.verification.passed is True
    assert NO_EVIDENCE_ANSWER in response.answer
    assert "soporte@example.com" in response.answer
    assert response.support_contact is not None


def test_answer_question_replaces_rejected_generation_with_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = build_retrieval(
        with_results=True
    )

    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_chat_model="test-model"
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "retrieve_documents",
        lambda *args, **kwargs: retrieval,
    )

    monkeypatch.setattr(
        rag_service,
        "generate_text",
        lambda **kwargs: (
            "Respuesta inventada [Fuente 1]."
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "verify_answer",
        lambda **kwargs: build_verification(
            passed=False
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "resolve_support_contact",
        lambda retrieval: demo_contact(),
    )

    response = answer_question(
        "¿Cuánto tarda un reembolso?"
    )

    assert UNVERIFIED_ANSWER in response.answer
    assert "Respuesta inventada" not in response.answer
    assert response.used_context is True
    assert response.verification.status == "rejected"
    assert response.support_contact is not None


def test_answer_question_wraps_chat_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_chat_model="test-model"
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "retrieve_documents",
        lambda *args, **kwargs: build_retrieval(
            with_results=True
        ),
    )

    def raise_chat_error(
        **kwargs: object,
    ) -> str:
        raise GeminiChatError(
            "chat failure"
        )

    monkeypatch.setattr(
        rag_service,
        "generate_text",
        raise_chat_error,
    )

    with pytest.raises(
        RagGenerationError,
        match="No fue posible generar la respuesta final",
    ):
        answer_question(
            "¿Cuánto tarda un reembolso?"
        )


def test_answer_question_wraps_verification_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_chat_model="test-model"
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "retrieve_documents",
        lambda *args, **kwargs: build_retrieval(
            with_results=True
        ),
    )

    monkeypatch.setattr(
        rag_service,
        "generate_text",
        lambda **kwargs: (
            "Respuesta con fuente [Fuente 1]."
        ),
    )

    def raise_verification_error(
        **kwargs: object,
    ) -> AnswerVerification:
        raise VerificationError(
            "verification failure"
        )

    monkeypatch.setattr(
        rag_service,
        "verify_answer",
        raise_verification_error,
    )

    with pytest.raises(
        RagGenerationError,
        match="no pudo verificarse",
    ):
        answer_question(
            "¿Cuánto tarda un reembolso?"
        )


def test_append_demo_contact_marks_contact_as_fictitious() -> None:
    message = append_demo_contact(
        "Mensaje base.",
        demo_contact(),
    )

    assert "Mensaje base." in message
    assert "soporte@example.com" in message
    assert "ficticio" in message
    assert "demostración" in message
