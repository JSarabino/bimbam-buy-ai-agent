"""Pruebas del reintento automático de la verificación."""

from __future__ import annotations

import pytest

from bimbam_assistant.application import verification_service
from bimbam_assistant.application.verification_service import (
    SemanticVerification,
    VerificationError,
    generate_structured_with_retry,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    GeminiChatError,
)


def raise_transient_error() -> None:
    """Genera un error con causa temporal reconocible."""

    try:
        raise RuntimeError(
            "503 Service Unavailable"
        )
    except RuntimeError as root_error:
        raise GeminiChatError(
            "Gemini no pudo verificar la respuesta."
        ) from root_error


def raise_permanent_error() -> None:
    """Genera un error que no debe reintentarse."""

    try:
        raise ValueError(
            "El esquema de salida es inválido."
        )
    except ValueError as root_error:
        raise GeminiChatError(
            "Gemini no pudo validar la salida."
        ) from root_error


def test_retries_once_after_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    waits: list[float] = []

    def fake_generator(**_: object) -> SemanticVerification:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise_transient_error()

        return SemanticVerification(
            is_supported=True,
            confidence=0.95,
            unsupported_claims=[],
            explanation="La respuesta está respaldada.",
        )

    monkeypatch.setattr(
        verification_service.time,
        "sleep",
        waits.append,
    )

    result = generate_structured_with_retry(
        system_instruction="Verifica.",
        user_prompt="Pregunta y contexto.",
        schema=SemanticVerification,
        max_attempts=2,
        retry_base_seconds=0.5,
        generator=fake_generator,
    )

    assert attempts == 2
    assert waits == [0.5]
    assert result.is_supported is True
    assert result.confidence == 0.95


def test_does_not_retry_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    waits: list[float] = []

    def fake_generator(**_: object) -> SemanticVerification:
        nonlocal attempts
        attempts += 1
        raise_permanent_error()

    monkeypatch.setattr(
        verification_service.time,
        "sleep",
        waits.append,
    )

    with pytest.raises(
        GeminiChatError
    ):
        generate_structured_with_retry(
            system_instruction="Verifica.",
            user_prompt="Pregunta y contexto.",
            schema=SemanticVerification,
            max_attempts=3,
            retry_base_seconds=0.5,
            generator=fake_generator,
        )

    assert attempts == 1
    assert waits == []


def test_raises_after_exhausting_transient_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    waits: list[float] = []

    def fake_generator(**_: object) -> SemanticVerification:
        nonlocal attempts
        attempts += 1
        raise_transient_error()

    monkeypatch.setattr(
        verification_service.time,
        "sleep",
        waits.append,
    )

    with pytest.raises(
        GeminiChatError
    ):
        generate_structured_with_retry(
            system_instruction="Verifica.",
            user_prompt="Pregunta y contexto.",
            schema=SemanticVerification,
            max_attempts=3,
            retry_base_seconds=0.25,
            generator=fake_generator,
        )

    assert attempts == 3
    assert waits == [
        0.25,
        0.5,
    ]


def test_verify_answer_wraps_exhausted_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bimbam_assistant.domain.models import (
        RetrievalResponse,
        RetrievedChunk,
    )

    retrieval = RetrievalResponse(
        query="¿Cuánto tarda un reembolso?",
        results=[
            RetrievedChunk(
                rank=1,
                vector_id=0,
                score=0.9,
                page_content=(
                    "El reembolso tarda entre 5 y 10 días hábiles."
                ),
                metadata={
                    "document_name": "Política de reembolsos",
                    "page_number": 4,
                },
            )
        ],
        context=(
            "[Fuente 1]\n"
            "El reembolso tarda entre 5 y 10 días hábiles."
        ),
        filters={},
    )

    monkeypatch.setattr(
        verification_service.time,
        "sleep",
        lambda _: None,
    )

    monkeypatch.setattr(
        verification_service,
        "generate_structured",
        lambda **_: raise_transient_error(),
    )

    # generate_structured_with_retry usa su valor predeterminado capturado
    # al definir la función. Por eso se sustituye la función completa para
    # aislar aquí el comportamiento de verify_answer.
    def always_fail(**_: object) -> SemanticVerification:
        raise_transient_error()

    monkeypatch.setattr(
        verification_service,
        "generate_structured_with_retry",
        always_fail,
    )

    with pytest.raises(
        VerificationError,
        match="agotar los reintentos",
    ):
        verification_service.verify_answer(
            query=retrieval.query,
            answer=(
                "El reembolso tarda de 5 a 10 días hábiles "
                "[Fuente 1]."
            ),
            retrieval=retrieval,
        )
