"""Verificación automática de respuestas generadas por el RAG."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, Field

from bimbam_assistant.domain.models import (
    AnswerVerification,
    RetrievalResponse,
)
from bimbam_assistant.infrastructure.gemini_provider import (
    GeminiChatError,
    generate_structured,
)


logger = logging.getLogger(__name__)


MIN_VERIFICATION_CONFIDENCE = 0.75

VERIFICATION_MAX_ATTEMPTS = 2
VERIFICATION_RETRY_BASE_SECONDS = 2.0

TRANSIENT_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline exceeded",
    "internal server error",
    "rate limit",
    "resource exhausted",
    "resource_exhausted",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "too many requests",
)

CITATION_PATTERN = re.compile(
    r"Fuente\s+(\d+)",
    flags=re.IGNORECASE,
)


class VerificationError(RuntimeError):
    """Error producido durante la verificación automática."""


class SemanticVerification(BaseModel):
    """Evaluación semántica devuelta por el modelo verificador."""

    is_supported: bool = Field(
        description=(
            "Verdadero solo cuando todas las afirmaciones factuales "
            "están respaldadas por el contexto."
        ),
    )

    confidence: float = Field(
        ge=0,
        le=1,
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    explanation: str


StructuredModelT = TypeVar(
    "StructuredModelT",
    bound=BaseModel,
)


VERIFICATION_SYSTEM_INSTRUCTION = """
Eres un verificador estricto de respuestas RAG.

Debes comparar la respuesta con el contexto documental y determinar
si cada afirmación factual está respaldada explícitamente.

Reglas:

1. No uses conocimiento externo.
2. No evalúes el estilo; evalúa únicamente fidelidad documental.
3. Una respuesta es válida solo si todas sus afirmaciones factuales
   están respaldadas.
4. Una inferencia sencilla es aceptable únicamente cuando se deriva
   directamente del contexto.
5. Marca como no respaldadas las fechas, plazos, condiciones,
   contactos, montos o procedimientos que no aparezcan en el contexto.
6. No consideres que una cita prueba una afirmación si el fragmento
   citado no contiene realmente esa información.
7. Una respuesta breve no es incorrecta por omitir detalles.
""".strip()


def extract_cited_sources(
    answer: str,
) -> list[int]:
    """Extrae números usados en referencias como [Fuente 1]."""

    return sorted(
        {
            int(value)
            for value in CITATION_PATTERN.findall(
                answer
            )
        }
    )


def build_verification_prompt(
    *,
    query: str,
    answer: str,
    retrieval: RetrievalResponse,
) -> str:
    """Construye el prompt usado para validar la respuesta."""

    return (
        "PREGUNTA\n"
        "--------\n"
        f"{query}\n\n"
        "RESPUESTA A EVALUAR\n"
        "-------------------\n"
        f"{answer}\n\n"
        "CONTEXTO DOCUMENTAL AUTORIZADO\n"
        "------------------------------\n"
        f"{retrieval.context}\n\n"
        "Determina si todas las afirmaciones de la respuesta están "
        "respaldadas por este contexto."
    )


def _iter_error_chain(
    error: BaseException,
) -> list[BaseException]:
    """Recorre el error y sus causas sin entrar en ciclos."""

    chain: list[BaseException] = []
    current: BaseException | None = error
    visited: set[int] = set()

    while current is not None:
        current_id = id(current)

        if current_id in visited:
            break

        visited.add(
            current_id
        )

        chain.append(
            current
        )

        current = (
            current.__cause__
            or current.__context__
        )

    return chain


def is_transient_verification_error(
    error: BaseException,
) -> bool:
    """Detecta fallos temporales que justifican un nuevo intento."""

    messages = " | ".join(
        str(item).lower()
        for item in _iter_error_chain(
            error
        )
    )

    return any(
        marker in messages
        for marker in TRANSIENT_ERROR_MARKERS
    )


def generate_structured_with_retry(
    *,
    system_instruction: str,
    user_prompt: str,
    schema: type[StructuredModelT],
    max_attempts: int = VERIFICATION_MAX_ATTEMPTS,
    retry_base_seconds: float = VERIFICATION_RETRY_BASE_SECONDS,
    generator: Callable[..., StructuredModelT] = generate_structured,
) -> StructuredModelT:
    """Genera una salida estructurada con reintento temporal controlado.

    Solo repite errores asociados a límites, indisponibilidad o timeouts.
    Una respuesta estructurada válida, aunque rechace el contenido, no
    se regenera.
    """

    if max_attempts < 1:
        raise ValueError(
            "max_attempts debe ser mayor o igual que 1."
        )

    if retry_base_seconds < 0:
        raise ValueError(
            "retry_base_seconds no puede ser negativo."
        )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            return generator(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                schema=schema,
            )

        except GeminiChatError as error:
            is_last_attempt = (
                attempt >= max_attempts
            )

            transient = (
                is_transient_verification_error(
                    error
                )
            )

            if (
                is_last_attempt
                or not transient
            ):
                raise

            wait_seconds = (
                retry_base_seconds
                * (2 ** (attempt - 1))
            )

            logger.warning(
                "La verificación falló temporalmente en el intento "
                "%s de %s. Nuevo intento en %.1f segundos. Error: %s",
                attempt,
                max_attempts,
                wait_seconds,
                error,
            )

            time.sleep(
                wait_seconds
            )

    raise GeminiChatError(
        "La verificación no produjo un resultado."
    )


def verify_answer(
    *,
    query: str,
    answer: str,
    retrieval: RetrievalResponse,
) -> AnswerVerification:
    """Aplica validación de citas y verificación semántica."""

    if not retrieval.has_results:
        return AnswerVerification(
            status="not_applicable",
            passed=True,
            semantic_supported=True,
            confidence=1.0,
            citations_present=False,
            cited_sources=[],
            invalid_citations=[],
            unsupported_claims=[],
            explanation=(
                "No se generó una respuesta con el modelo porque "
                "no existía evidencia documental."
            ),
        )

    cited_sources = extract_cited_sources(
        answer
    )

    valid_sources = {
        result.rank
        for result in retrieval.results
    }

    invalid_citations = sorted(
        set(cited_sources) - valid_sources
    )

    citations_present = bool(
        cited_sources
    )

    try:
        semantic = generate_structured_with_retry(
            system_instruction=(
                VERIFICATION_SYSTEM_INSTRUCTION
            ),
            user_prompt=build_verification_prompt(
                query=query,
                answer=answer,
                retrieval=retrieval,
            ),
            schema=SemanticVerification,
        )

    except GeminiChatError as error:
        transient = (
            is_transient_verification_error(
                error
            )
        )

        detail = (
            "después de agotar los reintentos"
            if transient
            else "por un error no recuperable"
        )

        raise VerificationError(
            "No fue posible verificar automáticamente "
            f"la respuesta {detail}."
        ) from error

    passed = (
        semantic.is_supported
        and semantic.confidence
        >= MIN_VERIFICATION_CONFIDENCE
        and citations_present
        and not invalid_citations
    )

    status = (
        "verified"
        if passed
        else "rejected"
    )

    logger.info(
        "Verificación finalizada: estado=%s, confianza=%.2f, "
        "citas=%s, citas inválidas=%s.",
        status,
        semantic.confidence,
        cited_sources,
        invalid_citations,
    )

    return AnswerVerification(
        status=status,
        passed=passed,
        semantic_supported=semantic.is_supported,
        confidence=semantic.confidence,
        citations_present=citations_present,
        cited_sources=cited_sources,
        invalid_citations=invalid_citations,
        unsupported_claims=semantic.unsupported_claims,
        explanation=semantic.explanation,
    )
