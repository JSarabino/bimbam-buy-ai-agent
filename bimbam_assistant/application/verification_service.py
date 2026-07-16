"""Verificación automática de respuestas generadas por el RAG."""

from __future__ import annotations

import logging
import re

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
        semantic = generate_structured(
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
        raise VerificationError(
            "No fue posible verificar automáticamente "
            "la respuesta."
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