"""Modelos de dominio utilizados por BimBam Assistant."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Fragmento recuperado mediante búsqueda semántica."""

    rank: int = Field(
        ge=1,
        description="Posición del resultado dentro de la búsqueda.",
    )

    vector_id: int = Field(
        ge=0,
        description="Posición del vector dentro del índice FAISS.",
    )

    score: float = Field(
        description="Similitud coseno entre la consulta y el chunk.",
    )

    page_content: str = Field(
        min_length=1,
        description="Contenido textual del chunk recuperado.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos originales del documento y la página.",
    )


class RetrievalResponse(BaseModel):
    """Resultado completo de una operación de recuperación."""

    query: str = Field(
        min_length=1,
        description="Pregunta normalizada del usuario.",
    )

    results: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks recuperados y ordenados por relevancia.",
    )

    context: str = Field(
        default="",
        description="Contexto ensamblado para el modelo generativo.",
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros de metadatos aplicados.",
    )

    @property
    def has_results(self) -> bool:
        """Indica si se recuperó al menos un fragmento."""

        return bool(self.results)

class AnswerVerification(BaseModel):
    """Resultado de la verificación automática de una respuesta."""

    status: Literal[
        "verified",
        "rejected",
        "not_applicable",
    ]

    passed: bool = Field(
        description=(
            "Indica si la respuesta superó todos los controles."
        ),
    )

    semantic_supported: bool = Field(
        description=(
            "Indica si las afirmaciones están respaldadas "
            "por el contexto."
        ),
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Confianza asignada por el verificador.",
    )

    citations_present: bool = Field(
        description="Indica si la respuesta contiene citas.",
    )

    cited_sources: list[int] = Field(
        default_factory=list,
    )

    invalid_citations: list[int] = Field(
        default_factory=list,
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    explanation: str = Field(
        description="Explicación resumida de la verificación.",
    )


class SupportContact(BaseModel):
    """Contacto alternativo utilizado por la demostración."""

    area: str
    email: str

    is_demo: bool = Field(
        default=True,
        description=(
            "Indica que el contacto es ficticio y solo se usa "
            "para demostración."
        ),
    )
    
class RagResponse(BaseModel):
    """Respuesta generada mediante recuperación aumentada."""

    query: str = Field(
        min_length=1,
        description="Pregunta normalizada del usuario.",
    )

    answer: str = Field(
        min_length=1,
        description="Respuesta generada a partir del contexto.",
    )

    retrieval: RetrievalResponse = Field(
        description="Resultado de recuperación utilizado.",
    )

    model_name: str = Field(
        min_length=1,
        description="Modelo generativo utilizado.",
    )

    used_context: bool = Field(
        description=(
            "Indica si existía evidencia documental para generar "
            "la respuesta."
        ),
    )
    
    verification: AnswerVerification

    support_contact: SupportContact | None = None

    @property
    def has_sources(self) -> bool:
        """Indica si la respuesta contiene fuentes recuperadas."""

        return bool(
            self.retrieval.results
        )

    @property
    def sources(self) -> list[RetrievedChunk]:
        """Devuelve los chunks utilizados como fuentes."""

        return self.retrieval.results
    
    @property
    def is_verified(self) -> bool:
        """Indica si la respuesta generada fue verificada."""

        return self.verification.status == "verified"