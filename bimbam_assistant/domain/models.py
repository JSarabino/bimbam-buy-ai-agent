"""Modelos de dominio utilizados por BimBam Assistant."""

from __future__ import annotations

from typing import Any

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