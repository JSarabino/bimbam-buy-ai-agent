"""Configuración central de BimBam Assistant.

Este módulo:

1. Localiza la raíz del proyecto.
2. Carga las variables del archivo .env.
3. Convierte las rutas relativas en rutas absolutas.
4. Convierte los valores numéricos al tipo correcto.
5. Valida configuraciones básicas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# ==========================================================
# 1. Raíz del proyecto
# ==========================================================

# Archivo actual:
# bimbam_assistant/core/config.py
#
# parents[0] = bimbam_assistant/core
# parents[1] = bimbam_assistant
# parents[2] = raíz del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================
# 2. Carga del archivo .env
# ==========================================================

ENV_FILE = PROJECT_ROOT / ".env"

# override=False significa que las variables proporcionadas por
# Docker, OCI o el sistema operativo tienen prioridad sobre .env.
load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ==========================================================
# 3. Error de configuración
# ==========================================================

class ConfigurationError(ValueError):
    """Representa un valor de configuración inválido."""


# ==========================================================
# 4. Funciones auxiliares
# ==========================================================

def _read_integer(variable_name: str, default: int) -> int:
    """Lee una variable de entorno y la convierte en entero."""

    raw_value = os.getenv(variable_name, str(default))

    try:
        return int(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"{variable_name} debe contener un número entero. "
            f"Valor recibido: {raw_value!r}."
        ) from error


def _read_float(variable_name: str, default: float) -> float:
    """Lee una variable de entorno y la convierte en decimal."""

    raw_value = os.getenv(variable_name, str(default))

    try:
        return float(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"{variable_name} debe contener un número decimal. "
            f"Valor recibido: {raw_value!r}."
        ) from error


def _resolve_project_path(
    variable_name: str,
    default: str,
) -> Path:
    """Resuelve una ruta relativa desde la raíz del proyecto."""

    configured_value = os.getenv(variable_name, default).strip()

    path = Path(configured_value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ==========================================================
# 5. Modelo de configuración
# ==========================================================

@dataclass(frozen=True)
class Settings:
    """Contiene la configuración utilizada por la aplicación."""

    # Información de la aplicación
    app_name: str
    app_environment: str
    app_version: str

    # Google Gemini
    google_api_key: str | None
    gemini_chat_model: str
    gemini_embedding_model: str
    gemini_temperature: float

    # Rutas
    project_root: Path
    documents_path: Path
    faiss_index_path: Path

    # Fragmentación
    chunk_size: int
    chunk_overlap: int

    # Recuperación
    retrieval_k: int
    retrieval_score_threshold: float

    @property
    def google_api_key_configured(self) -> bool:
        """Indica si la clave de Gemini tiene un valor."""

        return bool(
            self.google_api_key
            and self.google_api_key.strip()
        )
    
    @property
    def faiss_index_exists(self) -> bool:
        """Comprueba si el almacén vectorial está completo."""

        return (
            (self.faiss_index_path / "index.faiss").is_file()
            and (self.faiss_index_path / "documents.json").is_file()
            and (self.faiss_index_path / "manifest.json").is_file()
        )

    def validate(self) -> None:
        """Valida relaciones entre los valores configurados."""

        if self.chunk_size <= 0:
            raise ConfigurationError(
                "CHUNK_SIZE debe ser mayor que cero."
            )

        if self.chunk_overlap < 0:
            raise ConfigurationError(
                "CHUNK_OVERLAP no puede ser negativo."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP debe ser menor que CHUNK_SIZE."
            )

        if self.retrieval_k <= 0:
            raise ConfigurationError(
                "RETRIEVAL_K debe ser mayor que cero."
            )

        if not 0 <= self.retrieval_score_threshold <= 1:
            raise ConfigurationError(
                "RETRIEVAL_SCORE_THRESHOLD debe estar entre 0 y 1."
            )

    def require_documents_path(self) -> Path:
        """Comprueba que la carpeta de documentos exista."""

        if not self.documents_path.exists():
            raise ConfigurationError(
                "No se encontró la carpeta de documentos: "
                f"{self.documents_path}"
            )

        if not self.documents_path.is_dir():
            raise ConfigurationError(
                "DOCUMENTS_PATH no apunta a una carpeta: "
                f"{self.documents_path}"
            )

        return self.documents_path

    def require_google_api_key(self) -> str:
        """Devuelve la clave o genera un error si no está configurada."""

        if not self.google_api_key_configured:
            raise ConfigurationError(
                "GOOGLE_API_KEY no está configurada."
            )

        return str(self.google_api_key)


# ==========================================================
# 6. Construcción de la configuración
# ==========================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Construye y devuelve la configuración del proyecto."""

    settings = Settings(
        app_name=os.getenv(
            "APP_NAME",
            "BimBam Assistant",
        ).strip(),
        app_environment=os.getenv(
            "APP_ENV",
            "development",
        ).strip(),
        app_version=os.getenv(
            "APP_VERSION",
            "0.1.0",
        ).strip(),
        google_api_key=(
            os.getenv("GOOGLE_API_KEY", "").strip() or None
        ),
        gemini_chat_model=os.getenv(
            "GEMINI_CHAT_MODEL",
            "gemini-3.5-flash",
        ).strip(),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001",
        ).strip(),
        gemini_temperature=_read_float(
            "GEMINI_TEMPERATURE",
            0.0,
        ),
        project_root=PROJECT_ROOT,
        documents_path=_resolve_project_path(
            "DOCUMENTS_PATH",
            "data/documents",
        ),
        faiss_index_path=_resolve_project_path(
            "FAISS_INDEX_PATH",
            "storage/faiss_index",
        ),
        chunk_size=_read_integer(
            "CHUNK_SIZE",
            1000,
        ),
        chunk_overlap=_read_integer(
            "CHUNK_OVERLAP",
            150,
        ),
        retrieval_k=_read_integer(
            "RETRIEVAL_K",
            4,
        ),
        retrieval_score_threshold=_read_float(
            "RETRIEVAL_SCORE_THRESHOLD",
            0.30,
        ),
    )

    settings.validate()

    return settings