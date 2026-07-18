"""Proveedor de inteligencia artificial de Google Gemini.

Este módulo centraliza la integración de BimBam Assistant con Gemini
y se encarga de:

1. Crear y reutilizar los clientes de embeddings y chat.
2. Obtener los modelos, la clave y los parámetros desde la configuración.
3. Generar embeddings para documentos y consultas.
4. Generar respuestas de texto para la cadena RAG.
5. Generar respuestas estructuradas validadas mediante modelos Pydantic.
6. Validar la cantidad, dimensión y contenido de las respuestas recibidas.
7. Procesar embeddings por lotes y controlar reintentos ante límites de uso.
8. Traducir los errores del proveedor a excepciones propias de la aplicación.

La creación, persistencia y consulta del índice vectorial se implementan
en faiss_store.py.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import TypeVar
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from bimbam_assistant.core.config import (
    ConfigurationError,
    get_settings,
)


logger = logging.getLogger(__name__)


# Evita enviar todos los chunks en una única operación desde nuestra capa.
# El valor puede ajustarse posteriormente si el corpus crece.
DEFAULT_EMBEDDING_BATCH_SIZE = 20
DEFAULT_BATCH_DELAY_SECONDS = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SECONDS = 65


class GeminiEmbeddingError(RuntimeError):
    """Error producido durante la generación de embeddings."""
    
class GeminiChatError(RuntimeError):
    """Error producido al generar texto con Gemini."""
    
StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)

def _normalize_model_name(model_name: str) -> str:
    """Normaliza el identificador configurado para el modelo.

    Permite utilizar cualquiera de estas dos formas en .env:

        gemini-embedding-001
        models/gemini-embedding-001

    Internamente se conserva la forma sin el prefijo ``models/``.
    """

    normalized_name = model_name.strip()

    if normalized_name.startswith("models/"):
        normalized_name = normalized_name.removeprefix("models/")

    if not normalized_name:
        raise GeminiEmbeddingError(
            "GEMINI_EMBEDDING_MODEL no puede estar vacío."
        )

    return normalized_name


def _prepare_texts(
    texts: Sequence[str],
) -> list[str]:
    """Limpia y valida los textos antes de generar embeddings.

    No se eliminan elementos vacíos silenciosamente porque eso rompería
    la correspondencia entre la posición del chunk y la posición de su
    vector dentro del índice.
    """

    prepared_texts: list[str] = []

    for position, text in enumerate(texts):
        if not isinstance(text, str):
            raise GeminiEmbeddingError(
                "Todos los elementos deben ser texto. "
                f"Elemento inválido en la posición {position}."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise GeminiEmbeddingError(
                "No es posible generar un embedding para un texto vacío. "
                f"Posición recibida: {position}."
            )

        prepared_texts.append(cleaned_text)

    return prepared_texts


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> int:
    """Valida cantidad y dimensionalidad de los vectores.

    Devuelve la dimensión común de los embeddings.
    """

    if len(vectors) != expected_count:
        raise GeminiEmbeddingError(
            "Gemini devolvió una cantidad inesperada de embeddings. "
            f"Esperados: {expected_count}. "
            f"Recibidos: {len(vectors)}."
        )

    if not vectors:
        raise GeminiEmbeddingError(
            "Gemini no devolvió embeddings."
        )

    embedding_dimension = len(vectors[0])

    if embedding_dimension == 0:
        raise GeminiEmbeddingError(
            "Gemini devolvió un embedding vacío."
        )

    for position, vector in enumerate(vectors):
        if len(vector) != embedding_dimension:
            raise GeminiEmbeddingError(
                "Los embeddings no tienen una dimensión consistente. "
                f"Vector en posición {position}: {len(vector)}. "
                f"Dimensión esperada: {embedding_dimension}."
            )

    return embedding_dimension


@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Crea y reutiliza el cliente de embeddings de Gemini.

    La clave se obtiene desde Settings y nunca se imprime ni se almacena
    dentro de los metadatos documentales.
    """

    try:
        settings = get_settings()
        google_api_key = settings.require_google_api_key()

        model_name = _normalize_model_name(
            settings.gemini_embedding_model
        )

        model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=google_api_key,
        )

    except ConfigurationError as error:
        raise GeminiEmbeddingError(
            "No fue posible configurar el proveedor de embeddings: "
            f"{error}"
        ) from error

    except Exception as error:
        raise GeminiEmbeddingError(
            "No fue posible crear el cliente de embeddings de Gemini."
        ) from error

    logger.info(
        "Proveedor de embeddings configurado con el modelo %s.",
        model_name,
    )

    return model

@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Construye y reutiliza el modelo de chat configurado."""

    settings = get_settings()

    api_key = settings.require_google_api_key()

    logger.info(
        "Proveedor de chat configurado con el modelo %s.",
        settings.gemini_chat_model,
    )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        api_key=api_key,
        temperature=settings.gemini_temperature,
    )

def _extract_generated_text(
    content: object,
) -> str:
    """Extrae texto de la respuesta devuelta por Gemini."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(
                    block
                )

            elif isinstance(block, dict):
                block_text = block.get(
                    "text"
                )

                if isinstance(block_text, str):
                    text_parts.append(
                        block_text
                    )

        return "\n".join(
            text_parts
        ).strip()

    return ""

def generate_text(
    *,
    system_instruction: str,
    user_prompt: str,
) -> str:
    """Genera una respuesta de texto con Gemini."""

    normalized_system_instruction = (
        system_instruction.strip()
    )

    normalized_user_prompt = (
        user_prompt.strip()
    )

    if not normalized_system_instruction:
        raise GeminiChatError(
            "La instrucción del sistema no puede estar vacía."
        )

    if not normalized_user_prompt:
        raise GeminiChatError(
            "El prompt del usuario no puede estar vacío."
        )

    model = get_chat_model()

    try:
        response = model.invoke(
            [
                (
                    "system",
                    normalized_system_instruction,
                ),
                (
                    "human",
                    normalized_user_prompt,
                ),
            ]
        )

    except Exception as error:
        raise GeminiChatError(
            "Gemini no pudo generar la respuesta. Comprueba "
            "el modelo configurado, la conexión y los límites "
            "de uso de la API."
        ) from error

    generated_text = _extract_generated_text(
        response.content
    )

    if not generated_text:
        raise GeminiChatError(
            "Gemini devolvió una respuesta sin contenido textual."
        )

    logger.info(
        "Respuesta generada con el modelo %s.",
        get_settings().gemini_chat_model,
    )

    return generated_text

def generate_structured(
    *,
    system_instruction: str,
    user_prompt: str,
    schema: type[StructuredOutputT],
) -> StructuredOutputT:
    """Genera y valida una respuesta estructurada con Gemini."""

    normalized_system_instruction = (
        system_instruction.strip()
    )

    normalized_user_prompt = (
        user_prompt.strip()
    )

    if not normalized_system_instruction:
        raise GeminiChatError(
            "La instrucción del sistema no puede estar vacía."
        )

    if not normalized_user_prompt:
        raise GeminiChatError(
            "El prompt del usuario no puede estar vacío."
        )

    model = get_chat_model()

    try:
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
        )

        response = structured_model.invoke(
            [
                (
                    "system",
                    normalized_system_instruction,
                ),
                (
                    "human",
                    normalized_user_prompt,
                ),
            ]
        )

        if isinstance(response, schema):
            return response

        return schema.model_validate(
            response
        )

    except Exception as error:
        raise GeminiChatError(
            "Gemini no pudo generar o validar la respuesta "
            "estructurada."
        ) from error

def _is_rate_limit_error(error: Exception) -> bool:
    """Indica si el error corresponde a un límite de uso de Gemini."""

    status_code = getattr(
        error,
        "status_code",
        None,
    )

    error_code = getattr(
        error,
        "code",
        None,
    )

    error_message = str(error).upper()

    return (
        status_code == 429
        or error_code == 429
        or "429" in error_message
        or "RESOURCE_EXHAUSTED" in error_message
        or "TOO MANY REQUESTS" in error_message
    )

def embed_documents(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    batch_delay_seconds: float = DEFAULT_BATCH_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
) -> list[list[float]]:
    """Genera embeddings por lotes con control de límites de uso.

    Si Gemini devuelve un error 429, se vuelve a intentar únicamente
    el lote que falló. Los vectores generados por lotes anteriores
    permanecen en memoria durante la ejecución.
    """

    if batch_size <= 0:
        raise GeminiEmbeddingError(
            "batch_size debe ser mayor que cero."
        )

    if batch_delay_seconds < 0:
        raise GeminiEmbeddingError(
            "batch_delay_seconds no puede ser negativo."
        )

    if max_retries < 0:
        raise GeminiEmbeddingError(
            "max_retries no puede ser negativo."
        )

    if retry_base_seconds <= 0:
        raise GeminiEmbeddingError(
            "retry_base_seconds debe ser mayor que cero."
        )

    prepared_texts = _prepare_texts(texts)

    if not prepared_texts:
        return []

    model = get_embedding_model()
    vectors: list[list[float]] = []

    for start in range(
        0,
        len(prepared_texts),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(prepared_texts),
        )

        batch = prepared_texts[start:end]

        logger.info(
            "Generando embeddings para textos %s a %s de %s.",
            start + 1,
            end,
            len(prepared_texts),
        )

        batch_vectors: list[list[float]] | None = None

        for attempt in range(max_retries + 1):
            try:
                batch_vectors = model.embed_documents(
                    batch
                )

                _validate_vectors(
                    batch_vectors,
                    expected_count=len(batch),
                )

                break

            except GeminiEmbeddingError:
                raise

            except Exception as error:
                rate_limit_reached = _is_rate_limit_error(
                    error
                )

                last_attempt = attempt >= max_retries

                if not rate_limit_reached or last_attempt:
                    raise GeminiEmbeddingError(
                        "Gemini no pudo generar los embeddings del "
                        f"lote {start + 1}-{end}. Comprueba la clave, "
                        "el modelo, la conexión y los límites de uso "
                        "de la API."
                    ) from error

                waiting_seconds = (
                    retry_base_seconds
                    * (2 ** attempt)
                )

                logger.warning(
                    "Gemini alcanzó el límite de uso en el lote "
                    "%s-%s. Reintento %s de %s en %.0f segundos.",
                    start + 1,
                    end,
                    attempt + 1,
                    max_retries,
                    waiting_seconds,
                )

                time.sleep(
                    waiting_seconds
                )

        if batch_vectors is None:
            raise GeminiEmbeddingError(
                "No fue posible obtener los embeddings del lote "
                f"{start + 1}-{end}."
            )

        vectors.extend(
            batch_vectors
        )

        # Espera entre lotes, excepto después del último.
        if end < len(prepared_texts):
            logger.info(
                "Esperando %.0f segundos antes del siguiente lote.",
                batch_delay_seconds,
            )

            time.sleep(
                batch_delay_seconds
            )

    embedding_dimension = _validate_vectors(
        vectors,
        expected_count=len(prepared_texts),
    )

    logger.info(
        "Embeddings generados: %s vectores de %s dimensiones.",
        len(vectors),
        embedding_dimension,
    )

    return vectors


def embed_query(query: str) -> list[float]:
    """Genera el embedding de una consulta del usuario.

    LangChain utiliza internamente RETRIEVAL_QUERY cuando se llama
    a ``embed_query`` y no se establece un task_type global.
    """

    prepared_queries = _prepare_texts([query])
    model = get_embedding_model()

    try:
        vector = model.embed_query(
            prepared_queries[0]
        )

    except Exception as error:
        raise GeminiEmbeddingError(
            "Gemini no pudo generar el embedding de la consulta. "
            "Comprueba la clave, el modelo, la conexión y los límites "
            "de uso de la API."
        ) from error

    _validate_vectors(
        [vector],
        expected_count=1,
    )

    logger.info(
        "Embedding de consulta generado con %s dimensiones.",
        len(vector),
    )

    return vector