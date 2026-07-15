"""Almacenamiento vectorial local mediante FAISS.

Este módulo permite:

1. Crear un índice FAISS a partir de embeddings.
2. Guardar el índice, los chunks y un manifiesto.
3. Cargar un índice previamente generado.
4. Buscar chunks mediante similitud coseno.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from bimbam_assistant.core.config import get_settings


INDEX_FILE_NAME = "index.faiss"
DOCUMENTS_FILE_NAME = "documents.json"
MANIFEST_FILE_NAME = "manifest.json"


class FaissStoreError(RuntimeError):
    """Error relacionado con el almacenamiento vectorial."""


@dataclass(frozen=True)
class SearchResult:
    """Resultado de una búsqueda vectorial."""

    vector_id: int
    score: float
    document: Document


@dataclass
class FaissVectorStore:
    """Índice FAISS junto con sus documentos y manifiesto."""

    index: Any
    documents: list[Document]
    manifest: dict[str, Any]


def _normalize_model_name(model_name: str) -> str:
    """Elimina el prefijo opcional models/."""

    return model_name.strip().removeprefix("models/")


def _prepare_matrix(
    vectors: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convierte los vectores a una matriz float32 normalizada."""

    if not vectors:
        raise FaissStoreError(
            "No se recibieron vectores."
        )

    try:
        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as error:
        raise FaissStoreError(
            "Los embeddings no tienen un formato numérico válido."
        ) from error

    if matrix.ndim != 2:
        raise FaissStoreError(
            "Los embeddings deben formar una matriz bidimensional."
        )

    if matrix.shape[1] == 0:
        raise FaissStoreError(
            "Los embeddings no tienen dimensiones."
        )

    if not np.isfinite(matrix).all():
        raise FaissStoreError(
            "Los embeddings contienen valores NaN o infinitos."
        )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if np.any(norms == 0):
        raise FaissStoreError(
            "No se pueden indexar vectores con norma cero."
        )

    matrix = np.ascontiguousarray(
        matrix,
        dtype=np.float32,
    )

    # Al normalizar los vectores, IndexFlatIP se comporta
    # como una búsqueda por similitud coseno.
    faiss.normalize_L2(matrix)

    return matrix


def build_faiss_index(
    vectors: Sequence[Sequence[float]],
) -> Any:
    """Crea un índice FAISS exacto usando similitud coseno."""

    matrix = _prepare_matrix(vectors)

    embedding_dimension = int(
        matrix.shape[1]
    )

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(matrix)

    return index


def _build_manifest(
    index: Any,
    documents: Sequence[Document],
) -> dict[str, Any]:
    """Construye la información descriptiva del índice."""

    settings = get_settings()

    document_ids = {
        str(document.metadata.get("document_id"))
        for document in documents
        if document.metadata.get("document_id")
    }

    pages = {
        (
            str(document.metadata.get("source")),
            int(document.metadata.get("page_number", 0)),
        )
        for document in documents
    }

    categories = sorted(
        {
            str(document.metadata.get("category"))
            for document in documents
            if document.metadata.get("category")
        }
    )

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model": _normalize_model_name(
            settings.gemini_embedding_model
        ),
        "embedding_dimension": int(index.d),
        "distance_metric": "cosine_similarity",
        "index_type": "IndexFlatIP",
        "vector_count": int(index.ntotal),
        "chunk_count": len(documents),
        "document_count": len(document_ids),
        "page_count": len(pages),
        "categories": categories,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def save_vector_store(
    index: Any,
    documents: Sequence[Document],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Guarda el índice, los chunks y el manifiesto."""

    settings = get_settings()

    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else settings.faiss_index_path
    )

    if not documents:
        raise FaissStoreError(
            "No se recibieron chunks para guardar."
        )

    if int(index.ntotal) != len(documents):
        raise FaissStoreError(
            "La cantidad de vectores no coincide con "
            "la cantidad de chunks."
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_file = destination / INDEX_FILE_NAME
    documents_file = destination / DOCUMENTS_FILE_NAME
    manifest_file = destination / MANIFEST_FILE_NAME

    document_records = [
        {
            "vector_id": vector_id,
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for vector_id, document in enumerate(documents)
    ]

    manifest = _build_manifest(
        index,
        documents,
    )

    try:
        faiss.write_index(
            index,
            str(index_file),
        )

        with documents_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                document_records,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        with manifest_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except OSError as error:
        raise FaissStoreError(
            "No fue posible guardar el índice en "
            f"{destination}."
        ) from error

    return manifest


def create_and_save_vector_store(
    documents: Sequence[Document],
    vectors: Sequence[Sequence[float]],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Crea y guarda el almacén vectorial."""

    if len(documents) != len(vectors):
        raise FaissStoreError(
            "Cada chunk debe tener exactamente un embedding. "
            f"Chunks: {len(documents)}. "
            f"Embeddings: {len(vectors)}."
        )

    index = build_faiss_index(
        vectors
    )

    return save_vector_store(
        index,
        documents,
        output_path,
    )


def load_vector_store(
    index_path: Path | None = None,
) -> FaissVectorStore:
    """Carga el índice, los chunks y el manifiesto."""

    settings = get_settings()

    source = (
        index_path.expanduser().resolve()
        if index_path is not None
        else settings.faiss_index_path
    )

    index_file = source / INDEX_FILE_NAME
    documents_file = source / DOCUMENTS_FILE_NAME
    manifest_file = source / MANIFEST_FILE_NAME

    required_files = [
        index_file,
        documents_file,
        manifest_file,
    ]

    missing_files = [
        file.name
        for file in required_files
        if not file.is_file()
    ]

    if missing_files:
        raise FaissStoreError(
            "El índice está incompleto. Faltan: "
            f"{', '.join(missing_files)}."
        )

    try:
        index = faiss.read_index(
            str(index_file)
        )

        with documents_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        with manifest_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise FaissStoreError(
            f"No fue posible cargar el índice desde {source}."
        ) from error

    if not isinstance(records, list):
        raise FaissStoreError(
            "documents.json debe contener una lista."
        )

    documents: list[Document] = []

    for record in records:
        if not isinstance(record, dict):
            raise FaissStoreError(
                "documents.json contiene un registro inválido."
            )

        page_content = record.get(
            "page_content"
        )

        metadata = record.get(
            "metadata",
            {},
        )

        if not isinstance(page_content, str):
            raise FaissStoreError(
                "Un chunk no contiene texto válido."
            )

        if not isinstance(metadata, dict):
            raise FaissStoreError(
                "Un chunk no contiene metadatos válidos."
            )

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    if int(index.ntotal) != len(documents):
        raise FaissStoreError(
            "El número de vectores no coincide con documents.json."
        )

    if int(index.d) != int(
        manifest.get(
            "embedding_dimension",
            -1,
        )
    ):
        raise FaissStoreError(
            "La dimensión del índice no coincide con el manifiesto."
        )

    indexed_model = str(
        manifest.get(
            "embedding_model",
            "",
        )
    )

    current_model = _normalize_model_name(
        settings.gemini_embedding_model
    )

    if indexed_model != current_model:
        raise FaissStoreError(
            "El índice fue generado con un modelo diferente. "
            f"Índice: {indexed_model}. "
            f"Configuración: {current_model}."
        )

    return FaissVectorStore(
        index=index,
        documents=documents,
        manifest=manifest,
    )


def search_by_vector(
    store: FaissVectorStore,
    query_vector: Sequence[float],
    *,
    k: int = 4,
    score_threshold: float | None = None,
    filters: Mapping[str, object] | None = None,
) -> list[SearchResult]:
    """Busca los chunks más próximos a un vector de consulta."""

    if k <= 0:
        raise FaissStoreError(
            "k debe ser mayor que cero."
        )

    if (
        score_threshold is not None
        and not -1 <= score_threshold <= 1
    ):
        raise FaissStoreError(
            "score_threshold debe estar entre -1 y 1."
        )

    query_matrix = _prepare_matrix(
        [query_vector]
    )

    if int(query_matrix.shape[1]) != int(
        store.index.d
    ):
        raise FaissStoreError(
            "La dimensión de la consulta no coincide con el índice."
        )

    search_size = (
        int(store.index.ntotal)
        if filters
        else min(
            k,
            int(store.index.ntotal),
        )
    )

    scores, vector_ids = store.index.search(
        query_matrix,
        search_size,
    )

    results: list[SearchResult] = []

    for score, vector_id in zip(
        scores[0],
        vector_ids[0],
    ):
        vector_id = int(vector_id)
        score = float(score)

        if vector_id < 0:
            continue

        if (
            score_threshold is not None
            and score < score_threshold
        ):
            continue

        document = store.documents[
            vector_id
        ]

        if filters and not all(
            document.metadata.get(key) == value
            for key, value in filters.items()
        ):
            continue

        results.append(
            SearchResult(
                vector_id=vector_id,
                score=score,
                document=document,
            )
        )

        if len(results) >= k:
            break

    return results