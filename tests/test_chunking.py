"""Pruebas offline del procesamiento y chunking documental."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from bimbam_assistant.application import indexing_service
from bimbam_assistant.application.indexing_service import (
    ChunkingError,
    IndexingError,
    build_text_splitter,
    create_chunks,
    create_vector_index,
    validate_chunks_for_indexing,
)


def build_page(
    *,
    content: str,
    document_id: str = "manual",
    page_number: int = 1,
    category: str = "garantias",
    is_empty: bool = False,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "document_name": "Manual",
            "file_name": "manual.pdf",
            "source": "data/documents/manual.pdf",
            "page_number": page_number,
            "category": category,
            "is_empty": is_empty,
        },
    )


def build_valid_chunk(
    *,
    chunk_id: str = "manual-page-1-chunk-0",
    content: str = "Contenido válido",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": "data/documents/manual.pdf",
            "page_number": 1,
            "category": "garantias",
        },
    )


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap, expected_message",
    [
        (0, 0, "chunk_size debe ser mayor que cero"),
        (100, -1, "chunk_overlap no puede ser negativo"),
        (100, 100, "chunk_overlap debe ser menor que chunk_size"),
    ],
)
def test_build_text_splitter_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ChunkingError,
        match=expected_message,
    ):
        build_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_create_chunks_returns_empty_list_without_documents() -> None:
    assert create_chunks([]) == []


def test_create_chunks_preserves_page_boundaries_and_metadata() -> None:
    pages = [
        build_page(
            content="ALFA " * 45,
            document_id="doc-a",
            page_number=1,
        ),
        build_page(
            content="BETA " * 45,
            document_id="doc-a",
            page_number=2,
        ),
    ]

    chunks = create_chunks(
        pages,
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) > 2

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))

    for chunk in chunks:
        assert 0 < len(chunk.page_content) <= 80
        assert chunk.metadata["configured_chunk_size"] == 80
        assert chunk.metadata["configured_chunk_overlap"] == 10
        assert (
            chunk.metadata["chunk_character_count"]
            == len(chunk.page_content)
        )

        page_number = chunk.metadata["page_number"]

        if page_number == 1:
            assert "BETA" not in chunk.page_content
        elif page_number == 2:
            assert "ALFA" not in chunk.page_content
        else:
            pytest.fail(
                "El chunk mezcló o perdió el número de página."
            )


def test_create_chunks_skips_empty_pages() -> None:
    pages = [
        build_page(
            content="",
            page_number=1,
            is_empty=True,
        ),
        build_page(
            content="Contenido válido para indexar.",
            page_number=2,
        ),
    ]

    chunks = create_chunks(
        pages,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] == 2


def test_validate_chunks_accepts_valid_unique_chunks() -> None:
    validate_chunks_for_indexing(
        [
            build_valid_chunk(chunk_id="chunk-1"),
            build_valid_chunk(chunk_id="chunk-2"),
        ]
    )


@pytest.mark.parametrize(
    "chunk",
    [
        Document(
            page_content=" ",
            metadata={
                "chunk_id": "chunk-1",
                "source": "manual.pdf",
                "page_number": 1,
                "category": "garantias",
            },
        ),
        Document(
            page_content="Contenido",
            metadata={
                "source": "manual.pdf",
                "page_number": 1,
                "category": "garantias",
            },
        ),
        Document(
            page_content="Contenido",
            metadata={
                "chunk_id": "chunk-1",
                "page_number": 1,
                "category": "garantias",
            },
        ),
        Document(
            page_content="Contenido",
            metadata={
                "chunk_id": "chunk-1",
                "source": "manual.pdf",
                "category": "garantias",
            },
        ),
        Document(
            page_content="Contenido",
            metadata={
                "chunk_id": "chunk-1",
                "source": "manual.pdf",
                "page_number": 1,
                "category": "sin_clasificar",
            },
        ),
    ],
)
def test_validate_chunks_rejects_invalid_chunk(
    chunk: Document,
) -> None:
    with pytest.raises(IndexingError):
        validate_chunks_for_indexing([chunk])


def test_validate_chunks_rejects_duplicate_identifiers() -> None:
    chunks = [
        build_valid_chunk(chunk_id="duplicate"),
        build_valid_chunk(chunk_id="duplicate"),
    ]

    with pytest.raises(
        IndexingError,
        match="duplicados",
    ):
        validate_chunks_for_indexing(chunks)


def test_create_vector_index_uses_mocked_embeddings_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        build_valid_chunk(
            chunk_id="chunk-1",
            content="Primer contenido",
        ),
        build_valid_chunk(
            chunk_id="chunk-2",
            content="Segundo contenido",
        ),
    ]

    captured: dict[str, object] = {}

    def fake_embed_documents(
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        captured["texts"] = texts
        captured["batch_size"] = batch_size

        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    def fake_create_and_save_vector_store(
        received_chunks: list[Document],
        vectors: list[list[float]],
        destination: Path,
    ) -> dict[str, object]:
        captured["chunks"] = received_chunks
        captured["vectors"] = vectors
        captured["destination"] = destination

        return {
            "vector_count": 2,
            "embedding_dimension": 2,
        }

    monkeypatch.setattr(
        indexing_service,
        "get_settings",
        lambda: SimpleNamespace(
            faiss_index_path=tmp_path / "default-index"
        ),
    )

    monkeypatch.setattr(
        indexing_service,
        "embed_documents",
        fake_embed_documents,
    )

    monkeypatch.setattr(
        indexing_service,
        "create_and_save_vector_store",
        fake_create_and_save_vector_store,
    )

    output_path = tmp_path / "index"

    result = create_vector_index(
        chunks,
        output_path=output_path,
        batch_size=2,
    )

    assert captured["texts"] == [
        "Primer contenido",
        "Segundo contenido",
    ]
    assert captured["batch_size"] == 2
    assert captured["destination"] == output_path.resolve()
    assert result.chunk_count == 2
    assert result.embedding_dimension == 2
    assert result.output_path == output_path.resolve()


def test_create_vector_index_rejects_embedding_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        build_valid_chunk(chunk_id="chunk-1"),
        build_valid_chunk(chunk_id="chunk-2"),
    ]

    monkeypatch.setattr(
        indexing_service,
        "get_settings",
        lambda: SimpleNamespace(
            faiss_index_path=tmp_path
        ),
    )

    monkeypatch.setattr(
        indexing_service,
        "embed_documents",
        lambda texts, batch_size: [[1.0, 0.0]],
    )

    with pytest.raises(
        IndexingError,
        match="cantidad de embeddings",
    ):
        create_vector_index(
            chunks,
            output_path=tmp_path,
            batch_size=2,
        )
