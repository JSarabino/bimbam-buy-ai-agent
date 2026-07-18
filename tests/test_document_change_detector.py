"""Pruebas de detección de cambios en documentos."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bimbam_assistant.infrastructure.document_change_detector import (
    DocumentChangeDetectionError,
    build_corpus_manifest,
    calculate_file_sha256,
    compare_corpus_manifests,
    inspect_corpus_changes,
    load_corpus_manifest,
    save_corpus_manifest,
)


def write_document(
    directory: Path,
    filename: str,
    content: bytes,
) -> Path:
    path = directory / filename

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        content
    )

    return path


def test_calculates_stable_sha256(
    tmp_path: Path,
) -> None:
    document = write_document(
        tmp_path,
        "document.pdf",
        b"contenido-estable",
    )

    first_hash = calculate_file_sha256(
        document
    )

    second_hash = calculate_file_sha256(
        document
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_initial_corpus_marks_all_documents_as_added(
    tmp_path: Path,
) -> None:
    documents_path = (
        tmp_path
        / "documents"
    )

    write_document(
        documents_path,
        "uno.pdf",
        b"uno",
    )

    write_document(
        documents_path,
        "dos.pdf",
        b"dos",
    )

    current_manifest = build_corpus_manifest(
        documents_path
    )

    changes = compare_corpus_manifests(
        None,
        current_manifest,
    )

    assert changes.has_changes is True
    assert changes.added == (
        "dos.pdf",
        "uno.pdf",
    )
    assert changes.modified == ()
    assert changes.deleted == ()
    assert changes.previous_manifest_exists is False


def test_detects_added_modified_deleted_and_unchanged(
    tmp_path: Path,
) -> None:
    documents_path = (
        tmp_path
        / "documents"
    )

    manifest_path = (
        tmp_path
        / "index"
        / "corpus_manifest.json"
    )

    write_document(
        documents_path,
        "unchanged.pdf",
        b"igual",
    )

    write_document(
        documents_path,
        "modified.pdf",
        b"version-1",
    )

    write_document(
        documents_path,
        "deleted.pdf",
        b"se-eliminara",
    )

    previous_manifest = build_corpus_manifest(
        documents_path
    )

    save_corpus_manifest(
        previous_manifest,
        manifest_path,
    )

    (
        documents_path
        / "modified.pdf"
    ).write_bytes(
        b"version-2"
    )

    (
        documents_path
        / "deleted.pdf"
    ).unlink()

    write_document(
        documents_path,
        "added.pdf",
        b"nuevo",
    )

    (
        current_manifest,
        changes,
    ) = inspect_corpus_changes(
        documents_path=documents_path,
        manifest_path=manifest_path,
    )

    assert current_manifest["document_count"] == 3
    assert changes.added == (
        "added.pdf",
    )
    assert changes.modified == (
        "modified.pdf",
    )
    assert changes.deleted == (
        "deleted.pdf",
    )
    assert changes.unchanged == (
        "unchanged.pdf",
    )
    assert changes.changed_count == 3


def test_no_changes_after_saving_current_manifest(
    tmp_path: Path,
) -> None:
    documents_path = (
        tmp_path
        / "documents"
    )

    manifest_path = (
        tmp_path
        / "index"
        / "corpus_manifest.json"
    )

    write_document(
        documents_path,
        "document.pdf",
        b"contenido",
    )

    current_manifest = build_corpus_manifest(
        documents_path
    )

    save_corpus_manifest(
        current_manifest,
        manifest_path,
    )

    (
        next_manifest,
        changes,
    ) = inspect_corpus_changes(
        documents_path=documents_path,
        manifest_path=manifest_path,
    )

    assert next_manifest["document_count"] == 1
    assert changes.has_changes is False
    assert changes.unchanged == (
        "document.pdf",
    )


def test_saves_valid_json_manifest(
    tmp_path: Path,
) -> None:
    documents_path = (
        tmp_path
        / "documents"
    )

    manifest_path = (
        tmp_path
        / "index"
        / "corpus_manifest.json"
    )

    write_document(
        documents_path,
        "document.pdf",
        b"contenido",
    )

    manifest = build_corpus_manifest(
        documents_path
    )

    save_corpus_manifest(
        manifest,
        manifest_path,
    )

    loaded = load_corpus_manifest(
        manifest_path
    )

    assert loaded == manifest

    raw = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )

    assert raw["hash_algorithm"] == "sha256"


def test_rejects_missing_documents_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        DocumentChangeDetectionError,
        match="no existe",
    ):
        build_corpus_manifest(
            tmp_path
            / "missing"
        )
