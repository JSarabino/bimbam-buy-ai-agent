"""Pruebas de la decisión de indexación condicional."""

from __future__ import annotations

from bimbam_assistant.infrastructure.document_change_detector import (
    CorpusChangeSet,
)
from scripts.index_documents import (
    get_rebuild_reasons,
    parse_arguments,
    should_rebuild_index,
)


def build_changes(
    *,
    added: tuple[str, ...] = (),
    modified: tuple[str, ...] = (),
    deleted: tuple[str, ...] = (),
    unchanged: tuple[str, ...] = (),
    previous_manifest_exists: bool = True,
) -> CorpusChangeSet:
    return CorpusChangeSet(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
        previous_manifest_exists=previous_manifest_exists,
    )


def test_skips_when_index_and_corpus_are_current() -> None:
    changes = build_changes(
        unchanged=(
            "document.pdf",
        ),
    )

    assert should_rebuild_index(
        force=False,
        index_exists=True,
        changes=changes,
    ) is False


def test_rebuilds_when_force_is_enabled() -> None:
    changes = build_changes(
        unchanged=(
            "document.pdf",
        ),
    )

    assert should_rebuild_index(
        force=True,
        index_exists=True,
        changes=changes,
    ) is True


def test_rebuilds_when_index_is_missing() -> None:
    changes = build_changes(
        unchanged=(
            "document.pdf",
        ),
    )

    assert should_rebuild_index(
        force=False,
        index_exists=False,
        changes=changes,
    ) is True


def test_rebuilds_when_document_is_added() -> None:
    changes = build_changes(
        added=(
            "new.pdf",
        ),
    )

    assert should_rebuild_index(
        force=False,
        index_exists=True,
        changes=changes,
    ) is True


def test_rebuilds_when_document_is_modified() -> None:
    changes = build_changes(
        modified=(
            "changed.pdf",
        ),
    )

    assert should_rebuild_index(
        force=False,
        index_exists=True,
        changes=changes,
    ) is True


def test_rebuilds_when_document_is_deleted() -> None:
    changes = build_changes(
        deleted=(
            "removed.pdf",
        ),
    )

    assert should_rebuild_index(
        force=False,
        index_exists=True,
        changes=changes,
    ) is True


def test_parse_arguments_supports_force() -> None:
    arguments = parse_arguments(
        [
            "--force",
        ]
    )

    assert arguments.force is True


def test_rebuild_reasons_include_all_active_causes() -> None:
    changes = build_changes(
        added=(
            "new.pdf",
        ),
        modified=(
            "changed.pdf",
        ),
        deleted=(
            "removed.pdf",
        ),
    )

    reasons = get_rebuild_reasons(
        force=True,
        index_exists=False,
        changes=changes,
    )

    assert reasons == [
        "se solicitó reconstrucción forzada",
        "el índice FAISS no existe o está incompleto",
        "1 documento(s) agregado(s)",
        "1 documento(s) modificado(s)",
        "1 documento(s) eliminado(s)",
    ]
