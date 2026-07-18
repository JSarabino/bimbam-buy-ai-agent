"""Detección de cambios en el corpus documental mediante SHA-256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"
SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf",
}


class DocumentChangeDetectionError(RuntimeError):
    """Error producido al inspeccionar o guardar el manifiesto."""


@dataclass(frozen=True, slots=True)
class DocumentFingerprint:
    """Firma estable de un documento del corpus."""

    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CorpusChangeSet:
    """Diferencias entre el manifiesto anterior y el corpus actual."""

    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]
    previous_manifest_exists: bool

    @property
    def has_changes(self) -> bool:
        """Indica si debe reconstruirse el índice."""

        return bool(
            self.added
            or self.modified
            or self.deleted
        )

    @property
    def changed_count(self) -> int:
        """Cantidad total de archivos agregados, modificados o eliminados."""

        return (
            len(self.added)
            + len(self.modified)
            + len(self.deleted)
        )


def _utc_now_iso() -> str:
    """Devuelve una marca UTC serializable."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def calculate_file_sha256(
    file_path: Path,
    *,
    block_size: int = 1024 * 1024,
) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""

    if block_size <= 0:
        raise ValueError(
            "block_size debe ser mayor que cero."
        )

    if not file_path.is_file():
        raise DocumentChangeDetectionError(
            f"No existe el archivo: {file_path}"
        )

    digest = hashlib.sha256()

    try:
        with file_path.open("rb") as file_handle:
            while block := file_handle.read(
                block_size
            ):
                digest.update(
                    block
                )

    except OSError as error:
        raise DocumentChangeDetectionError(
            f"No fue posible leer el archivo: {file_path}"
        ) from error

    return digest.hexdigest()


def discover_corpus_documents(
    documents_path: Path,
) -> list[Path]:
    """Descubre los documentos admitidos en orden estable."""

    if not documents_path.is_dir():
        raise DocumentChangeDetectionError(
            "El directorio del corpus no existe: "
            f"{documents_path}"
        )

    return sorted(
        (
            path
            for path in documents_path.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_DOCUMENT_SUFFIXES
            )
        ),
        key=lambda path: path.relative_to(
            documents_path
        ).as_posix().lower(),
    )


def fingerprint_document(
    file_path: Path,
    *,
    documents_path: Path,
) -> DocumentFingerprint:
    """Construye la firma de un documento."""

    try:
        relative_path = file_path.relative_to(
            documents_path
        ).as_posix()

        size_bytes = file_path.stat().st_size

    except OSError as error:
        raise DocumentChangeDetectionError(
            f"No fue posible inspeccionar: {file_path}"
        ) from error

    return DocumentFingerprint(
        relative_path=relative_path,
        sha256=calculate_file_sha256(
            file_path
        ),
        size_bytes=size_bytes,
    )


def build_corpus_manifest(
    documents_path: Path,
) -> dict[str, Any]:
    """Genera el manifiesto actual del corpus."""

    documents = discover_corpus_documents(
        documents_path
    )

    fingerprints = [
        fingerprint_document(
            document,
            documents_path=documents_path,
        )
        for document in documents
    ]

    return {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "hash_algorithm": "sha256",
        "document_count": len(
            fingerprints
        ),
        "documents": [
            asdict(
                fingerprint
            )
            for fingerprint in fingerprints
        ],
    }


def load_corpus_manifest(
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Carga el manifiesto anterior o devuelve None si no existe."""

    if not manifest_path.exists():
        return None

    if not manifest_path.is_file():
        raise DocumentChangeDetectionError(
            "La ruta del manifiesto no es un archivo: "
            f"{manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            manifest = json.load(
                file_handle
            )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise DocumentChangeDetectionError(
            "No fue posible leer el manifiesto del corpus."
        ) from error

    if not isinstance(
        manifest,
        dict,
    ):
        raise DocumentChangeDetectionError(
            "El manifiesto del corpus tiene un formato inválido."
        )

    return manifest


def save_corpus_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Guarda el manifiesto de forma atómica."""

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                manifest,
                file_handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        temporary_path.replace(
            manifest_path
        )

    except OSError as error:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise DocumentChangeDetectionError(
            "No fue posible guardar el manifiesto del corpus."
        ) from error


def _manifest_documents_by_path(
    manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Indexa los documentos del manifiesto por ruta relativa."""

    if manifest is None:
        return {}

    documents = manifest.get(
        "documents",
        [],
    )

    if not isinstance(
        documents,
        list,
    ):
        raise DocumentChangeDetectionError(
            "La lista de documentos del manifiesto es inválida."
        )

    indexed_documents: dict[str, dict[str, Any]] = {}

    for document in documents:
        if not isinstance(
            document,
            dict,
        ):
            raise DocumentChangeDetectionError(
                "Existe una entrada inválida en el manifiesto."
            )

        relative_path = document.get(
            "relative_path"
        )

        sha256 = document.get(
            "sha256"
        )

        if not isinstance(
            relative_path,
            str,
        ) or not isinstance(
            sha256,
            str,
        ):
            raise DocumentChangeDetectionError(
                "Una entrada del manifiesto no contiene ruta y hash válidos."
            )

        indexed_documents[
            relative_path
        ] = document

    return indexed_documents


def compare_corpus_manifests(
    previous_manifest: dict[str, Any] | None,
    current_manifest: dict[str, Any],
) -> CorpusChangeSet:
    """Compara el estado anterior con el estado actual."""

    previous_documents = (
        _manifest_documents_by_path(
            previous_manifest
        )
    )

    current_documents = (
        _manifest_documents_by_path(
            current_manifest
        )
    )

    previous_paths = set(
        previous_documents
    )

    current_paths = set(
        current_documents
    )

    added = sorted(
        current_paths - previous_paths
    )

    deleted = sorted(
        previous_paths - current_paths
    )

    shared_paths = (
        previous_paths
        & current_paths
    )

    modified = sorted(
        relative_path
        for relative_path in shared_paths
        if (
            previous_documents[
                relative_path
            ].get("sha256")
            != current_documents[
                relative_path
            ].get("sha256")
        )
    )

    unchanged = sorted(
        relative_path
        for relative_path in shared_paths
        if (
            previous_documents[
                relative_path
            ].get("sha256")
            == current_documents[
                relative_path
            ].get("sha256")
        )
    )

    return CorpusChangeSet(
        added=tuple(
            added
        ),
        modified=tuple(
            modified
        ),
        deleted=tuple(
            deleted
        ),
        unchanged=tuple(
            unchanged
        ),
        previous_manifest_exists=(
            previous_manifest is not None
        ),
    )


def inspect_corpus_changes(
    *,
    documents_path: Path,
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    CorpusChangeSet,
]:
    """Genera el manifiesto actual y calcula los cambios."""

    previous_manifest = load_corpus_manifest(
        manifest_path
    )

    current_manifest = build_corpus_manifest(
        documents_path
    )

    changes = compare_corpus_manifests(
        previous_manifest,
        current_manifest,
    )

    return (
        current_manifest,
        changes,
    )


def default_corpus_manifest_path(
    faiss_index_path: Path,
) -> Path:
    """Construye la ruta estándar del manifiesto."""

    return (
        faiss_index_path
        / CORPUS_MANIFEST_FILENAME
    )
