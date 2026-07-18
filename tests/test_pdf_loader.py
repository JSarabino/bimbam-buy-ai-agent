"""Pruebas offline de carga y extracción de documentos PDF."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from bimbam_assistant.infrastructure import pdf_loader
from bimbam_assistant.infrastructure.pdf_loader import (
    PdfLoadingError,
    clean_text,
    find_pdf_files,
    load_pdf,
    load_pdf_documents,
)


def create_pdf(path: Path, page_texts: list[str]) -> None:
    """Crea un PDF pequeño para las pruebas."""

    document = pymupdf.open()

    for text in page_texts:
        page = document.new_page()

        if text:
            page.insert_text(
                (72, 72),
                text,
            )

    document.save(path)
    document.close()


def test_clean_text_normalizes_invisible_characters_and_spaces() -> None:
    raw_text = (
        "  Primera\x00   línea  \r\n"
        "\r\n"
        "\r\n"
        "Segunda\u200b\tlínea\u00ad  "
    )

    assert clean_text(raw_text) == (
        "Primera línea\n\n"
        "Segunda línea"
    )


def test_find_pdf_files_filters_and_orders_case_insensitively(
    tmp_path: Path,
) -> None:
    (tmp_path / "zeta.PDF").write_bytes(b"pdf")
    (tmp_path / "Alpha.pdf").write_bytes(b"pdf")
    (tmp_path / "notes.txt").write_text(
        "not a pdf",
        encoding="utf-8",
    )

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "ignored.pdf").write_bytes(b"pdf")

    files = find_pdf_files(tmp_path)

    assert [path.name for path in files] == [
        "Alpha.pdf",
        "zeta.PDF",
    ]


@pytest.mark.parametrize(
    "setup_mode, expected_message",
    [
        ("missing", "No se encontró la carpeta"),
        ("file", "no corresponde a una carpeta"),
        ("empty", "No se encontraron archivos PDF"),
    ],
)
def test_find_pdf_files_rejects_invalid_locations(
    tmp_path: Path,
    setup_mode: str,
    expected_message: str,
) -> None:
    target = tmp_path / "documents"

    if setup_mode == "file":
        target.write_text(
            "content",
            encoding="utf-8",
        )
    elif setup_mode == "empty":
        target.mkdir()

    with pytest.raises(
        PdfLoadingError,
        match=expected_message,
    ):
        find_pdf_files(target)


def test_load_pdf_extracts_pages_and_traceable_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = (
        tmp_path
        / "Manual de Garantía de Productos de BimBam Buy.pdf"
    )

    create_pdf(
        pdf_path,
        [
            (
                "Esta pagina contiene suficiente texto nativo para "
                "comprobar la extraccion del documento."
            ),
            "",
        ],
    )

    monkeypatch.setattr(
        pdf_loader,
        "get_settings",
        lambda: SimpleNamespace(
            project_root=tmp_path
        ),
    )

    pages = load_pdf(
        pdf_path,
        project_root=tmp_path,
    )

    assert len(pages) == 2

    first_page = pages[0]
    second_page = pages[1]

    assert (
        first_page.metadata["document_name"]
        == "Manual de Garantía de Productos de BimBam Buy"
    )
    assert first_page.metadata["category"] == "garantias"
    assert first_page.metadata["source"] == pdf_path.name
    assert first_page.metadata["page_number"] == 1
    assert first_page.metadata["page_index"] == 0
    assert first_page.metadata["total_pages"] == 2
    assert first_page.metadata["is_empty"] is False
    assert (
        first_page.metadata["character_count"]
        == len(first_page.page_content)
    )

    assert second_page.metadata["page_number"] == 2
    assert second_page.metadata["is_empty"] is True
    assert second_page.page_content == ""


def test_load_pdf_rejects_non_pdf_file(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text(
        "not a pdf",
        encoding="utf-8",
    )

    with pytest.raises(
        PdfLoadingError,
        match="no tiene extensión PDF",
    ):
        load_pdf(
            text_file,
            project_root=tmp_path,
        )


def test_load_pdf_wraps_invalid_pdf_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text(
        "this is not a real PDF",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pdf_loader,
        "get_settings",
        lambda: SimpleNamespace(
            project_root=tmp_path
        ),
    )

    with pytest.raises(
        PdfLoadingError,
        match="No fue posible procesar",
    ):
        load_pdf(
            invalid_pdf,
            project_root=tmp_path,
        )


def test_load_pdf_documents_combines_all_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"

    create_pdf(
        first_pdf,
        ["Contenido del primer documento con texto suficiente."],
    )

    create_pdf(
        second_pdf,
        [
            "Primera pagina del segundo documento con suficiente texto.",
            "Segunda pagina del segundo documento con suficiente texto.",
        ],
    )

    monkeypatch.setattr(
        pdf_loader,
        "get_settings",
        lambda: SimpleNamespace(
            project_root=tmp_path
        ),
    )

    pages = load_pdf_documents(tmp_path)

    assert len(pages) == 3
    assert {
        page.metadata["file_name"]
        for page in pages
    } == {
        "first.pdf",
        "second.pdf",
    }
