"""Tests for roadmap.loaders, using small temporary files and the stage 2 sample data."""

from pathlib import Path

import pytest

from roadmap import loaders

DATA = Path(__file__).resolve().parents[1] / "02-rag-fundamentals" / "data"


def test_load_text_keeps_content_and_file_name(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("# Title\n\nSome text.", encoding="utf-8")

    [document] = loaders.load_text(file)

    assert document.page_content == "# Title\n\nSome text."
    assert document.metadata == {"source": "notes.md"}


def test_load_pdf_returns_one_document_per_page():
    documents = loaders.load_pdf(DATA / "handbook.pdf")

    assert len(documents) == 3
    assert [d.metadata["page"] for d in documents] == [1, 2, 3]
    assert all(d.metadata["source"] == "handbook.pdf" for d in documents)
    assert "28 vacation days" in " ".join(d.page_content for d in documents)


def test_load_html_removes_page_chrome_and_marks_headings(tmp_path):
    file = tmp_path / "page.html"
    file.write_text(
        "<html><head><title>FAQ</title><script>track()</script></head><body>"
        "<nav>Home | Log out</nav><h1>Help</h1><p>Line one\n   continues here.</p>"
        "<ul><li>First tip</li></ul><footer>Cookies</footer></body></html>",
        encoding="utf-8",
    )

    [document] = loaders.load_html(file)

    assert document.page_content == "# Help\n\nLine one continues here.\n\n- First tip"
    assert document.metadata == {"source": "page.html", "title": "FAQ"}


def test_load_file_picks_loader_by_extension_and_rejects_unknown(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.docx").write_text("?", encoding="utf-8")

    assert loaders.load_file(tmp_path / "a.txt")[0].page_content == "hello"
    with pytest.raises(ValueError, match="supported file types"):
        loaders.load_file(tmp_path / "b.docx")


def test_load_directory_loads_supported_files_in_name_order():
    documents = loaders.load_directory(DATA)

    sources = [d.metadata["source"] for d in documents]
    assert sources == ["handbook.md", "handbook.pdf", "handbook.pdf", "handbook.pdf", "it_faq.html"]
