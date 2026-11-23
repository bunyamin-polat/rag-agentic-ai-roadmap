"""Small document loaders: turn text, Markdown, PDF, and HTML files into LangChain Documents.

A loader is just code that reads a file and returns `Document` objects (text + metadata).
Stage 2, Notebook 2 builds these step by step; projects import them from here.

`pypdf` and `beautifulsoup4` are imported inside the functions, so the rest of the
`roadmap` package works without them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

PathLike = str | Path


def load_text(path: PathLike) -> list[Document]:
    """Load a plain text or Markdown file as one Document."""
    path = Path(path)
    return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})]


def load_pdf(path: PathLike) -> list[Document]:
    """Load a PDF as one Document per page. Pages without text are skipped."""
    from pypdf import PdfReader

    path = Path(path)
    documents = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(page_content=text, metadata={"source": path.name, "page": number})
            )
    return documents


def load_html(path: PathLike) -> list[Document]:
    """Load an HTML page as one Document with clean text.

    Menus, footers, scripts, and styles are removed. Headings become Markdown headings
    (`#`, `##`, ...) and list items become `- ` lines, so the text can be split like Markdown.
    """
    from bs4 import BeautifulSoup

    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    blocks = []
    for element in (soup.body or soup).find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ").split())
        if not text:
            continue
        if element.name.startswith("h"):
            text = "#" * int(element.name[1]) + " " + text
        elif element.name == "li":
            text = "- " + text
        blocks.append(text)

    return [
        Document(page_content="\n\n".join(blocks), metadata={"source": path.name, "title": title})
    ]


LOADERS: dict[str, Callable[[PathLike], list[Document]]] = {
    ".txt": load_text,
    ".md": load_text,
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
}


def load_file(path: PathLike) -> list[Document]:
    """Load a file with the loader that matches its extension."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"Cannot load {path.name}: supported file types are {supported}.")
    return loader(path)


def load_directory(folder: PathLike) -> list[Document]:
    """Load every supported file in a folder (not its subfolders), in name order."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in LOADERS)
    return [document for file in files for document in load_file(file)]
