import pytest
from pathlib import Path
from ebooklib import epub


@pytest.fixture
def sample_epub_path(tmp_path: Path) -> Path:
    """
    Create a sample EPUB book and save it to the specified path.

    Args:
        tmp_path (Path): The temporary path for creating the EPUB file.

    Returns:
        Path: The path to the created EPUB file.
    """
    epub_path = tmp_path / "test.epub"
    book = epub.EpubBook()
    book.set_title("Sample Book")
    book.add_author("Author")

    # Create chapter 1
    chapter1 = epub.EpubHtml(title="Chapter 1", file_name="chap_01.xhtml", lang="en")
    chapter1.content = "<html><body><h1>Chapter 1</h1><p>This is the first chapter.</p></body></html>"
    book.add_item(chapter1)

    # Create chapter 2
    chapter2 = epub.EpubHtml(title="Chapter 2", file_name="chap_02.xhtml", lang="en")
    chapter2.content = "<html><body><h1>Chapter 2</h1><p>This is the second chapter.</p></body></html>"
    book.add_item(chapter2)

    # Create a TOC and spine
    book.toc = (
        epub.Link(chapter1.file_name, "Chapter 1", "chap_01"),
        epub.Link(chapter2.file_name, "Chapter 2", "chap_02"),
    )
    book.spine = ["nav", chapter1, chapter2]

    # Add default NCX
    book.add_item(epub.EpubNcx())

    # Write the EPUB file
    epub.write_epub(epub_path, book, {})

    return epub_path
