import pytest
import os
from ebooklib import epub
from book_summarizer.epub_extractor import EpubExtractor
from pathlib import Path


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
    chapter1.content = "<html><body><h1>Chapter 1</h1>" "<p>This is the first chapter.</p></body></html>"
    book.add_item(chapter1)

    # Create chapter 2
    chapter2 = epub.EpubHtml(title="Chapter 2", file_name="chap_02.xhtml", lang="en")
    chapter2.content = "<html><body><h1>Chapter 2</h1>" "<p>This is the second chapter.</p></body></html>"
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


@pytest.fixture
def invalid_epub_path() -> str:
    """
    Fixture to provide an invalid EPUB path.

    Returns:
        str: An invalid EPUB file path.
    """
    return "invalid/path/to/epub.epub"


class TestEpubExtractor:

    def test_validate_file_path_valid(self, sample_epub_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        assert extractor.epub_file_path == sample_epub_path

    def test_validate_file_path_invalid(self, invalid_epub_path: str):
        with pytest.raises(FileNotFoundError):
            EpubExtractor(invalid_epub_path)

    def test_clean_text(self):
        # Create a dummy instance without validating file path
        extractor = EpubExtractor.__new__(EpubExtractor)
        dirty_text = "   This is a test.   \n\n\n"
        clean_text = extractor._clean_text(dirty_text)
        assert clean_text == "This is a test."

        dirty_text = "\n\n\n\nMultiple\n\nnewlines\n\n"
        clean_text = extractor._clean_text(dirty_text)
        assert clean_text == "Multiple\nnewlines"

    def test_get_chapters(self, sample_epub_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        chapters = extractor._get_chapters()
        assert len(chapters) == 2
        assert "This is the first chapter." in chapters[0]
        assert "This is the second chapter." in chapters[1]

    def test_write_to_txt(self, sample_epub_path: Path, tmp_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        output_path = tmp_path / "output.txt"
        chapters = extractor._get_chapters()
        extractor._write_to_txt(chapters, str(output_path))
        assert output_path.exists()
        with output_path.open("r") as f:
            content = f.read()
        assert "This is the first chapter." in content
        assert "This is the second chapter." in content

    def test_generate_output_path(self, sample_epub_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        expected_output_path = sample_epub_path.with_suffix(".txt")
        assert extractor._generate_output_path() == str(expected_output_path)

    def test_save(self, sample_epub_path: Path, tmp_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        output_path = tmp_path / "output.txt"
        extractor.save(output_path=str(output_path))
        assert output_path.exists()
        with output_path.open("r") as f:
            content = f.read()
        assert "This is the first chapter." in content
        assert "This is the second chapter." in content

    def test_save_default_path(self, sample_epub_path: Path):
        extractor = EpubExtractor(sample_epub_path)
        output_path = Path(extractor.save())
        assert output_path.exists()
        with output_path.open("r") as f:
            content = f.read()
        assert "This is the first chapter." in content
        assert "This is the second chapter." in content


# Suppress specific warnings for clean test output
@pytest.fixture(autouse=True)
def suppress_warnings():
    import warnings

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
