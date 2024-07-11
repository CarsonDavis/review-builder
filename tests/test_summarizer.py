from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from book_summarizer import BookSummarizer

# Load the API key which OpenAI will read from the environment
load_dotenv()


@pytest.fixture
def mock_extractor(mocker: Any) -> MagicMock:
    mock_extractor = mocker.patch("book_summarizer.EpubExtractor")
    instance = mock_extractor.return_value
    instance.chapters = ["Chapter 1 This is the first chapter.", "Chapter 2 This is the second chapter."]
    return instance


@pytest.fixture
def summarizer(sample_epub_path: Path, mock_extractor: MagicMock) -> BookSummarizer:
    with patch("book_summarizer.EpubExtractor._validate_file_path"):
        return BookSummarizer(sample_epub_path)


def test_summarize_text(summarizer: BookSummarizer) -> None:
    summary = summarizer.summarize_text(text="This is a test.")
    assert len(summary) > 0  # Check that the summary is not empty


def test_summarize_text_with_chunking(summarizer: BookSummarizer) -> None:
    text = "This is a test." * 100
    summary = summarizer.summarize_text_with_chunking(text)
    assert len(summary) > 0  # Check that the summary is not empty


def test_summarize_book(summarizer: BookSummarizer, tmp_path: Path) -> None:
    output_path = tmp_path / "book_summary.md"
    summarizer.summarize_book(output_filename=str(output_path))
    assert output_path.exists()
    with open(output_path) as f:
        content = f.read()
    assert "## Chapter 1" in content
    assert len(content) > 0  # Check that the content is not empty
    assert "error" not in content.lower(), "Error string found in summary"


def test_log_recent_experiment(summarizer: BookSummarizer, tmp_path: Path) -> None:
    summarizer.recent_experiment = {
        "model": "gpt-3.5-turbo",
        "system_prompt": "Test system prompt",
        "instruction": "Test instruction",
        "summary": "Test summary",
        "text": "Test text",
    }
    log_path = tmp_path / "prompting_log.md"
    summarizer.log_recent_experiment(filename=str(log_path))
    assert log_path.exists()
    with open(log_path) as f:
        content = f.read()
    assert "Test system prompt" in content
    assert "Test instruction" in content
    assert "Test summary" in content


if __name__ == "__main__":
    pytest.main()
