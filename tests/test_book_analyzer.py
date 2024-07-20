from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from book_summarizer.book_analyzer import BookAnalyzer
from book_summarizer.llm_core import GPT35Turbo


@pytest.fixture
def mock_extractor(mocker: Any) -> MagicMock:
    mock_extractor = mocker.patch("book_summarizer.epub_extractor.EpubExtractor")
    instance = mock_extractor.return_value
    instance._get_chapters.return_value = [
        "Chapter 1 This is the first chapter.",
        "Chapter 2 This is the second chapter.",
    ]
    return instance


@pytest.fixture
def mock_cost_calculator(mocker: Any) -> MagicMock:
    mock_cost_calculator = mocker.patch("book_summarizer.cost_calculator.CostCalculator")
    instance = mock_cost_calculator.return_value
    instance.calculate_cost.return_value = 1.23
    instance.encoding.encode.side_effect = lambda text: text.split()  # Simple tokenizer mock
    return instance


@pytest.fixture
def analyzer(sample_epub_path: Path, mock_extractor: MagicMock) -> BookAnalyzer:
    with patch("book_summarizer.epub_extractor.EpubExtractor._validate_file_path"):
        return BookAnalyzer(sample_epub_path)


def test_word_counts(analyzer: BookAnalyzer) -> None:
    """
    Test that the total word count and chapter word counts are correct.
    Note: Chapters contain the entire content, including the title of the chapter.
    """
    total_word_count, chapter_word_counts = analyzer.word_counts()
    assert total_word_count == 16
    assert chapter_word_counts == [8, 8]


def test_token_counts(analyzer: BookAnalyzer, mock_cost_calculator: MagicMock) -> None:
    """
    Test that the token counts are correctly calculated for different models.
    """
    token_counts = analyzer.token_counts()
    assert "gpt-4o" in token_counts
    assert "gpt-4o-mini" in token_counts
    assert token_counts["gpt-4o"][0] == 20
    assert token_counts["gpt-4o-mini"][0] == 20


def test_word_frequencies(analyzer: BookAnalyzer) -> None:
    """
    Test that the word frequencies are correctly calculated.
    """
    word_frequencies = analyzer.word_frequencies()
    expected_frequencies = {
        "Chapter": 2,
        "This": 2,
        "is": 2,
        "the": 2,
        "chapter": 2,
        ".": 2,
        "1": 1,
        "first": 1,
        "2": 1,
        "second": 1,
    }
    assert word_frequencies == expected_frequencies


def test_calculate_cost(analyzer: BookAnalyzer) -> None:
    """
    Test that the cost calculation is correct for GPT-3.5 Turbo.
    """
    model_client = GPT35Turbo()
    cost = analyzer.calculate_cost(model_client)
    assert cost == 9e-06


def test_write_statistics(analyzer: BookAnalyzer, tmp_path: Path) -> None:
    """
    Test that headers are correctly written to a file.
    """
    output_path = tmp_path / "output_stats.md"
    analyzer.write_statistics(output_path)
    assert output_path.exists()
    with open(output_path) as f:
        content = f.read()
    assert "# Book Statistics" in content
    assert "## Overview" in content
    assert "Total Word Count" in content
    assert "## Word and Token Counts per Chapter" in content
    assert "## Word Frequencies (First 200 Words)" in content


if __name__ == "__main__":
    pytest.main()
