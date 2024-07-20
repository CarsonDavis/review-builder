import pytest

from book_summarizer.llm_core import GPT4O, GPT4oMini, GPT35Turbo
from book_summarizer.text_processing import TextProcessor

# Mock text and token data for testing
mock_text = "This is a test text for tokenization and chunking."
mock_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def processor_35turbo():
    """Fixture for TextProcessor using GPT-3.5-Turbo model."""
    return TextProcessor(model=GPT35Turbo())


@pytest.fixture
def processor_4o():
    """Fixture for TextProcessor using GPT-4O model."""
    return TextProcessor(model=GPT4O())


def test_tokenize_text(processor_35turbo):
    """Validates that text is tokenized into a list of integers using GPT-3.5-Turbo model."""
    tokens = processor_35turbo.tokenize_text(mock_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


def test_chunk_tokens(processor_35turbo):
    """Validates that a list of tokens is correctly chunked with overlap using GPT-3.5-Turbo model."""
    chunk_size = 5
    overlap = 1
    chunks = processor_35turbo.chunk_tokens(mock_tokens, chunk_size, overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, list) for chunk in chunks)
    assert len(chunks[0]) == chunk_size


def test_chunk_text(processor_35turbo):
    """Validates that text is correctly chunked into smaller pieces using GPT-3.5-Turbo model."""
    chunk_size = 10
    overlap = 5
    chunks = processor_35turbo.chunk_text(mock_text, chunk_size, overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_tokenize_text_gpt4o(processor_4o):
    """Validates that text is tokenized into a list of integers using GPT-4O model."""
    tokens = processor_4o.tokenize_text(mock_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


def test_chunk_tokens_gpt4o(processor_4o):
    """Validates that a list of tokens is correctly chunked with overlap using GPT-4O model."""
    chunk_size = 5
    overlap = 1
    chunks = processor_4o.chunk_tokens(mock_tokens, chunk_size, overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, list) for chunk in chunks)
    assert len(chunks[0]) == chunk_size


def test_chunk_text_gpt4o(processor_4o):
    """Validates that text is correctly chunked into smaller pieces using GPT-4O model."""
    chunk_size = 10
    overlap = 5
    chunks = processor_4o.chunk_text(mock_text, chunk_size, overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_tokenize_empty_text(processor_35turbo):
    """Validates that tokenizing empty text returns an empty list."""
    tokens = processor_35turbo.tokenize_text("")
    assert tokens == []


def test_chunk_empty_text(processor_35turbo):
    """Validates that chunking empty text returns an empty list."""
    chunks = processor_35turbo.chunk_text("", chunk_size=10, overlap=5)
    assert chunks == []


def test_chunk_size_larger_than_text(processor_35turbo):
    """Validates that chunking with chunk size larger than text length returns the text as a single chunk."""
    short_text = "Short text"
    chunks = processor_35turbo.chunk_text(short_text, chunk_size=100, overlap=5)
    assert chunks == [short_text]


def test_default_model_initialization():
    """Validates that TextProcessor initializes with GPT4oMini by default."""
    processor = TextProcessor()
    assert isinstance(processor.model, GPT4oMini)


def test_special_characters(processor_35turbo):
    """Validates that tokenizing and chunking text with special characters works correctly."""
    special_text = "Text with special characters: ä, ö, ü, ß, 😊"
    tokens = processor_35turbo.tokenize_text(special_text)
    chunks = processor_35turbo.chunk_text(special_text, chunk_size=10, overlap=5)
    assert isinstance(tokens, list)
    assert isinstance(chunks, list)
    assert all(isinstance(token, int) for token in tokens)
    assert all(isinstance(chunk, str) for chunk in chunks)
