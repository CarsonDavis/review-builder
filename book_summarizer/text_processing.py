import re

import tiktoken

from book_summarizer.llm_core import GPT4O, GPT4oMini, LLMClient


def find_boolean_in_string(text: str) -> bool:
    """
    Searches for the words 'true' or 'false' in any capitalization within a string.
    Returns the associated boolean value. If no match is found, returns True.

    Args:
        text (str): The input string to search within.

    Returns:
        bool: The boolean value associated with the found word, or True if no match is found.
    """
    # Compile regex patterns to match 'true' or 'false' in any capitalization
    true_pattern = re.compile(r"\btrue\b", re.IGNORECASE)
    false_pattern = re.compile(r"\bfalse\b", re.IGNORECASE)

    if true_pattern.search(text):
        return True
    elif false_pattern.search(text):
        return False

    return True


class TextProcessor:
    def __init__(self, model: LLMClient | None = None):
        self.model = model or GPT4oMini()

    def tokenize_text(self, text: str) -> list[int]:
        encoding = tiktoken.encoding_for_model(self.model.model_name)
        return encoding.encode(text)

    def chunk_tokens(self, tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunks.append(tokens[i : i + chunk_size])
            if i + chunk_size >= len(tokens):
                break
        return chunks

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        tokens = self.tokenize_text(text)
        tokenized_chunks = self.chunk_tokens(tokens, chunk_size, overlap)
        encoding = tiktoken.encoding_for_model(self.model.model_name)
        return [encoding.decode(chunk) for chunk in tokenized_chunks]


# Example usage
if __name__ == "__main__":
    # Create a TextProcessor with the default GPT-3.5-Turbo model
    processor = TextProcessor()

    text = "Your long text goes here..."

    # Chunk the text
    chunks = processor.chunk_text(text, chunk_size=processor.model.max_tokens - 50, overlap=50)
    for chunk in chunks:
        print(chunk)

    # Create a TextProcessor with the GPT-4O model
    processor_gpt4o = TextProcessor(GPT4O())

    # Chunk the text with the GPT-4O model
    chunks_gpt4o = processor_gpt4o.chunk_text(text, chunk_size=processor_gpt4o.model.max_tokens - 50, overlap=50)
    for chunk in chunks_gpt4o:
        print(chunk)
