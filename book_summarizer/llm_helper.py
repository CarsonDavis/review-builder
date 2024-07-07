import datetime
import os
from typing import Callable, Optional

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .epub_extractor import EpubExtractor

# Load the API key which OpenAI will read from the environment
load_dotenv()
client = OpenAI()


def validate_model_name(func: Callable) -> Callable:
    """Decorator to validate the model argument."""

    def wrapper(self, *args, **kwargs):
        model = kwargs.get("model") or (args[1] if len(args) > 1 else None)
        if model not in self.ACCEPTABLE_MODELS:
            raise ValueError(f"Model {model} is not known. Choose from {list(self.ACCEPTABLE_MODELS.keys())}.")
        return func(self, *args, **kwargs)

    return wrapper


class BookSummarizer:
    DEFAULT_SYSTEM_PROMPT = "You are a skilled textual analyst that can synthesize the key concepts in long text and identify crucial details to retain."
    DEFAULT_INSTRUCTION_PROMPT = "Make a list of the key points made by the author in the following chapter. Under each point, list out the reasons or evidence given."
    DEFAULT_COMBINER_PROMPT = "I will provide you with several summaries of different parts of a chapter. Combine these summaries into one single summary."

    SUMMARY_SIZE = 1500  # gpt-3.5-turbo summaries for 12k chapters were 500 tokens. 1500 should be safe.
    ACCEPTABLE_MODELS = {
        "gpt-3.5-turbo": {"max_tokens": 16385 - SUMMARY_SIZE},
        "gpt-4o": {"max_tokens": 128000 - SUMMARY_SIZE},
    }
    CHUNK_OVERLAP = 50

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.extractor = EpubExtractor(epub_path)
        self.chapters = self.extractor.chapters
        self.recent_experiment = None

    def _tokenize_text(self, text: str, model: str) -> list[int]:
        """
        Tokenizes the input text using the specified model's encoding.

        Args:
            text (str): The text to be tokenized.
            model (str): The model name to determine the encoding.

        Returns:
            list[int]: A list of token ids.
        """
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode(text)

    def _chunk_tokens(self, tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
        """
        Splits a list of tokens into smaller chunks with a specified overlap.

        Args:
            tokens (list[int]): The list of token ids to be chunked.
            chunk_size (int): The maximum number of tokens per chunk.
            overlap (int): The number of overlapping tokens between consecutive chunks.

        Returns:
            list[list[int]]: A list of token chunks.
        """
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunks.append(tokens[i : i + chunk_size])
            if i + chunk_size >= len(tokens):
                break
        return chunks

    @validate_model_name
    def chunk_text(self, text: str, model: str) -> list[str]:
        """
        Chunks the input text into smaller pieces based on token limits, including overlap.

        Args:
            text (str): The text to be chunked.
            model (str): The model name to determine the encoding.

        Returns:
            list[str]: A list of text chunks.
        """
        max_tokens = self.ACCEPTABLE_MODELS[model]["max_tokens"]
        tokens = self._tokenize_text(text, model)
        tokenized_chunks = self._chunk_tokens(tokens, max_tokens, self.CHUNK_OVERLAP)
        encoding = tiktoken.encoding_for_model(model)
        return [encoding.decode(chunk) for chunk in tokenized_chunks]

    @validate_model_name
    def _call_gpt(self, model: str, system_prompt: str, instruction: str) -> str:
        """
        Calls the GPT model with specified prompts and returns the completion.

        Args:
            model (str): The model name to use for generating completion.
            system_prompt (str): The system prompt for the model.
            instruction (str): The user instruction for the model.

        Returns:
            str: The generated completion text.
        """
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
        )
        return completion.choices[0].message.content

    @validate_model_name
    def summarize_text(
        self,
        text: str,
        model: str = "gpt-3.5-turbo",
        custom_system_prompt: Optional[str] = None,
        custom_instruction: Optional[str] = None,
    ) -> str:
        """
        Summarizes the given text using the specified model. Does not handle chunking.

        Args:
            text (str): The text to be summarized.
            model (str): The model name to use for summarization.
            custom_system_prompt (Optional[str]): Custom system prompt for the model. If None, uses the default prompt.
            custom_instruction (Optional[str]): Custom user instruction for the model. If None, uses the default prompt.
                The text will be automatically appended to the instruction.

        Returns:
            str: The generated summary.
        """
        system_prompt = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        instruction_with_text = (
            f"{custom_instruction}\n{text}" if custom_instruction else f"{self.DEFAULT_INSTRUCTION_PROMPT}\n{text}"
        )

        summary = self._call_gpt(model, system_prompt, instruction_with_text)
        self.recent_experiment = {
            "model": model,
            "system_prompt": system_prompt,
            "instruction": custom_instruction or self.DEFAULT_INSTRUCTION_PROMPT,
            "summary": summary,
            "text": text,
        }
        return summary

    @validate_model_name
    def summarize_text_with_chunking(
        self,
        text: str,
        summarizer_model: str = "gpt-3.5-turbo",
        combiner_model: str = "gpt-4o",
        custom_combiner_prompt: Optional[str] = None,
    ) -> str:
        """
        Summarizes the given text by chunking it and then combining the chunk summaries.
        By default, gpt-3.5-turbo is used for summarizing chunks and gpt-4o for combining summaries.

        Args:
            text (str): The text to be summarized.
            summarizer_model (str): The model name to use for summarizing chunks.
            combiner_model (str): The model name to use for combining summaries.
            custom_combiner_prompt (Optional[str]): Custom prompt for combining summaries. If None, uses the default prompt.

        Returns:
            str: The combined summary.
        """
        chunks = self.chunk_text(text, summarizer_model)

        appended_summaries = ""
        for chunk in chunks:
            appended_summaries += self.summarize_text(chunk, model=summarizer_model)
            appended_summaries += "\n"

        combined_summary = self.summarize_text(
            text=appended_summaries,
            custom_instruction=custom_combiner_prompt or self.DEFAULT_COMBINER_PROMPT,
            model=combiner_model,
        )
        return combined_summary

    @validate_model_name
    def summarize_book(
        self,
        output_filename: str = "book_summary.md",
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> None:
        """
        Summarizes the entire book and saves the summary to a file.

        Args:
            output_filename (str): The filename to save the book summary.
            model (str): The model name to use for summarization.
            system_prompt (Optional[str]): The system prompt for the model.
            instruction (Optional[str]): The user instruction for the model.
        """
        with open(output_filename, "w") as file:
            for index, chapter in enumerate(self.chapters):
                summary = self.summarize_text_with_chunking(chapter, model, system_prompt, instruction)
                file.write(f"## Chapter {index + 1}\n")
                file.write(summary)
                file.write("\n\n")
        print(f"Book summary saved to {output_filename}")

    def log_recent_experiment(self, filename: str = "prompting_log.md") -> None:
        """
        Logs the most recent experiment to a file.

        Args:
            filename (str): The filename to save the log entry. If None, "prompting_log.md" is used.
        """
        if not self.recent_experiment:
            print("No recent experiment to log.")
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "a") as file:
            file.write(f"## Log Entry - {timestamp}\n")
            file.write(f"**Model:** {self.recent_experiment['model']}\n")
            file.write(f"**System Prompt:** {self.recent_experiment['system_prompt']}\n")
            file.write(f"**Instruction:** {self.recent_experiment['instruction']}\n")
            file.write(f"**Summary:**\n{self.recent_experiment['summary']}\n")
            file.write("\n---\n")


# Example usage
if __name__ == "__main__":
    summarizer = BookSummarizer("The Road to Wigan Pier.epub")
    chapters = summarizer.chapters

    # Summarize a single chapter
    chapter_summary = summarizer.summarize_text(chapters[0])
    print(chapter_summary)

    # Summarize the entire book
    summarizer.summarize_book("The_Road_to_Wigan_Pier_Summary.md")

    # Use a custom prompt
    custom_system_prompt = "You are an expert in economic history analyzing George Orwell's perspectives."
    custom_instruction = (
        "Highlight the key economic arguments Orwell makes in this chapter. Provide examples and evidence he uses."
    )
    custom_summary = summarizer.summarize_text(
        chapters[1], system_prompt=custom_system_prompt, custom_instruction=custom_instruction
    )
    print(custom_summary)

    # Log the most recent experiment
    summarizer.log_recent_experiment("custom_prompting_log.md")
