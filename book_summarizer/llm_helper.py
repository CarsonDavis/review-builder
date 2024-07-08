import datetime
import time
from collections.abc import Callable

import tiktoken
from dotenv import load_dotenv
from joblib import Parallel, delayed
from openai import OpenAI

from .epub_extractor import EpubExtractor

# Load the API key which OpenAI will read from the environment
load_dotenv()
client = OpenAI()


def validate_model_name(func: Callable) -> Callable:
    """Decorator to validate any argument containing the word 'model'."""

    def wrapper(self, *args, **kwargs):
        # Extract all keyword arguments that contain 'model' in their name
        model_kwargs = {k: v for k, v in kwargs.items() if "model" in k}
        # Extract all positional arguments that contain 'model' in their name (assuming standard naming conventions)
        model_args = [arg for name, arg in zip(func.__code__.co_varnames[1:], args) if "model" in name]

        # Combine all found model arguments
        all_models = list(model_kwargs.values()) + model_args

        for model in all_models:
            if model not in self.VALID_MODELS:
                raise ValueError(f"Model {model} is not known. Choose from {list(self.VALID_MODELS.keys())}.")

        return func(self, *args, **kwargs)

    return wrapper


def summarize_chapter(
    chapter,
    summarizer_model,
    custom_summarizer_prompt,
    custom_summarizer_instruction,
    combiner_model,
    custom_combiner_prompt,
    summarizer_func,
    max_retries=5,
):
    retry_count = 0
    while retry_count < max_retries:
        try:
            summary = summarizer_func(
                text=chapter,
                summarizer_model=summarizer_model,
                custom_summarizer_prompt=custom_summarizer_prompt,
                custom_summarizer_instruction=custom_summarizer_instruction,
                combiner_model=combiner_model,
                custom_combiner_prompt=custom_combiner_prompt,
            )
            return summary
        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower():
                retry_count += 1
                wait_time = 2**retry_count  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: {e}"
    return f"Error: Rate limit exceeded after {max_retries} retries."


class BookSummarizer:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a skilled textual analyst that can synthesize the key concepts in "
        "long text and identify crucial details to retain."
    )
    DEFAULT_INSTRUCTION_PROMPT = (
        "Make a list of the key points made by the author in the following chapter. "
        "Under each point, list out the reasons or evidence given."
    )
    DEFAULT_COMBINER_PROMPT = (
        "I will provide you with several summaries of different parts of a chapter. "
        "Combine these summaries into one single summary. The final summary should "
        "have a list of the key points made by the author in the following chapter. "
        "Under each point, list out the reasons or evidence given."
    )
    DEFAULT_SUMMARIZER_MODEL = "gpt-3.5-turbo"
    DEFAULT_COMBINER_MODEL = "gpt-4o"

    SUMMARY_SIZE = 1500  # gpt-3.5-turbo summaries for 12k chapters were 500 tokens. 1500 should be safe.
    VALID_MODELS = {
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
        max_tokens = self.VALID_MODELS[model]["max_tokens"]
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
        model: str | None = None,
        custom_system_prompt: str | None = None,
        custom_instruction: str | None = None,
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
        model = model or self.DEFAULT_SUMMARIZER_MODEL
        system_prompt = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        instruction_with_text = (
            f"{custom_instruction}\n{text}" if custom_instruction else f"{self.DEFAULT_INSTRUCTION_PROMPT}\n{text}"
        )

        summary = self._call_gpt(model=model, system_prompt=system_prompt, instruction=instruction_with_text)
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
        summarizer_model: str | None = None,
        custom_summarizer_prompt: str | None = None,
        custom_summarizer_instruction: str | None = None,
        combiner_model: str | None = None,
        custom_combiner_prompt: str | None = None,
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
        summarizer_model = summarizer_model or self.DEFAULT_SUMMARIZER_MODEL
        chunks = self.chunk_text(
            text=text,
            model=summarizer_model,
        )

        appended_summaries = ""
        for chunk in chunks:
            appended_summaries += self.summarize_text(
                text=chunk,
                model=summarizer_model,
                custom_system_prompt=custom_summarizer_prompt,
                custom_instruction=custom_summarizer_instruction,
            )
            appended_summaries += "\n"

        if len(chunks) > 1:
            combined_summary = self.summarize_text(
                text=appended_summaries,
                model=combiner_model or self.DEFAULT_COMBINER_MODEL,
                custom_system_prompt=custom_summarizer_prompt or self.DEFAULT_SYSTEM_PROMPT,
                custom_instruction=custom_combiner_prompt or self.DEFAULT_COMBINER_PROMPT,
            )
        else:
            combined_summary = appended_summaries

        return combined_summary

    # @validate_model_name
    def summarize_book(
        self,
        output_filename: str = "book_summary.md",
        summarizer_model: str = "gpt-3.5-turbo",
        custom_summarizer_prompt: str | None = None,
        custom_summarizer_instruction: str | None = None,
        combiner_model: str = "gpt-4o",
        custom_combiner_prompt: str | None = None,
    ) -> None:
        """
        Summarizes the entire book and saves the summary to a file.

        Args:
            output_filename (str): The filename to save the book summary.
            summarizer_model (str): The model name to use for summarization.
            custom_summarizer_prompt (Optional[str]): Custom system prompt for the summarizer model.
            custom_summarizer_instruction (Optional[str]): Custom user instruction for the summarizer model.
            combiner_model (str): The model name to use for combining summaries.
            custom_combiner_prompt (Optional[str]): Custom prompt for the combiner model.
        """

        results = Parallel(n_jobs=-1)(
            delayed(summarize_chapter)(
                chapter,
                summarizer_model,
                custom_summarizer_prompt,
                custom_summarizer_instruction,
                combiner_model,
                custom_combiner_prompt,
                self.summarize_text_with_chunking,
            )
            for chapter in self.chapters[:5]
        )

        with open(output_filename, "w") as file:
            for index, summary in enumerate(results):
                if summary.startswith("Error:"):
                    print(f"Error summarizing chapter {index + 1}: {summary}")
                else:
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
        chapters[1],
        model="gpt-3.5-turbo",
        custom_system_prompt=custom_system_prompt,
        custom_instruction=custom_instruction,
    )
    print(custom_summary)

    # Log the most recent experiment
    summarizer.log_recent_experiment("custom_prompting_log.md")
