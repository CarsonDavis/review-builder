import os
from functools import wraps

import weave
from dotenv import load_dotenv
from joblib import Parallel, delayed

from book_summarizer.default_prompts import DEFAULT_PROMPTS
from book_summarizer.epub_extractor import EpubExtractor
from book_summarizer.llm_core import GPT4O, GPT4oMini, LLMClient
from book_summarizer.text_processing import TextProcessor, find_boolean_in_string

# Load the API key which OpenAI will read from the environment
load_dotenv()


def conditional_wandb_log(func):
    """this wraps the weave.op() decorator so that it is only applied if the log_to_wandb attribute is set to True"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "log_to_wandb", False):
            # Apply the @weave.op() decorator
            decorated_func = weave.op()(func)
            return decorated_func(self, *args, **kwargs)
        else:
            # Call the original function
            return func(self, *args, **kwargs)

    return wrapper


class BookSummarizer:
    SUMMARY_SIZE = 1500  # gpt-3.5-turbo summaries for 12k chapters were 500 tokens. 1500 should be safe.
    CHUNK_OVERLAP = 50

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.extractor = EpubExtractor(epub_path)
        self.chapters = self.extractor.chapters
        self.log_to_wandb = False

    def _default_save_path(self) -> str:
        return os.path.splitext(self.epub_path)[0] + "_summary.md"

    def _deduce_worthiness(
        self,
        chapter_text: str,
        characters: int,
        model: LLMClient = GPT4oMini(),
        system_prompt: str = DEFAULT_PROMPTS["worthiness_prompt"],
        instruction: str = DEFAULT_PROMPTS["worthiness_instruction"],
    ) -> str:
        instruction_with_text = f"{instruction}\n{chapter_text[:characters]}"
        worthiness_boolean = model.call(system_prompt, instruction_with_text)
        return find_boolean_in_string(worthiness_boolean)

    def _deduce_chapter_title(
        self,
        chapter_text: str,
        characters: int,
        model: LLMClient = GPT4O(),
        system_prompt: str = DEFAULT_PROMPTS["chapter_prompt"],
        instruction: str = DEFAULT_PROMPTS["chapter_instruction"],
    ) -> str:
        """
        Deduces the chapter title from the beginning of the chapter text.

        Args:
            chapter_text (str): The text of the chapter.
            characters (int): The number of characters in the chapter to use for the deduction.

        Returns:
            str: The deduced chapter title.
        """
        instruction_with_text = f"{instruction}\n{chapter_text[:characters]}"
        chapter_title = model.call(system_prompt, instruction_with_text)
        return chapter_title

    def deduce_chapter_metadata(self, chapter: str, deduction_limit: int) -> dict:
        title = self._deduce_chapter_title(chapter, deduction_limit)
        worthiness = self._deduce_worthiness(chapter, deduction_limit)
        return {"title": title, "worthiness": worthiness, "chapter": chapter}

    def log_future_calls_to_wandb(self, project_name: str = "book-summarizer") -> None:
        """will log future calls of summarize_text to wandb."""
        weave.init(project_name)
        self.log_to_wandb = True

    @conditional_wandb_log
    def summarize_text(
        self,
        text: str,
        model: LLMClient = GPT4oMini(),
        system_prompt: str = DEFAULT_PROMPTS["summarizer_prompt"],
        instruction: str = DEFAULT_PROMPTS["summarizer_instruction"],
    ) -> str:
        """
        Summarizes the given text using the specified model. Does not handle chunking.

        Args:
            text (str): The text to be summarized.
            model (Optional[LLMClient]): The model to use for summarization.
            system_prompt (Optional[str]): Custom system prompt for the model. If None, uses the default prompt.
            instruction (Optional[str]): Custom user instruction for the model. If None, uses the default prompt.
                The text will be automatically appended to the instruction.

        Returns:
            str: The generated summary.
        """
        instruction_with_text = f"{instruction}\n{text}"
        summary = model.call(system_prompt, instruction_with_text)
        return summary

    def summarize_text_with_chunking(
        self,
        text: str,
        summarizer_model: LLMClient = GPT4oMini(),
        summarizer_prompt: str = DEFAULT_PROMPTS["summarizer_prompt"],
        summarizer_instruction: str = DEFAULT_PROMPTS["summarizer_instruction"],
        combiner_model: LLMClient = GPT4O(),
        combiner_prompt: str = DEFAULT_PROMPTS["combiner_prompt"],
    ) -> str:
        """
        Summarizes the given text by chunking it and then combining the chunk summaries.
        By default, gpt-3.5-turbo is used for summarizing chunks and gpt-4o for combining summaries.

        Args:
            text (str): The text to be summarized.
            summarizer_model (Optional[LLMClient]): The model to use for summarizing chunks.
            combiner_model (Optional[LLMClient]): The model to use for combining summaries.
            combiner_prompt (Optional[str]): Custom prompt for combining summaries. If None, uses the default prompt.

        Returns:
            str: The combined summary.
        """
        chunk_size = summarizer_model.max_tokens - self.SUMMARY_SIZE

        chunks = TextProcessor(summarizer_model).chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=self.CHUNK_OVERLAP,
        )

        appended_summaries = ""
        for chunk in chunks:
            appended_summaries += self.summarize_text(
                text=chunk,
                model=summarizer_model,
                system_prompt=summarizer_prompt,
                instruction=summarizer_instruction,
            )
            appended_summaries += "\n"

        if len(chunks) > 1:
            combined_summary = self.summarize_text(
                text=appended_summaries,
                model=combiner_model,
                system_prompt=summarizer_prompt,
                instruction=combiner_prompt,
            )
        else:
            combined_summary = appended_summaries

        return combined_summary

    def summarize_book(
        self,
        output_filename: str | None = None,
        summarizer_model: LLMClient = GPT4oMini(),
        summarizer_prompt: str = DEFAULT_PROMPTS["summarizer_prompt"],
        summarizer_instruction: str = DEFAULT_PROMPTS["summarizer_instruction"],
        combiner_model: LLMClient = GPT4O(),
        combiner_prompt: str = DEFAULT_PROMPTS["combiner_prompt"],
    ) -> None:
        """
        Summarizes the entire book and saves the summary to a file.

        Args:
            output_filename (Optional[str]): The filename to save the book summary.
            summarizer_model (LLMClient): The model to use for summarization.
            summarizer_prompt (Optional[str]): Custom system prompt for the summarizer model.
            summarizer_instruction (Optional[str]): Custom user instruction for the summarizer model.
            combiner_model (LLMClient): The model to use for combining summaries.
            combiner_prompt (Optional[str]): Custom prompt for the combiner model.
        """
        chapter_metadata = Parallel(n_jobs=-1)(
            delayed(self.deduce_chapter_metadata)(chapter, 500) for chapter in self.chapters
        )

        # Filter chapters based on worthiness
        worthy_chapters = [
            (index, meta["chapter"]) for index, meta in enumerate(chapter_metadata) if meta["worthiness"]
        ]

        # Parallelize the summarization process
        summarized_results = Parallel(n_jobs=-1)(
            delayed(self.summarize_text_with_chunking)(
                chapter,
                summarizer_model,
                summarizer_prompt,
                summarizer_instruction,
                combiner_model,
                combiner_prompt,
            )
            for _, chapter in worthy_chapters
        )

        # Mapping summaries back to the chapter indices
        summary_dict = {index: summary for (index, _), summary in zip(worthy_chapters, summarized_results)}

        with open(output_filename, "w") as file:
            for index, meta in enumerate(chapter_metadata):
                file.write(f"## {meta['title']}\n")
                summary = summary_dict.get(index, "Evaluated as not worth summarizing.")
                file.write(summary)
                file.write("\n\n")
        print(f"Book summary saved to {output_filename}")


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
    system_prompt = "You are an expert in economic history analyzing George Orwell's perspectives."
    instruction = (
        "Highlight the key economic arguments Orwell makes in this chapter. Provide examples and evidence he uses."
    )
    summary = summarizer.summarize_text(
        chapters[1],
        model=GPT4O(),
        system_prompt=system_prompt,
        instruction=instruction,
    )
    print(summary)
