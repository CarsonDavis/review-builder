import datetime
import os
import time

from dotenv import load_dotenv
from joblib import Parallel, delayed

from book_summarizer.default_prompts import DEFAULT_PROMPTS
from book_summarizer.epub_extractor import EpubExtractor
from book_summarizer.llm_core import GPT4O, GPT35Turbo, LLMClient
from book_summarizer.text_processing import TextProcessor, find_boolean_in_string

# Load the API key which OpenAI will read from the environment
load_dotenv()


def summarize_chapter(
    chapter: str,
    summarizer_model: LLMClient,
    custom_summarizer_prompt: str,
    custom_summarizer_instruction: str,
    combiner_model: LLMClient,
    custom_combiner_prompt: str,
    summarizer_func,
    max_retries: int = 5,
) -> str:
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
    DEFAULT_SUMMARIZER_MODEL = GPT35Turbo()
    DEFAULT_COMBINER_MODEL = GPT4O()

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
        self.text_processor = TextProcessor()
        self.default_prompts = DEFAULT_PROMPTS

    def _default_save_path(self) -> str:
        return os.path.splitext(self.epub_path)[0] + "_summary.md"

    def _deduce_worthiness(
        self,
        chapter_text: str,
        characters: int,
        model: LLMClient = DEFAULT_SUMMARIZER_MODEL,
        system_prompt: str | None = None,
        instruction: str | None = None,
    ) -> str:
        system_prompt = system_prompt or self.default_prompts["worthiness_prompt"]
        instruction = instruction or self.default_prompts["worthiness_instruction"]

        instruction_with_text = f"{instruction}\n{chapter_text[:characters]}"
        worthiness_boolean = model.call(system_prompt, instruction_with_text)

        return find_boolean_in_string(worthiness_boolean)

    def _deduce_chapter_title(
        self,
        chapter_text: str,
        characters: int,
        model: LLMClient = DEFAULT_COMBINER_MODEL,
        system_prompt: str | None = None,
        instruction: str | None = None,
    ) -> str:
        """
        Deduces the chapter title from the beginning of the chapter text.

        Args:
            chapter_text (str): The text of the chapter.
            characters (int): The number of characters in the chapter to use for the deduction.

        Returns:
            str: The deduced chapter title.
        """
        system_prompt = system_prompt or self.default_prompts["chapter_prompt"]
        instruction = instruction or self.default_prompts["chapter_instruction"]

        instruction_with_text = f"{instruction}\n{chapter_text[:characters]}"
        chapter_title = model.call(system_prompt, instruction_with_text)

        return chapter_title

    def summarize_text(
        self,
        text: str,
        model: LLMClient | None = None,
        custom_system_prompt: str | None = None,
        custom_instruction: str | None = None,
    ) -> str:
        """
        Summarizes the given text using the specified model. Does not handle chunking.

        Args:
            text (str): The text to be summarized.
            model (Optional[LLMClient]): The model to use for summarization.
            custom_system_prompt (Optional[str]): Custom system prompt for the model. If None, uses the default prompt.
            custom_instruction (Optional[str]): Custom user instruction for the model. If None, uses the default prompt.
                The text will be automatically appended to the instruction.

        Returns:
            str: The generated summary.
        """
        model = model or self.DEFAULT_SUMMARIZER_MODEL
        system_prompt = custom_system_prompt or self.default_prompts["summarizer_prompt"]
        instruction_with_text = (
            f"{custom_instruction}\n{text}"
            if custom_instruction
            else f"{self.default_prompts['summarizer_instruction']}\n{text}"
        )

        summary = model.call(system_prompt, instruction_with_text)
        self.recent_experiment = {
            "model": model.model_name,
            "system_prompt": system_prompt,
            "instruction": custom_instruction or self.default_prompts["summarizer_instruction"],
            "summary": summary,
            "text": text,
        }
        return summary

    def summarize_text_with_chunking(
        self,
        text: str,
        summarizer_model: LLMClient | None = None,
        custom_summarizer_prompt: str | None = None,
        custom_summarizer_instruction: str | None = None,
        combiner_model: LLMClient | None = None,
        custom_combiner_prompt: str | None = None,
    ) -> str:
        """
        Summarizes the given text by chunking it and then combining the chunk summaries.
        By default, gpt-3.5-turbo is used for summarizing chunks and gpt-4o for combining summaries.

        Args:
            text (str): The text to be summarized.
            summarizer_model (Optional[LLMClient]): The model to use for summarizing chunks.
            combiner_model (Optional[LLMClient]): The model to use for combining summaries.
            custom_combiner_prompt (Optional[str]): Custom prompt for combining summaries. If None, uses the default prompt.

        Returns:
            str: The combined summary.
        """
        summarizer_model = summarizer_model or self.DEFAULT_SUMMARIZER_MODEL
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
                custom_system_prompt=custom_summarizer_prompt,
                custom_instruction=custom_summarizer_instruction,
            )
            appended_summaries += "\n"

        if len(chunks) > 1:
            combined_summary = self.summarize_text(
                text=appended_summaries,
                model=combiner_model or self.DEFAULT_COMBINER_MODEL,
                custom_system_prompt=custom_summarizer_prompt or self.default_prompts["summarizer_prompt"],
                custom_instruction=custom_combiner_prompt or self.default_prompts["combiner_prompt"],
            )
        else:
            combined_summary = appended_summaries

        return combined_summary

    def gpt_chapter_metadata(self, chapter: str, deduction_limit: int) -> dict:
        title = self._deduce_chapter_title(chapter, deduction_limit)
        worthiness = self._deduce_worthiness(chapter, deduction_limit)
        return {"title": title, "worthiness": worthiness, "chapter": chapter}

    def summarize_book(
        self,
        output_filename: str | None = None,
        summarizer_model: LLMClient = GPT35Turbo(),
        custom_summarizer_prompt: str | None = None,
        custom_summarizer_instruction: str | None = None,
        combiner_model: LLMClient = GPT4O(),
        custom_combiner_prompt: str | None = None,
    ) -> None:
        """
        Summarizes the entire book and saves the summary to a file.

        Args:
            output_filename (Optional[str]): The filename to save the book summary.
            summarizer_model (LLMClient): The model to use for summarization.
            custom_summarizer_prompt (Optional[str]): Custom system prompt for the summarizer model.
            custom_summarizer_instruction (Optional[str]): Custom user instruction for the summarizer model.
            combiner_model (LLMClient): The model to use for combining summaries.
            custom_combiner_prompt (Optional[str]): Custom prompt for the combiner model.
        """

        chapter_metadata = Parallel(n_jobs=-1)(
            delayed(self.gpt_chapter_metadata)(chapter, 500) for chapter in self.chapters
        )

        # Filter chapters based on worthiness
        worthy_chapters = [
            (index, meta["chapter"]) for index, meta in enumerate(chapter_metadata) if meta["worthiness"]
        ]

        # Parallelize the summarization process
        summarized_results = Parallel(n_jobs=-1)(
            delayed(summarize_chapter)(
                chapter,
                summarizer_model,
                custom_summarizer_prompt,
                custom_summarizer_instruction,
                combiner_model,
                custom_combiner_prompt,
                self.summarize_text_with_chunking,
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
        model=GPT4O(),
        custom_system_prompt=custom_system_prompt,
        custom_instruction=custom_instruction,
    )
    print(custom_summary)

    # Log the most recent experiment
    summarizer.log_recent_experiment("custom_prompting_log.md")
