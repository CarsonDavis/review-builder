from openai import OpenAI
from epub_extractor import EpubExtractor
from cost_calculator import CostCalculator
import datetime

import datetime
from dotenv import load_dotenv
import os

# this loads the api key which OpenAI will read from the environment
load_dotenv()
client = OpenAI()


class BookSummarizer:
    DEFAULT_SYSTEM_PROMPT = "You are a skilled textual analyst that can synthesize the key concepts in long text and identify crucial details to retain."
    DEFAULT_INSTRUCTION_PROMPT = "Make a list of the key points made by the author in the following chapter. Under each point, list out the reasons or evidence given."

    def __init__(self, epub_path):
        self.epub_path = epub_path
        self.extractor = EpubExtractor(epub_path)
        self.chapters = self.extractor.extract()

    def summarize_text(self, text, model="gpt-3.5-turbo", system_prompt=None, instruction=None):
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        instruction = f"{instruction}\n{text}" or f"{self.DEFAULT_INSTRUCTION_PROMPT}\n{text}"

        summary = self._call_gpt(text, model, system_prompt, instruction)
        self._log_experiment(text, model, system_prompt, instruction, summary)
        return summary

    def summarize_book(
        self, output_filename="book_summary.md", model="gpt-3.5-turbo", system_prompt=None, instruction=None
    ):
        with open(output_filename, "w") as file:
            for index, chapter in enumerate(self.chapters):
                summary = self.summarize_text(chapter, model, system_prompt, instruction)
                file.write(f"## Chapter {index + 1}\n")
                file.write(summary)
                file.write("\n\n")
        print(f"Book summary saved to {output_filename}")

    def _call_gpt(self, model, system_prompt, instruction):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
        )
        return completion.choices[0].message.content

    def _log_experiment(self, model, system_prompt, instruction, summary, filename="prompting_log.md"):
        """this function allows you to test many prompts an log the results to a file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, "a") as file:
            file.write(f"## Log Entry - {timestamp}\n")
            file.write(f"**Model:** {model}\n")
            file.write(f"**System Prompt:** {system_prompt}\n")
            file.write(f"**Instruction:** {instruction}\n")
            file.write(f"**Summary:**\n{summary}\n")
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


# Example usage
chapter_text = "Your chapter text here"
summary = get_chapter_summary(chapter_text)
print(summary)


summary = get_chapter_summary(
    chapter=chapters[15],
    model="gpt-3.5-turbo",
    system_prompt="You are a skilled textual analyst that can sythesize the key concepts in long text and identify crucial details to retain.",
    instruction=f"Make a list of the key points made by the author in the following chapter. Under each point, list out the reasons or evidence given {chapters[15]}",
)
print(summary)
