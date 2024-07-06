from openai import OpenAI
from epub_extractor import EpubExtractor
from dotenv import load_dotenv
import datetime
import os

# Load the API key which OpenAI will read from the environment
load_dotenv()
client = OpenAI()


class BookSummarizer:
    DEFAULT_SYSTEM_PROMPT = "You are a skilled textual analyst that can synthesize the key concepts in long text and identify crucial details to retain."
    DEFAULT_INSTRUCTION_PROMPT = "Make a list of the key points made by the author in the following chapter. Under each point, list out the reasons or evidence given."

    def __init__(self, epub_path):
        self.epub_path = epub_path
        self.extractor = EpubExtractor(epub_path)
        self.chapters = self.extractor.chapters
        self.recent_experiment = None

    def summarize_text(self, text, model="gpt-3.5-turbo", system_prompt=None, instruction=None):
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        instruction_with_text = (
            f"{instruction}\n{text}" if instruction else f"{self.DEFAULT_INSTRUCTION_PROMPT}\n{text}"
        )

        summary = self._call_gpt(model, system_prompt, instruction_with_text)
        self.recent_experiment = {
            "model": model,
            "system_prompt": system_prompt,
            "instruction": instruction or self.DEFAULT_INSTRUCTION_PROMPT,
            "summary": summary,
            "text": text,
        }
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

    def log_recent_experiment(self, filename="prompting_log.md"):
        """Logs the most recent experiment to a file"""
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
        chapters[1], system_prompt=custom_system_prompt, instruction=custom_instruction
    )
    print(custom_summary)

    # Log the most recent experiment
    summarizer.log_most_recent_experiment("custom_prompting_log.md")
