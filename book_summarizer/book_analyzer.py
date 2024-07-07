import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Tuple
import sys
import os
from .epub_extractor import EpubExtractor
from .cost_calculator import (
    CostCalculator,
)

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt")


class BookAnalyzer:
    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.extractor = EpubExtractor(epub_path)
        self.chapters = self.extractor._get_chapters()
        self.tokenized_chapters = [word_tokenize(chapter) for chapter in self.chapters]

    def _generate_default_save_path(self) -> str:
        return os.path.splitext(self.epub_path)[0] + "_stats.md"

    def get_word_counts(self) -> Tuple[int, List[int]]:
        total_word_count = sum(len(tokens) for tokens in self.tokenized_chapters)
        chapter_word_counts = [len(tokens) for tokens in self.tokenized_chapters]
        return total_word_count, chapter_word_counts

    def get_token_counts(self) -> Dict[str, Tuple[int, List[int]]]:
        token_counts = {}
        for model in CostCalculator.model_costs:
            calculator = CostCalculator(model)
            total_token_count = sum(len(calculator.encoding.encode(chapter)) for chapter in self.chapters)
            chapter_token_counts = [len(calculator.encoding.encode(chapter)) for chapter in self.chapters]
            token_counts[model] = (total_token_count, chapter_token_counts)
        return token_counts

    def get_word_frequency(self) -> Dict[str, int]:
        all_tokens = [token for tokens in self.tokenized_chapters for token in tokens]
        frequency = Counter(all_tokens)
        sorted_frequency = dict(frequency.most_common())
        return sorted_frequency

    def calculate_book_cost(self, model_name: str) -> float:
        full_text = " ".join([" ".join(chapter) for chapter in self.chapters])
        calculator = CostCalculator(model_name)
        return calculator.calculate_cost(full_text)

    def write_word_statistics(self, save_path: str = None) -> None:
        save_path = save_path or self._generate_default_save_path()
        total_word_count, chapter_word_counts = self.get_word_counts()
        token_counts = self.get_token_counts()
        word_frequencies = self.get_word_frequency()

        models = list(token_counts.keys())

        with open(save_path, "w") as file:
            file.write(f"# Book Statistics\n\n")

            file.write("\n## Overview\n\n")
            file.write(f"Total Word Count: {total_word_count:,}\n\n")

            file.write("| Model | Cost |\n")
            file.write("|-------|------|\n")
            for model in CostCalculator.model_costs:
                cost = self.calculate_book_cost(model)
                file.write(f"| {model} | ${cost:.2f} |\n")
            file.write("\n")

            file.write(f"## Word and Token Counts per Chapter\n\n")
            file.write("| Chapter | Words | " + " | ".join(f"{model} Tokens" for model in models) + " |\n")
            file.write("|---------|------------|" + " | ".join(["-------------"] * len(models)) + "|\n")

            for i, word_count in enumerate(chapter_word_counts):
                formatted_word_count = f"{word_count:,}"
                token_counts_row = " | ".join(f"{token_counts[model][1][i]:,}" for model in models)
                file.write(f"| {i+1} | {formatted_word_count} | {token_counts_row} |\n")

            file.write(f"## Word Frequencies (First 200 Words)\n\n")
            file.write("| Word | Frequency |\n")
            file.write("|------|-----------|\n")
            for i, (word, frequency) in enumerate(word_frequencies.items()):
                if i >= 200:
                    break
                file.write(f"| {word} | {frequency} |\n")
            file.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python epub_word_analyzer.py <epub_path>")
        sys.exit(1)

    epub_path = sys.argv[1]

    try:
        analyzer = BookAnalyzer(epub_path)
        output_file = os.path.splitext(epub_path)[0] + "_word_stats.md"
        analyzer.write_word_statistics(output_file)
        print(f"Word statistics extracted and saved to {output_file}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
