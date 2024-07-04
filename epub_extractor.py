import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import sys
import os


class EpubExtractor:
    def __init__(self, epub_file_path: str):
        self.epub_file_path = epub_file_path
        self.chapters = self._get_chapters()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n+", "\n", text)
        text = text.strip()
        return text

    def _get_chapters(self) -> list[str]:
        chapters = []
        book = epub.read_epub(self.epub_file_path)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8")
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text()
                text = self._clean_text(text)
                if text:
                    chapters.append(text)
        return chapters

    def _write_to_txt(self, chapters: list[str], filename: str) -> None:
        with open(filename, "w") as file:
            for chapter in chapters:
                file.write(chapter + "\n\n")

    def extract_and_save(self) -> str:
        if not os.path.exists(self.epub_file_path):
            raise FileNotFoundError(f"File {self.epub_file_path} does not exist.")

        output_file = os.path.splitext(self.epub_file_path)[0] + ".txt"
        self._write_to_txt(self.chapters, output_file)
        return output_file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <epub_file_path>")
        sys.exit(1)

    epub_file_path = sys.argv[1]

    try:
        extractor = EpubExtractor(epub_file_path)
        output_file = extractor.extract_and_save()
        print(f"Chapters extracted and saved to {output_file}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
