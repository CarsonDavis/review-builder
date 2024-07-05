import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import sys
import os


class EpubExtractor:
    """
    Extracts chapterized text from EPUB files.
    Extracted chapters are accessed with the `chapters` attribute and can be saved to a text file with the `save` method.

    Attributes
    ----------
    epub_file_path : str
        The file path to the EPUB file.
    chapters : list of str
        The chapters extracted from the EPUB file.
    """

    def __init__(self, epub_file_path: str):
        """
        Validates the incoming file path and extracts the chapters from the EPUB file.

        Parameters
        ----------
        epub_file_path : str
            The file path to the EPUB file.
        """
        self.epub_file_path = epub_file_path
        self._validate_file_path()
        self.chapters = self._get_chapters()

    def _validate_file_path(self) -> None:
        """
        Validates if the provided file path exists.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(self.epub_file_path):
            raise FileNotFoundError(f"File {self.epub_file_path} does not exist.")

    def _clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing extra newlines and trimming spaces.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            The cleaned text.
        """
        text = re.sub(r"\n+", "\n", text)
        text = text.strip()
        return text

    def _get_chapters(self) -> list[str]:
        """
        Extracts and cleans the text from all chapters in the EPUB file.

        Returns
        -------
        list of str
            A list of strings, each representing a chapter.
        """
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
        """
        Writes the extracted chapters to a text file.

        Parameters
        ----------
        chapters : list of str
            A list of chapter texts.
        filename : str
            The name of the output text file.
        """
        with open(filename, "w") as file:
            for chapter in chapters:
                file.write(chapter + "\n\n")

    def _generate_output_path(self) -> str:
        """
        Generates a default output path for the extracted text file by changing the extension from epub to txt.

        Returns
        -------
        str
            The default output path.
        """
        return os.path.splitext(self.epub_file_path)[0] + ".txt"

    def save(self, output_path=None) -> str:
        """
        Saves the extracted chapters to a text file.

        Parameters
        ----------
        output_path : str, optional
            Custom path for the output file. If not provided, a default path from _generate_output_path() is used.

        Returns
        -------
        str
            The path to the saved text file.
        """
        output_path = output_path or self._generate_output_path()
        self._write_to_txt(self.chapters, output_path)
        return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <epub_file_path>")
        sys.exit(1)

    epub_file_path = sys.argv[1]

    try:
        extractor = EpubExtractor(epub_file_path)
        output_file = extractor.save()
        print(f"Chapters extracted and saved to {output_file}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
