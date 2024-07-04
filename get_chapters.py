import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

book = epub.read_epub("The Road to Wigan Pier.epub")


def clean_text(text):
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    return text


def get_chapters(book):
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode("utf-8")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text()
            text = clean_text(text)
            if text:
                chapters.append(text)


def write_to_txt(chapters, filename):
    with open(filename, "w") as file:
        for chapter in chapters:
            file.write(chapter + "\n\n")
