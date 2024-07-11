## Project Overview
I listen to a lot of nonfiction. At the end of a book, I like to consolidate all my thoughts alongside the author's arguments and quotes.

It's very hard to come back to a 700 page book after reading it and remember all the author's points and what I thought about them.


### Goals
I'd like a system to:
- give chapter by chapter summaries
- help me find and remember key arguments, facts, and quotes

The idea is to use the summaries as an aide to engaging with the author's key points and to stimulate my own memory.

### Restrictions
The cost of the entire experience -- summarizing, finding quotes, ideally even some back and forth with the llm -- should cost less than $2 per book.

### Pricing
For reference, here is the GPT pricing structure. I have also provided cost breakdowns for summarization of both a [short](example_book_statistics_short.md) and [long](example_book_statistics_long.md) book.

<img width="981" alt="image" src="https://github.com/CarsonDavis/review-builder/assets/14339518/d495f99d-c815-43cf-94c9-a0f1a284ea8f">

## Quick Start Tutorial
So far, I've built out the summarization portion of the project. It makes good summaries, especially with 4o, but I think it could benefit from some additional promt engineering.

This repository contains a few useful classes, including EpubExtractor, CostCalculator, BookAnalyzer, and BookSummarizer. These classes are designed to help you extract text from EPUB files, calculate costs for processing text, analyze book statistics, and summarize chapters.

### Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install ebooklib beautifulsoup4 nltk openai python-dotenv tiktoken
```

### BookSummarizer

The `BookSummarizer` uses the OpenAI API to summarize chapters from an EPUB file.

It comes with some nice features:
- parallel processing of chapters
- dual-model, chunk-based summarization of large texts
- fine-tuneable with custom prompts

#### Setup

Ensure you have your [OpenAI API key](https://platform.openai.com/api-keys) stored in a `.env` file:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

#### Usage

```python
from book_summarizer import BookSummarizer

# Initialize with the path to your EPUB file
summarizer = BookSummarizer("path/to/your/book.epub")

# Summarize a single chapter
chapter_summary = summarizer.summarize_text(summarizer.chapters[0])
print(chapter_summary)

# Summarize the entire book and save to a markdown file
summarizer.summarize_book("book_summary.md")
```

If you are fine-tuning the summarization prompts, you can provide a custom system prompt and instruction and then log the results.

```python
from book_summarizer import BookSummarizer
from book_summarizer.llm_core import GPT4O

summarizer = BookSummarizer("path/to/your/book.epub")

# Use a custom prompt for summarization
custom_system_prompt = "You are an expert in economic history analyzing George Orwell's perspectives."
custom_instruction = (
    "Highlight the key economic arguments Orwell makes in this chapter. Provide examples and evidence he uses."
)
summary = summarizer.summarize_text(
    text=summarizer.chapters[1],
    model=GPT4O(), # you can override the default 3.5 by passing a different model
    system_prompt=custom_system_prompt,
    instruction=custom_instruction,
)
print(summary)
```

#### Metadata Extraction and Worthiness
"Chapters" in an ebook are really more like sections, and often include the Title page, Index, and other pieces which don't make sense to summarize. Additionally, GPT-3.5-turbo will often hallucinate summaries when given a Copyright page or other non-content text.

Worthiness is evaluated during the summarization process, but you can also evaluate it independently:

```python
from book_summarizer import BookSummarizer
from book_summarizer.llm_core import GPT4O, GPT35Turbo

summarizer = BookSummarizer("path/to/your/book.epub")

for index, chapter in enumerate(summarizer.chapters):
    title=summarizer._deduce_chapter_title(
        chapter_text=chapter,
        characters=500,
        model=GPT4O(),
    )
    worthiness=summarizer._deduce_worthiness(
        chapter_text=chapter,
        characters=500,
        model=GPT35Turbo(),
    )
    print(f"{index}: {worthiness} - {title}")
```


# Log the most recent experiment
```python
summarizer.log_recent_experiment("custom_prompting_log.md")
```

### BookAnalyzer

The `BookAnalyzer` class analyzes the text extracted from an EPUB file, providing word and token counts, word frequencies, and cost calculations for processing the book with different models. If you are worried about costs, you may want to run this before using the `BookSummarizer`.

I have provided example outputs for both a [short](example_book_statistics_short.md) and [long](example_book_statistics_long.md) book.

#### Usage

```python
from book_summarizer import BookAnalyzer

# Initialize with the path to your EPUB file
analyzer = BookAnalyzer("path/to/your/book.epub")

# The primary output is a well-formatted markdown file containing costs, word counts, and frequencies
analyzer.write_statistics("book_stats.md")
```

You can also retrieve specific information from the analyzer object:

```python
analyzer.word_counts()
analyzer.token_counts()
analyzer.word_frequencies()
analyzer.calculate_cost(GPT35Turbo())
```


### CostCalculator

You can use the  `CostCalculator` class independently to calculate the cost of processing arbitrary text with different language models.

#### Usage

```python
from book_summarizer import CostCalculator
from book_summarizer.llm_core import GPT35Turbo

calculator = CostCalculator(GPT35Turbo())

# Calculate the cost of a sample text
text = "Sample text to calculate cost."
cost = calculator.calculate_cost(text)
print(f"Cost to process text: ${cost:.6f}")
```

### EpubExtractor

The `EpubExtractor` extracts and removes excess newlines from a EPUB files. It works at the level of chapters.

#### Usage

```python
from book_summarizer import EpubExtractor

# Initialize with the path to your EPUB file
extractor = EpubExtractor("path/to/your/book.epub")

# The extracted chapters are stored in the chapters attribute
extractor.chapters

# Save chapters to a text file. By default it saves to your original filename.txt.
extractor.save("output.txt")
```
