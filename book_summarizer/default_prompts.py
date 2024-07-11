DEFAULT_PROMPTS = {
    "summarizer_prompt": (
        "You are a skilled textual analyst that can synthesize the key concepts in "
        "long text and identify crucial details to retain."
    ),
    "summarizer_instruction": (
        "Make a list of the key points made by the author in the following chapter. "
        "Under each point, list out the reasons or evidence given."
    ),
    "combiner_prompt": (
        "I will provide you with several summaries of different parts of a chapter. "
        "Combine these summaries into one single summary. The final summary should "
        "have a list of the key points made by the author in the following chapter. "
        "Under each point, list out the reasons or evidence given."
    ),
    "chapter_prompt": (
        "Your job is to deduce the title of a section of a book based on its content. "
        "It may be a Title page, Index, Chapter, Copyright Page or any other part of a book. "
        "Respond only with the title you have deduced and nothing else."
        "If the chapter has a number, put it before the chapter title, as in Chapter 2: A New Dawn"
        "If the content is not a clearly defined section of a book, write 'unknown'"
    ),
    "chapter_instruction": (
        "Here are the first 500 characters of a section of a book. Please deduce the title of this section:"
    ),
    "worthiness_prompt": (
        "Your job is to evaluate a sample of text to see if it is part of a section worth summarizing."
        "You respond only with boolean values: 'True' if the text is worth summarizing, 'False' if it is not."
    ),
    "worthiness_instruction": (
        "Here is are the first 500 characters of a section of a book. "
        "Respond True if the section is a chapter, preface, or other section worth summarizing. "
        "Respond False if the section is a title page, table of contents, or otherwise not worth summarizing."
    ),
}
