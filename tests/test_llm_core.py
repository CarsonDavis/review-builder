from book_summarizer.llm_core import GPT4O, GPT35Turbo


# I'd probably like to test more stuff, like whether the call method works...
def test_gpt35turbo_properties():
    """Validates that GPT-3.5-Turbo properties are correctly set."""
    gpt = GPT35Turbo()
    assert gpt.model_name == "gpt-3.5-turbo"
    assert gpt.max_tokens == 16385
    assert gpt.cost_per_token == 0.5 / 1000000


def test_gpt4o_properties():
    """Validates that GPT-4O properties are correctly set."""
    gpt = GPT4O()
    assert gpt.model_name == "gpt-4o"
    assert gpt.max_tokens == 128000
    assert gpt.cost_per_token == 5 / 1000000
