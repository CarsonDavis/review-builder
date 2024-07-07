import pytest
from book_summarizer import CostCalculator


@pytest.fixture
def valid_model_name() -> str:
    return list(CostCalculator.VALID_MODELS.keys())[0]


@pytest.fixture
def invalid_model_name() -> str:
    return "invalid-model"


@pytest.fixture
def text_sample() -> str:
    return "This is a sample text for testing token count and cost calculation."


def test_validate_model_name_valid(valid_model_name: str) -> None:
    calculator = CostCalculator(valid_model_name)
    assert calculator.model_name == valid_model_name


def test_validate_model_name_invalid(invalid_model_name: str) -> None:
    with pytest.raises(ValueError, match=f"No cost data for {invalid_model_name}"):
        CostCalculator(invalid_model_name)


def test_count_tokens(valid_model_name: str, text_sample: str) -> None:
    """
    checks that num_tokens is updated correctly when count_tokens is called
    and that the number of tokens returned is equal to the length of the encoded text.
    """
    calculator = CostCalculator(valid_model_name)
    token_count = calculator.count_tokens(text_sample)
    assert calculator.num_tokens == token_count
    assert calculator.num_tokens == len(calculator.encoding.encode(text_sample))


def test_get_cost_per_token(valid_model_name: str) -> None:
    calculator = CostCalculator(valid_model_name)
    cost_per_token = calculator._get_cost_per_token()
    assert cost_per_token == calculator.VALID_MODELS[valid_model_name]["cost_per_token"]


def test_calculate_cost(valid_model_name: str, text_sample: str) -> None:
    calculator = CostCalculator(valid_model_name)
    cost = calculator.calculate_cost(text_sample)
    expected_token_count = len(calculator.encoding.encode(text_sample))
    expected_cost = expected_token_count * calculator._get_cost_per_token()
    assert cost == expected_cost


def test_calculate_cost_empty_string(valid_model_name: str) -> None:
    calculator = CostCalculator(valid_model_name)
    cost = calculator.calculate_cost("")
    assert cost == 0.0
    assert calculator.num_tokens == 0


if __name__ == "__main__":
    pytest.main()
