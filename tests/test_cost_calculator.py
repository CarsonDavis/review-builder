import pytest
from book_summarizer import CostCalculator


@pytest.fixture
def valid_model_name():
    return list(CostCalculator.VALID_MODELS.keys())[0]


@pytest.fixture
def invalid_model_name():
    return "invalid-model"


@pytest.fixture
def text_sample():
    return "This is a sample text for testing token count and cost calculation."


def test_validate_model_name_valid(valid_model_name):
    calculator = CostCalculator(valid_model_name)
    assert calculator.model_name == valid_model_name


def test_validate_model_name_invalid(invalid_model_name):
    with pytest.raises(ValueError, match=f"No cost data for {invalid_model_name}"):
        CostCalculator(invalid_model_name)


def test_count_tokens(valid_model_name, text_sample):
    "tests that the token counter is at least giving a valid integer"
    calculator = CostCalculator(valid_model_name)
    token_count = calculator.count_tokens(text_sample)
    assert token_count > 0
    assert calculator.num_tokens == token_count


def test_get_cost_per_token(valid_model_name):
    calculator = CostCalculator(valid_model_name)
    cost_per_token = calculator._get_cost_per_token()
    assert cost_per_token == calculator.VALID_MODELS[valid_model_name]["cost_per_token"]


def test_calculate_cost(valid_model_name, text_sample):
    "tests that the cost calculator is at least giving a value"
    calculator = CostCalculator(valid_model_name)
    cost = calculator.calculate_cost(text_sample)
    assert cost > 0
    assert cost == calculator.num_tokens * calculator._get_cost_per_token()


if __name__ == "__main__":
    pytest.main()
