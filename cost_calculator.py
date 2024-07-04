import tiktoken


class CalculateCost:
    model_costs = {
        "gpt-4o": 5 / 1000000,
        "gpt-3.5-turbo": 0.5 / 1000000,
    }

    def __init__(self, model_name):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self._validate_model_name()

    def _validate_model_name(self):
        if self.model_name not in self.model_costs:
            raise ValueError(f"No cost data for {self.model_name}")

    def _get_cost_per_token(self):
        return self.model_costs[self.model_name]

    def count_tokens(self, text):
        tokens = self.encoding.encode(text)
        return len(tokens)

    def calculate_cost(self, text):
        tokens = self.count_tokens(text)
        cost = tokens * self._get_cost_per_token()
        return cost


# Example usage:
# calculator = CalculateCost("gpt-3.5-turbo")
# print(calculator.calculate_cost("Some sample text to encode."))
