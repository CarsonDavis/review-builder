import tiktoken


class CostCalculator:
    """
    A class used to calculate the cost of processing text with different openai models.

    Attributes
    ----------
    model_costs : dict
        A dictionary containing the cost per token for different models.
    model_name : str
        The name of the model to use for calculations.
    encoding : object
        The encoding object used to tokenize text for the specified model.
    token_count : int
        Number of tokens in the text. Will be 0 until the text is tokenized.
    """

    model_costs: dict[str, float] = {
        "gpt-4o": 5 / 1000000,
        "gpt-3.5-turbo": 0.5 / 1000000,
    }

    def __init__(self, model_name: str):
        """
        Validates the model name and uses tiktoken to get the encoding object for the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to use for calculations.
        """
        self.model_name: str = model_name
        self._validate_model_name()
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.num_tokens: int = 0

    def _validate_model_name(self) -> None:
        """
        Validates if the provided model name exists in the model costs dictionary.

        Raises
        ------
        ValueError
            If the model name does not exist in the model costs dictionary.
        """
        if self.model_name not in self.model_costs:
            raise ValueError(f"No cost data for {self.model_name}")

    def _get_cost_per_token(self) -> float:
        """
        Retrieves the cost per token for the initialized model.

        Returns
        -------
        float
            The cost per token.
        """
        return self.model_costs[self.model_name]

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the provided text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        -------
        int
            The number of tokens.
        """
        tokens = self.encoding.encode(text)
        self.num_tokens = len(tokens)
        return self.num_tokens

    def calculate_cost(self, text: str) -> float:
        """
        Calculates the cost of processing the provided text based on the number of tokens.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        -------
        float
            The cost of processing the text.
        """
        tokens = self.count_tokens(text)
        cost = tokens * self._get_cost_per_token()
        return cost


# Example usage:
# calculator = CalculateCost("gpt-3.5-turbo")
# print(calculator.calculate_cost("Some sample text to encode."))
