import tiktoken

from book_summarizer.llm_core import LLMClient


class CostCalculator:
    """
    A class used to calculate the cost of processing text with different OpenAI models.

    Attributes
    ----------
    model_client : LLMClient
        An instance of an LLMClient subclass used for calculations.
    encoding : object
        The encoding object used to tokenize text for the specified model.
    num_tokens : int
        Number of tokens in the text. Will be 0 until the text is tokenized.
    """

    def __init__(self, model_client: LLMClient):
        """
        Initializes the CostCalculator with an LLMClient instance and uses tiktoken
        to get the encoding object for the specified model.

        Parameters
        ----------
        model_client : LLMClient
            An instance of an LLMClient subclass.
        """
        self.model_client: LLMClient = model_client
        self.encoding = tiktoken.encoding_for_model(model_client.model_name)
        self.num_tokens: int = 0

    def _get_cost_per_token(self) -> float:
        """
        Retrieves the cost per token for the initialized model.

        Returns
        -------
        float
            The cost per token.
        """
        return self.model_client.cost_per_token

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
# gpt_35_turbo_client = GPT35Turbo()
# calculator = CostCalculator(gpt_35_turbo_client)
# print(calculator.calculate_cost("Some sample text to encode."))
