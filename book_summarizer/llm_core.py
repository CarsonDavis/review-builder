import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Load the API key which OpenAI will read from the environment
load_dotenv()
CLIENT = OpenAI()


class LLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        pass

    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        pass

    @abstractmethod
    def call(self, system_prompt: str, instruction: str) -> str:
        pass


def retry_handler(func: Callable, *args, max_retries: int = 5, **kwargs) -> Any:
    retry_count = 0
    while retry_count < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower():
                retry_count += 1
                wait_time = 3**retry_count  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: {e}"
    return f"Error: Rate limit exceeded after {max_retries} retries."


class GPTClient(LLMClient):
    client = CLIENT

    def call(self, system_prompt: str, instruction: str, max_retries: int = 5) -> str:
        return retry_handler(self._make_request, system_prompt, instruction, max_retries=max_retries)

    def _make_request(self, system_prompt: str, instruction: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
        )
        return self._parse_response(response)

    def _parse_response(self, response):
        return response.choices[0].message.content


class GPT35Turbo(GPTClient):
    model_name = "gpt-3.5-turbo"
    max_tokens = 16385
    cost_per_token = 0.5 / 1000000


class GPT4O(GPTClient):
    model_name = "gpt-4o"
    max_tokens = 128000
    cost_per_token = 5 / 1000000


class GPT4oMini(GPTClient):
    model_name = "gpt-4o-mini"
    max_tokens = 128000
    cost_per_token = 0.15 / 1000000
