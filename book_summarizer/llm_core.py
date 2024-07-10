from abc import ABC, abstractmethod

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


class GPTClient(LLMClient):
    client = CLIENT

    def call(self, system_prompt: str, instruction: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
        )
        return completion.choices[0].message.content


class GPT35Turbo(GPTClient):
    model_name = "gpt-3.5-turbo"
    max_tokens = 16385
    cost_per_token = 0.5 / 1000000


class GPT4O(GPTClient):
    model_name = "gpt-4o"
    max_tokens = 128000
    cost_per_token = 5 / 1000000
