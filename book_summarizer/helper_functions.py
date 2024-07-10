import re
from collections.abc import Callable
from functools import wraps


def find_boolean_in_string(text: str) -> bool:
    """
    Searches for the words 'true' or 'false' in any capitalization within a string.
    Returns the associated boolean value. If no match is found, returns True.

    Args:
        text (str): The input string to search within.

    Returns:
        bool: The boolean value associated with the found word, or True if no match is found.
    """
    # Compile regex patterns to match 'true' or 'false' in any capitalization
    true_pattern = re.compile(r"\btrue\b", re.IGNORECASE)
    false_pattern = re.compile(r"\bfalse\b", re.IGNORECASE)

    if true_pattern.search(text):
        return True
    elif false_pattern.search(text):
        return False

    return True


def validate_model_name(func: Callable) -> Callable:
    """
    Decorator to validate any argument containing the word 'model'. Against
    a dict of VALID_MODELS within the class. Raises a ValueError if the model is not found.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract all keyword arguments that contain 'model' in their name
        model_kwargs = {k: v for k, v in kwargs.items() if "model" in k}
        # Extract all positional arguments that contain 'model' in their name (assuming standard naming conventions)
        model_args = [arg for name, arg in zip(func.__code__.co_varnames[1:], args) if "model" in name]

        # Combine all found model arguments
        all_models = list(model_kwargs.values()) + model_args

        for model in all_models:
            if model not in self.VALID_MODELS:
                raise ValueError(f"Model {model} is not known. Choose from {list(self.VALID_MODELS.keys())}.")

        return func(self, *args, **kwargs)

    return wrapper
