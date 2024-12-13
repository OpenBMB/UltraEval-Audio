import ast
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Process(ABC):
    @abstractmethod
    def __call__(self, answer: str) -> str:
        raise NotImplementedError()


class ContentExtract(Process):

    def __call__(self, answer: str) -> str:
        try:
            answer = answer.strip()
            if answer.startswith("```json"):
                answer = answer[7:-3].strip()
            elif answer.startswith("```"):
                answer = answer[3:-3].strip()
            return json.loads(answer)["content"]
        except Exception as e:
            try:
                return ast.literal_eval(answer)["content"]
            except Exception as e:
                logger.warning(f"process {answer} fail: {str(e)}")
        return answer


class JsonExtract(Process):
    """
    Extract a specific key from a json string.
    the key is specified by the `extract_key` parameter.
    if the key is not found, return the `default_value` if specified,
    otherwise raise a KeyError.
    """

    def __init__(self, extract_key: str, default_value: str = None):
        """
        Initialize the JsonExtract process.
        Args:
            extract_key: required, the key to extract from the json string.
            default_value: optional, the default value to return if the key is not found.

        Returns: JsonExtract object.

        """
        self.extract_key = extract_key
        self.default_value = default_value

    def __call__(self, answer: str) -> any:
        """
        Extract the value of the `extract_key` from the json string `answer`.
        Args:
            answer: required, the json string to extract the value from.

        Returns: any, the value of the `extract_key` in the json string `answer`.

        """
        if isinstance(answer, str):
            d = json.loads(answer.strip())
        elif isinstance(answer, dict):
            d = answer
        else:
            raise ValueError("answer must be a string or a dict")

        if self.default_value is not None:
            return d.get(self.extract_key, self.default_value)
        return d[self.extract_key]
