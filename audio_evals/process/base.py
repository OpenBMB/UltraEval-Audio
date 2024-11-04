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
