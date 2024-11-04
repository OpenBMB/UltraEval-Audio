import json
import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List


class Dataset(ABC):
    def __init__(self, default_task: str, ref_col: str):
        self.task_name = default_task
        self.ref_col = ref_col

    @abstractmethod
    def load(self) -> List[Dict[str, any]]:
        raise NotImplementedError()


class JsonlFile(Dataset):
    def __init__(self, f_name: str, default_task: str, ref_col: str):
        super().__init__(default_task, ref_col)
        self.f_name = f_name

    def load(self) -> List[Dict[str, any]]:
        with open(self.f_name) as f:
            return [json.loads(line) for line in f]


class RelativePath(JsonlFile):
    def __init__(
        self, f_name: str, default_task: str, ref_col: str, file_path_prefix: str
    ):
        super().__init__(f_name, default_task, ref_col)
        if not file_path_prefix.endswith("/"):
            file_path_prefix += "/"
        self.file_path = file_path_prefix

    def load(self) -> List[Dict[str, any]]:
        res = []
        with open(self.f_name) as f:
            for line in f:
                doc = json.loads(line)
                for k, v in doc.items():
                    # automatically convert relative paths to absolute paths
                    temp = os.path.join(self.file_path, str(v))
                    if os.path.exists(temp) and os.path.isfile(temp):
                        doc[k] = temp
                res.append(doc)
        return res
