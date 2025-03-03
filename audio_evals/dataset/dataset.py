import json
import os.path
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class Dataset(ABC):
    def __init__(self, default_task: str, ref_col: str, col_aliases=None):
        if col_aliases is None:
            col_aliases = {}
        self.col_aliases = col_aliases
        self.task_name = default_task
        self.ref_col = ref_col

    @abstractmethod
    def load(self, limit=0) -> List[Dict[str, any]]:
        raise NotImplementedError()

    def resume_from(self, f_name: str):
        from audio_evals.dataset.resume import ResumeDataset

        return ResumeDataset(self, f_name)


class JsonlFile(Dataset):
    def __init__(self, f_name: str, default_task: str, ref_col: str, col_aliases=None):
        super().__init__(default_task, ref_col, col_aliases)
        self.f_name = f_name

    def add_col_alias(self, df):
        for k, v in self.col_aliases.items():
            if v in df.columns:
                raise ValueError(f"Column alias {v} already exists in the dataframe")
            df[v] = df[k]
        return df

    def load(self, limit=0) -> List[Dict[str, any]]:
        df = pd.read_json(self.f_name, lines=True)
        if limit > 0:
            df = df[:limit]
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")


class RelativePath(JsonlFile):
    def __init__(
        self,
        f_name: str,
        default_task: str,
        ref_col: str,
        file_path_prefix: str,
        col_aliases=None,
    ):
        super().__init__(f_name, default_task, ref_col, col_aliases)
        if not file_path_prefix.endswith("/"):
            file_path_prefix += "/"
        self.file_path = file_path_prefix

    def load(self, limit=0) -> List[Dict[str, any]]:
        df = pd.read_json(self.f_name, lines=True)
        if limit > 0:
            df = df[:limit]

        def abs_path(x):
            temp = os.path.join(self.file_path, str(x))
            if os.path.exists(temp) and os.path.isfile(temp):
                return temp
            return x

        df["WavPath"] = df["WavPath"].progress_apply(abs_path)
        df = self.add_col_alias(df)
        return df.to_dict(orient="records")
