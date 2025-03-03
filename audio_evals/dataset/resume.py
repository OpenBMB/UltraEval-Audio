import json
from typing import List, Dict, Union

from audio_evals.dataset.dataset import Dataset
from tqdm import tqdm


class ResumeDataset(Dataset):
    def __init__(self, raw_dataset: Union[str, Dataset], resume_file: str):
        if isinstance(raw_dataset, str):
            from audio_evals.registry import registry

            raw_dataset = registry.get_dataset(raw_dataset)
        super().__init__(
            raw_dataset.task_name, raw_dataset.ref_col, raw_dataset.col_aliases
        )
        self.raw_dataset = raw_dataset
        self.resume_file = resume_file

    def load(self, limit=0) -> List[Dict[str, any]]:
        data = self.raw_dataset.load(limit)
        with open(self.resume_file, "r") as f:
            for line in tqdm(f):
                doc = json.loads(line)
                if doc["type"] == "error":
                    continue
                idx = int(doc["id"])
                if "eval_info" not in data[idx]:
                    data[idx]["eval_info"] = {}
                data[idx]["eval_info"].update({doc["type"]: doc["data"]})
        return data
