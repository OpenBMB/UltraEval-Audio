import logging
import os
from typing import Dict, List, Optional

import soundfile as sf
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk

from audio_evals.dataset.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)


def save_audio_to_local(ds: Dataset, save_path: str):
    """
    save audio file to local.
    :param ds:
    :param save_path:
    :return:
    """

    def save_audio(example, index):
        audio_array = example["audio"]["array"]
        output_path = os.path.join(save_path, f"{index}.wav")
        example["WavPath"] = output_path
        d = os.path.dirname(output_path)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(output_path):
            sf.write(output_path, audio_array, example["audio"]["sampling_rate"])
            logger.info(f"save audio to {output_path}")
        return example

    ds = ds.map(save_audio, with_indices=True)
    return ds


def load_audio_hf_dataset(name, subset=None, split="", local_path="", col_aliases=None):
    if col_aliases is None:
        col_aliases = {}
    if local_path:
        ds = load_from_disk(local_path)
    else:
        load_args = {"path": name}
        if subset:
            load_args["name"] = subset
        if split:
            load_args["split"] = split
        try:
            ds = load_dataset(**load_args, trust_remote_code=True)
        except Exception as e:
            logger.error(f"load args is {load_args}load dataset error: {e}")
            raise e

    for k, v in col_aliases.items():
        if v in ds.column_names:
            raise ValueError(f"col_aliases conflict with existing column name: {v}")
        ds = ds.rename_column(k, v)

    def conv2ds(ds):
        save_path = f"raw/{name}/"
        if subset:
            save_path += f"{subset}/"
        if split:
            save_path += f"{split}/"

        os.makedirs(save_path, exist_ok=True)
        return list(save_audio_to_local(ds, save_path))

    if isinstance(ds, DatasetDict):
        result = []
        for k in ds:
            reload_ds = {
                "name": name,
                "subset": subset,
                "split": k,
                "local_path": local_path,
                "col_aliases": col_aliases,
            }
            result.extend(load_audio_hf_dataset(**reload_ds))
        return result
    return conv2ds(ds)


class Huggingface(BaseDataset):
    def __init__(
        self,
        name: str,
        default_task: str,
        ref_col: str,
        subset: Optional[str] = None,
        split: str = "",
        local_path: str = "",
        col_aliases: Dict[str, str] = None,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        self.name = name
        self.subset = subset
        self.split = split
        self.local_path = local_path

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(
            "start load data, it will take a while for download dataset when first load dataset"
        )
        res = load_audio_hf_dataset(
            self.name, self.subset, self.split, self.local_path, self.col_aliases
        )
        return res[:limit] if limit > 0 else res
