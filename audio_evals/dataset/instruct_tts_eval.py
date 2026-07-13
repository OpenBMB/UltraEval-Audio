import logging
from typing import Dict, List, Optional

from audio_evals.dataset.huggingface import Huggingface

logger = logging.getLogger(__name__)

INSTRUCTION_TYPES = ["APS", "DSD", "RP"]


class InstructTTSEvalDataset(Huggingface):
    """
    Dataset for InstructTTSEval benchmark (https://github.com/KexinHUANG19/InstructTTSEval).

    Subclasses ``Huggingface`` and overrides ``load()`` to expand each sample
    into one row per instruction type (APS / DSD / RP), giving up to 3 000 rows
    per language split.

    Each resulting row contains:
        id              : "<original_id>_<instruction_type>"
        text            : text to be synthesised
        instruction     : natural-language style instruction
        instruction_type: "APS" | "DSD" | "RP"
        WavPath         : local path to the saved reference audio (16 kHz WAV)
    """

    def __init__(
        self,
        default_task: str,
        ref_col: str,
        split: str = "en",
        local_path: str = "",
        instruction_types: Optional[List[str]] = None,
        col_aliases: Optional[Dict[str, str]] = None,
    ):
        # Alias reference_audio -> audio so the parent's save_audio_to_local works
        merged_aliases = {"reference_audio": "audio"}
        if col_aliases:
            merged_aliases.update(col_aliases)
        super().__init__(
            name="CaasiHUANG/InstructTTSEval",
            default_task=default_task,
            ref_col=ref_col,
            split=split,
            local_path=local_path,
            col_aliases=merged_aliases,
        )
        self.instruction_types = instruction_types or INSTRUCTION_TYPES

    def load(self, limit: int = 0) -> List[Dict[str, any]]:
        # Parent handles download, audio saving, and WavPath injection
        raw_rows = super().load()

        rows = []
        for sample in raw_rows:
            for inst_type in self.instruction_types:
                inst_value = sample.get(inst_type)
                if inst_value is None:
                    continue

                if isinstance(inst_value, str):
                    instruction = inst_value
                elif isinstance(inst_value, dict):
                    instruction = inst_value.get("instruction", "")
                else:
                    instruction = str(inst_value)

                rows.append(
                    {
                        "id": f"{sample['id']}_{inst_type}",
                        "text": sample["text"],
                        "instruction": instruction,
                        "instruction_type": inst_type,
                        "WavPath": sample.get("WavPath", ""),
                    }
                )

        logger.info(
            "Loaded %d rows from InstructTTSEval split=%s "
            "(%d samples × %d instruction types)",
            len(rows),
            self.split,
            len(raw_rows),
            len(self.instruction_types),
        )

        return rows[:limit] if limit > 0 else rows
