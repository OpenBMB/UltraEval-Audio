import logging
from typing import Dict, List, Optional

from audio_evals.dataset.huggingface import Huggingface

logger = logging.getLogger(__name__)

# Category-level guidance from the official benchmark
# (https://github.com/boson-ai/EmergentTTS-Eval-public, Apache-2.0).
# It is exposed as an `instruction` column so the same dataset can be used
# with the benchmark's "strong prompting" mode through the shared
# `instruct-tts` prompt.
CATEGORY_DESCRIPTIONS = {
    "Emotions": (
        "Emotional expressiveness: Ensure a clear and distinct transition "
        "between quoted dialogues and narrative text. Deliver the quoted "
        "dialogues with high emotional expressiveness."
    ),
    "Paralinguistics": (
        "Paralinguistical cues: Express interjections, onomatopoeia, "
        "capitalization, vowel elongation, hyphenation/syllable stress, "
        "stuttering and pacing cues(elipses, punctuations, etc.) naturally "
        "and realistically."
    ),
    "Syntactic Complexity": (
        "Syntactical Complexity: Maintain clarity in complex sentence "
        "structures through appropriate prosody, pausing and stress to convey "
        "the intended meaning of the sentence very clearly. Handle homographs "
        "with appropriate pronunciation."
    ),
    "Foreign Words": (
        "Foreign words: Pronounce foreign words and phrases with their "
        "appropriate pronunciation or anglicized version, sound like a "
        "natural bi-lingual speaker doing smooth code-switching."
    ),
    "Questions": (
        "Questions: Apply the appropriate intonation pattern for "
        "interrogative sentences(questions) and declarative sentences."
    ),
    "Pronunciation": (
        "Complex Pronunciation: Pronounce currency, numerals, emails, "
        "passwords, urls, dates, times, phone numbers, street addresses, "
        "equations, initialisms, acronyms, tounge twisters(speak fast while "
        "maintaining clarity), etc. with precision, clarity and "
        "case-sensitivity wherever applicable."
    ),
}


class EmergentTTSEvalDataset(Huggingface):
    """
    Dataset for EmergentTTS-Eval
    (https://huggingface.co/datasets/bosonai/EmergentTTS-Eval).

    The official Hugging Face dataset stores 1,645 English test cases with the
    following columns:

        category            : one of the six benchmark categories
        text_to_synthesize  : text the TTS model must synthesize
        evolution_depth     : how many LLM expansion rounds produced the case
        language            : "en"
        audio               : gpt-4o-mini-tts (alloy) baseline audio

    ``Huggingface`` saves the embedded baseline audio to local WAV files and
    injects a ``WavPath`` column, which the win-rate evaluator compares the
    model output against.  ``text_to_synthesize`` is renamed to ``text`` to
    match the framework's prompt/model conventions.

    Two extra columns are added by :meth:`load`:

        id           : source row index (also seeds the A/B order)
        instruction  : official strong-prompting description
    """

    NAME = "bosonai/EmergentTTS-Eval"

    def __init__(
        self,
        default_task: str,
        ref_col: str,
        subset: str = "default",
        split: str = "train",
        local_path: str = "",
        col_aliases: Optional[Dict[str, str]] = None,
    ):
        merged_aliases = {"text_to_synthesize": "text"}
        if col_aliases:
            merged_aliases.update(col_aliases)
        super().__init__(
            name=self.NAME,
            default_task=default_task,
            ref_col=ref_col,
            subset=subset,
            split=split,
            local_path=local_path,
            col_aliases=merged_aliases,
        )

    def load(self, limit: int = 0) -> List[Dict[str, any]]:
        rows = super().load()
        for index, row in enumerate(rows):
            row["id"] = index
            row["instruction"] = CATEGORY_DESCRIPTIONS.get(row.get("category", ""), "")
        logger.info(
            "Loaded %d rows from %s (subset=%s, split=%s)",
            len(rows),
            self.name,
            self.subset,
            self.split,
        )
        return rows[:limit] if limit > 0 else rows
