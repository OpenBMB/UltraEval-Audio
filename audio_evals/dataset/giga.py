import logging
from typing import List, Dict
from audio_evals.dataset.huggingface import Huggingface, load_audio_hf_dataset

logger = logging.getLogger(__name__)


conversational_filler = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
]
unk_tags = ["<UNK>", "<unk>"]
gigaspeech_punctuations = [
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
]
gigaspeech_garbage_utterance_tags = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
non_scoring_words = (
    conversational_filler
    + unk_tags
    + gigaspeech_punctuations
    + gigaspeech_garbage_utterance_tags
)


def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace("-", " ")

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return " ".join(remaining_words)


class GigaSpeechDataset(Huggingface):

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(
            "start load data, it will take a while for download dataset when first load dataset"
        )
        raw = load_audio_hf_dataset(
            self.name, self.subset, self.split, self.local_path, self.col_aliases
        )
        res = []
        for item in raw:
            item["text"] = asr_text_post_processing(item["text"])
            if item["text"]:
                res.append(item)
        if limit > 0:
            res = res[:limit]
        return res
