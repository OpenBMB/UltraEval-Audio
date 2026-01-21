import os
import re
import logging
from typing import Dict, List
import soundfile as sf
from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download
from audio_evals.dataset.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)


class MiniMaxTTSDataset(BaseDataset):
    def __init__(
        self,
        name: str = "MiniMaxAI/TTS-Multilingual-Test-Set",
        default_task: str = "tts",
        ref_col: str = "text",
        col_aliases: Dict[str, str] = None,
        language: str = None,
        languages: List[str] = None,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        self.name = name

        if language:
            self.languages = [language]
        elif languages:
            self.languages = languages
        else:
            self.languages = [
                "arabic", "cantonese", "chinese", "czech", "dutch", "english",
                "finnish", "french", "german", "greek", "hindi", "indonesian",
                "italian", "japanese", "korean", "polish", "portuguese", "romanian",
                "russian", "spanish", "thai", "turkish", "ukrainian", "vietnamese"
            ]

    def _strip_numeric_prefix(self, label: str) -> str:
        """Remove numeric prefix from label (e.g., '0arabic_female' -> 'arabic_female')"""
        return re.sub(r'^\d+', '', label)

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(f"Loading MiniMax TTS dataset: {self.name} for languages: {self.languages}")

        try:
            ds = load_dataset(self.name, trust_remote_code=True)
            if isinstance(ds, DatasetDict):
                ds = ds["train"]
        except Exception as e:
            logger.error(f"Failed to load dataset {self.name}: {e}")
            return []

        save_path = f"raw/{self.name}/speaker"
        os.makedirs(save_path, exist_ok=True)

        # Build speaker map: speaker_label -> wav_path
        speaker_map = {}
        for example in ds:
            if "label" not in example:
                continue

            # Get label name
            if isinstance(example["label"], int):
                label_name = ds.features["label"].int2str(example["label"])
            else:
                label_name = str(example["label"])

            # Remove numeric prefix (e.g., '0arabic_female' -> 'arabic_female')
            speaker_label = self._strip_numeric_prefix(label_name)

            audio = example["audio"]
            wav_path = os.path.join(save_path, f"{speaker_label}.wav")
            if not os.path.exists(wav_path):
                sf.write(wav_path, audio["array"], audio["sampling_rate"])
                logger.info(f"Saved audio to {wav_path}")
            speaker_map[speaker_label] = wav_path

        logger.info(f"Built speaker_map with {len(speaker_map)} speakers: {list(speaker_map.keys())[:5]}...")

        # Load prompt transcriptions
        prompt_texts = {}
        try:
            prompt_text_file = hf_hub_download(
                repo_id=self.name,
                filename="speaker/prompt_text.txt",
                repo_type="dataset"
            )
            with open(prompt_text_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "|" not in line:
                        continue
                    parts = line.split("|", 1)
                    if len(parts) >= 2:
                        spk_id = parts[0].strip()
                        p_text = parts[1].strip()
                        prompt_texts[spk_id] = p_text
        except Exception as e:
            logger.warning(f"Failed to load prompt_text.txt: {e}")

        # Load test sentences for each language
        results = []
        for lang in self.languages:
            try:
                text_file = hf_hub_download(
                    repo_id=self.name,
                    filename=f"text/{lang}.txt",
                    repo_type="dataset"
                )

                with open(text_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "|" not in line:
                            continue
                        parts = line.split("|", 1)
                        if len(parts) < 2:
                            continue
                        speaker_label = parts[0].strip()
                        text = parts[1].strip()

                        if speaker_label not in speaker_map:
                            logger.warning(f"Speaker '{speaker_label}' not found in speaker_map")
                            continue

                        results.append({
                            "WavPath": speaker_map[speaker_label],
                            "prompt_audio": speaker_map[speaker_label],
                            "prompt_text": prompt_texts.get(speaker_label, ""),
                            "text": text,
                            "ans": text,
                            "language": lang,
                            "speaker": speaker_label
                        })
            except Exception as e:
                logger.warning(f"Error loading language {lang}: {e}")

        if limit > 0:
            results = results[:limit]

        logger.info(f"Loaded {len(results)} samples from MiniMax TTS dataset")
        return results
