import os
import logging
from typing import Dict, List
from huggingface_hub import hf_hub_download, list_repo_tree
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

    def _get_language_from_speaker(self, speaker_label: str) -> str:
        """Extract language from speaker label (e.g., 'chinese_female' -> 'chinese')"""
        parts = speaker_label.rsplit('_', 1)
        return parts[0] if len(parts) > 1 else speaker_label

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(f"Loading MiniMax TTS dataset: {self.name} for languages: {self.languages}")

        # Load prompt transcriptions: filename -> prompt_text
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
                        audio_name = parts[0].strip()
                        p_text = parts[1].strip()
                        prompt_texts[audio_name] = p_text
        except Exception as e:
            logger.warning(f"Failed to load prompt_text.txt: {e}")

        # Build speaker map: speaker_label -> (audio_path, prompt_filename)
        # Download speaker audio files from the speaker/ folder
        speaker_map = {}  # speaker_label -> audio_path
        speaker_filenames = {}  # speaker_label -> filename (for prompt_text lookup)

        # Get all speakers needed from the text files first
        speakers_needed = set()
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
                        if len(parts) >= 2:
                            speaker_label = parts[0].strip()
                            speakers_needed.add(speaker_label)
            except Exception as e:
                logger.warning(f"Error reading text file for language {lang}: {e}")

        # Download prompt audio for each speaker from speaker/{language}/{speaker_label}/
        for speaker_label in speakers_needed:
            language = self._get_language_from_speaker(speaker_label)
            speaker_folder = f"speaker/{language}/{speaker_label}"
            
            try:
                # List files in the speaker folder to find the prompt audio
                files = list(list_repo_tree(
                    self.name,
                    path_in_repo=speaker_folder,
                    repo_type="dataset"
                ))
                
                # Find the mp3 file
                for f in files:
                    if f.path.endswith('.mp3'):
                        audio_path = hf_hub_download(
                            repo_id=self.name,
                            filename=f.path,
                            repo_type="dataset"
                        )
                        speaker_map[speaker_label] = audio_path
                        speaker_filenames[speaker_label] = os.path.basename(f.path)
                        logger.info(f"Downloaded prompt audio for {speaker_label}: {audio_path}")
                        break
            except Exception as e:
                logger.warning(f"Failed to download speaker audio for {speaker_label}: {e}")

        logger.info(f"Built speaker_map with {len(speaker_map)} speakers: {list(speaker_map.keys())[:5]}...")

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

                        # Get prompt text using the filename
                        prompt_filename = speaker_filenames.get(speaker_label, "")
                        prompt_text = prompt_texts.get(prompt_filename, "")

                        results.append({
                            "WavPath": speaker_map[speaker_label],
                            "prompt_audio": speaker_map[speaker_label],
                            "prompt_text": prompt_text,
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
