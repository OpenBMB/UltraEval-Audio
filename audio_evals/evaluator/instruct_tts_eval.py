import json
import logging
import os
import re
from typing import Dict, Optional

from audio_evals.evaluator.base import Evaluator

logger = logging.getLogger(__name__)

_PROMPT_PLACEHOLDER = "<此处插入待评测的语音风格描述>"
_DEFAULT_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "instruct_tts_eval_prompt.txt"
)


def extract_json_from_text(text):
    """Extract JSON object from text, handling multiple possible formats"""
    logging.info(f"Original response content:\n{text}")

    # Method 1: If JSON is wrapped in markdown code blocks
    try:
        if "```json" in text and "```" in text:
            json_text = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if json_text:
                parsed = json.loads(json_text.group(1))
                logging.info(f"Successfully extracted using method 1: {parsed}")
                return parsed
    except Exception as e:
        logging.warning(f"Method 1 extraction failed: {e}")

    # Method 2: Find JSON objects in text
    try:
        json_pattern = re.compile(r'\{(?:[^{}]|"(?:\\.|[^"\\])*")*\}')
        matches = json_pattern.findall(text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "一致性" in parsed:
                    logging.info(f"Successfully extracted using method 2: {parsed}")
                    return parsed
            except:
                continue
    except Exception as e:
        logging.warning(f"Method 2 extraction failed: {e}")

    # Method 3: Try to load the entire text directly
    try:
        parsed = json.loads(text.strip())
        logging.info(f"Successfully extracted using method 3: {parsed}")
        return parsed
    except Exception as e:
        logging.warning(f"Method 3 extraction failed: {e}")

    logging.error("All JSON extraction methods failed")
    return None


class InstructTTSEvalGemini(Evaluator):
    """
    Evaluator for InstructTTSEval using the framework's Gemini model via registry.

    Builds a PromptStruct containing the filled evaluation prompt and the generated
    audio file, calls the registered Gemini model, and parses the binary consistency
    judgement (一致性: true/false).

    Requires:
        GOOGLE_API_KEY  environment variable (consumed by audio_evals.models.google.Gemini).

    Args:
        model_name  : Registry name of the Gemini model (default: gemini-2.5-pro).
        prompt_file : Path to the evaluation prompt template.  Defaults to the
                      bundled ``instruct_tts_eval_prompt.txt``.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        prompt_file: str = _DEFAULT_PROMPT_FILE,
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

        with open(prompt_file, "r", encoding="utf-8") as fh:
            self.template = fh.read()

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        audio_path = str(pred)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Generated audio file not found: {audio_path}")

        instruction = kwargs.get("instruction", "")
        instruction_type = kwargs.get("instruction_type", "")

        prompt_text = self.template.replace(_PROMPT_PLACEHOLDER, instruction)

        prompt_struct = [
            {
                "role": "user",
                "contents": [
                    {"type": "text", "value": prompt_text},
                    {"type": "audio", "value": audio_path},
                ],
            }
        ]

        for retry in range(5):
            try:
                response = self.model.inference(
                    prompt_struct,
                    temperature=0,
                    responseMimeType="text/plain",
                    safetySettings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ],
                )
                result = extract_json_from_text(response)
                if result and "一致性" in result:
                    return {
                        "gemini_score": (
                            (1 if result["一致性"] is True else 0)
                            if result["一致性"] is not None
                            else None
                        ),
                        "raw_output": response,
                        "instruction_type": instruction_type,
                    }
                else:
                    logging.error(
                        f"Failed to extract consistency field from text: {response}"
                    )

            except Exception as e:
                logging.error(
                    f"[Audio: {audio_path}] Error occurred during processing: {str(e)}"
                )

        return {
            "gemini_score": None,
            "instruction_type": instruction_type,
            "raw_output": response,
        }
