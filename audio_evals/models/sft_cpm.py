import json
import logging
import time
from typing import Dict
import uuid
from audio_evals.base import PromptStruct
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated
import select

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/minicpm_0_5B/main.py")
class SFTMiniCPMoAudio(OfflineModel):
    def __init__(self, path: str, sample_params: Dict = None, *args, **kwargs):
        self.command_args = {
            "path": path,
        }
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_content(self, content: Dict):
        assert "type" in content
        return {"type": content["type"], content["type"]: content["value"]}

    def _parse_role_content(self, role_content: Dict):
        for k in ["contents"]:
            if isinstance(role_content[k], list):
                role_content["content"] = [
                    self._parse_content(item) for item in role_content.pop(k)
                ]
            else:
                role_content["content"] = role_content.pop(k)
        return role_content

    def _inference(self, prompt: PromptStruct, **kwargs):
        system_prompt, user_prompt, audio = (
            "You are a helpful assistant. You can accept audio and text input and output "
            "voice and text.",
            "",
            "",
        )
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "text":
                        user_prompt = line["value"]
                    if line["type"] == "audio":
                        audio = line["value"]

            if content["role"] == "system":
                msg_line = {"role": "system", "content": []}
                for line in content["contents"]:
                    if line["type"] == "text":
                        system_prompt = line["value"]

        text = "<|im_start|>system\n{}<|im_end|> \n<|im_start|>user\n{}\n<|audio_start|><|audio|><|audio_end|><|im_end|> \n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>".format(
            system_prompt, user_prompt
        )

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        json_content = json.dumps({"text": text, "audio": audio}, ensure_ascii=False)
        request = f"{prefix}{json_content}\n"
        # Send request
        try:
            _, wlist, xlist = select.select(
                [], [self.process.stdin], [self.process.stdin], 60
            )
            if xlist:
                raise RuntimeError("Encodec stdin broken (select reported error).")
            if not wlist:
                raise TimeoutError("Timeout waiting for Encodec stdin.")
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("Encodec process stdin pipe is broken.")
        except Exception as e:
            raise RuntimeError(f"Error writing to Encodec process stdin: {e}")

        max_wait_time = 120  # Adjust timeout as needed for encoding
        start_time = time.time()
        response_line = None

        while time.time() - start_time < max_wait_time:
            try:
                reads, _, xlist = select.select(
                    [self.process.stdout, self.process.stderr],
                    [],
                    [self.process.stdout, self.process.stderr],
                    1.0,
                )
                if xlist:
                    raise RuntimeError(
                        "Encodec stdout/stderr broken (select reported error)."
                    )

                for read_stream in reads:
                    if read_stream is self.process.stderr:
                        error_output = self.process.stderr.readline().strip()
                        if error_output:
                            logger.error(f"Encodec stderr: {error_output}")
                    elif read_stream is self.process.stdout:
                        result = self.process.stdout.readline().strip()
                        logger.debug(f"Received from Encodec process: {result}")
                        if result:
                            if result.startswith(prefix):
                                response_line = result[len(prefix) :]
                                self.process.stdin.write(f"{prefix}ok\n")
                                self.process.stdin.flush()
                                res = json.loads(response_line)
                                if len(res) == 1:
                                    return res["text"]
                                return res
                            elif result.startswith("Error:"):
                                # Log other lines but don't treat as the response
                                raise RuntimeError(
                                    "cpm0-0.5B failed: {}".format(result)
                                )
                            else:
                                logger.info(result)

            except Exception as e:
                raise RuntimeError(f"Error reading from cpm0-0.5B process: {e}")

        if not response_line:
            raise TimeoutError(
                f"Timeout waiting for response from cpm0-0.5B process for request {uid}"
            )
