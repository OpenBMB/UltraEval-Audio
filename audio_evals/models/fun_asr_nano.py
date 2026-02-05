from audio_evals.models.offline_model import OfflineModel
from audio_evals.base import PromptStruct
from typing import Dict, Any, List, Union
from funasr import AutoModel
import json
from funasr.register import tables
import sys
import os
import importlib

LOCAL_REPO_PATH = "/home/arda/yanzhang/repo/Fun-ASR"

if LOCAL_REPO_PATH not in sys.path:
    sys.path.insert(0, LOCAL_REPO_PATH)
    print(f"[Info] Added local repo to sys.path: {LOCAL_REPO_PATH}")

FunASRNanoClass = None

try:
    
    import model
    if hasattr(model, "FunASRNano"):
        FunASRNanoClass = model.FunASRNano
        print("[Success] Imported FunASRNano from local 'model.py'")
    else:

        from funasr.models.fun_asr_nano.model import FunASRNano
        FunASRNanoClass = FunASRNano
        print("[Success] Imported FunASRNano from 'funasr.models.fun_asr_nano.model'")

except ImportError as e:
    print(f"[Warning] Import failed: {e}")
    try:
        from init_model import model as init_model_pkg
        if hasattr(init_model_pkg, "FunASRNano"):
            FunASRNanoClass = init_model_pkg.FunASRNano
            print("[Success] Imported FunASRNano from 'init_model/model.py'")
    except Exception as e2:
        print(f"[Error] Could not find FunASRNano class anywhere: {e2}")


if FunASRNanoClass:
    tables.register("model_classes", "FunASRNano")(FunASRNanoClass)
    print("[Info] Registered 'FunASRNano' class successfully.")
else:
    print("[Fatal] FunASRNano Class is missing! Inference will fail.")

class FunASRNanoModel(OfflineModel):
    def __init__(
        self, 
        path: str,              
        is_chat: bool = False,   
        device: str = "cuda:0",  
        sample_params: Dict[str, Any] = None
    ):
        self.is_chat = is_chat
        self.sample_params = sample_params or {}

        print(f"Loading FunASR model from: {path}")

        # 初始化 AutoModel
        self.model = AutoModel(
            model=path,
            trust_remote_code=True, # 允许加载
            device=device,
            disable_update=True,
        )

    def inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio_file = ""
        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio_file = line["value"]
                        break

        try:
            res = self.model.generate(audio_file, language="中文", itn=True)
            text = res[0]["text"] if res else ""
        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            text = ""
        return json.dumps({"content": text}, ensure_ascii=False)