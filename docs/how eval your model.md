
In the QuickStart, it's easy to launch an eval task, but your model not be integrated AudioEvals, how can eval it?

Here are steps:

# model api
> your model is deployed as a service

## 1. add model inference code

add a py-file in `audio_evals/models/` path, content like:

```PYTHON
from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct

class MyAudioModel(APIModel):
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # TODO
        # my request code
```

reference: `audio_evals/models/google.py`


## 2. register the model

in the `registry/model/` path, new a yaml file, with content:

```yaml
$name:  # the name after command: --model $name
  class: audio_evals.models.$new_file.$MyAudioModel
  args:
    ...  # your specific args. If not need args, just fill args: {}


```


# offline model


## 1. add model inference code (optional)
> if your model is supported with huggingface AutoModelForCausalLM, you can skip this step.

add a py-file in `audio_evals/models/` path, content like:
```PYTHON
from audio_evals.models.offline_model import  OfflineModel
from audio_evals.base import PromptStruct
from typing import Dict

class MyAudioModel(OfflineModel):
    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        super().__init__(is_chat, sample_params)
        # TODO
        # init code

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        # TODO
        # inference code
```

## 2. register the model

the `registry/model/` path, new a yaml file, with content:

```yaml
$name:  # the name after command: --model $name
  class: audio_evals.models.offline_model.OfflineModel
  args:
    path:   # the name of model from huggingface model or the download model path download from huggingface


```


after registry model, you can eval your model with `--model $name`, enjoy 😘
