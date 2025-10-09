

## 1.  ./nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12

ref:
https://github.com/pytorch/pytorch/issues/111469

two solutions:
- you can update your nvidia to match torch
- use your python env nvidia path not system, like: `export LD_LIBRARY_PATH=$HOME/path/to/my/venv3115/lib64/
python3.11/site-packages/nvidia/nvjitlink/lib` or`export LD_LIBRARY_PATH=env/lib/python3.10/site-packages/nvidia/nvjitlink/lib`

## 2. ConnectionError: Couldn't reach 'TwinkStart/xx' on the Hub (LocalEntryNotFoundError) | An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the specified revision on the local disk.
Please check your internet connection and try again.


make sure you can access the huggingface hub, you maybe need use proxy:

> export HF_ENDPOINT=https://hf-mirror.com


## 3. gigaspeech: 'None Type' object is not callable

Gigaspeech is not a directly accessible dataset; you need to request permission from the authors.
https://huggingface.co/datasets/speechcolab/gigaspeech

When you attempt to download it, you will encounter a login page. If you do not have permission, there will be an HF link prompting you to apply for access.

If the above does not appear, enter the following code in the Python interactive shell:

```python
from datasets import load_dataset
gs_test = load_dataset("speechcolab/gigaspeech", "test")
```

If this code runs successfully, you can proceed with the evaluation.

## 4. The official evaluation prompts for MiniCPM-O 2.6

1. ASR zh: --prompt mini-cpm-omni-asr-zh
2. ASR en: --prompt mini-cpm-omni-asr-en
3. AST 2zh: --prompt mini-cpm-omni-asr-zh
4. AST 2en: --prompt mini-cpm-omni-ast-en
5. emotion analysis: --prompt mini-cpm-omni-emotion_analysis
