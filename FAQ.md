

## 1.  ./nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12

ref:
https://github.com/pytorch/pytorch/issues/111469

two solutions:
- you can update your nvidia to match torch
- use your python env nvidia path not system, like: `export LD_LIBRARY_PATH=$HOME/path/to/my/venv3115/lib64/
python3.11/site-packages/nvidia/nvjitlink/lib` or`export LD_LIBRARY_PATH=env/lib/python3.10/site-packages/nvidia/nvjitlink/lib`

## 2. ConnectionError: Couldn't reach 'TwinkStart/xx' on the Hub (LocalEntryNotFoundError)

make sure you can access the huggingface hub, you maybe need use proxy:

> export HF_ENDPOINT=https://hf-mirror.com
