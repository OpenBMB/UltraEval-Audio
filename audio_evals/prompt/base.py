from functools import singledispatch
from typing import Any, Dict, List

from jinja2 import StrictUndefined, Template
from jinja2.exceptions import UndefinedError

from audio_evals.base import PromptStruct


@singledispatch
def _load(t: Any, **kwargs: Any) -> Any:
    return t


@_load.register
def _(t: str, **kwargs: Any) -> str:
    template = Template(t, undefined=StrictUndefined)
    try:
        return template.render(**kwargs)
    except UndefinedError as e:
        raise ValueError("{}: template is {}\ndoc is {}".format(e, t, kwargs))


@_load.register
def _(t: list, **kwargs: Any) -> List[Any]:
    return [_load(item, **kwargs) for item in t]


@_load.register
def _(t: dict, **kwargs: Any) -> Dict[Any, Any]:
    return {k: _load(v, **kwargs) for k, v in t.items()}


class Prompt:
    def __init__(self, template: PromptStruct):
        self.prompt = template

    def load(self, **kwargs):
        return _load(self.prompt, **kwargs)
