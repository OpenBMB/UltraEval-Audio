from dataclasses import dataclass, field
from typing import Dict, List, Union


class EarlyStop(Exception):
    pass


"""
request format
eg1: how are you
eg2: [{'role': 'user', 'content': 'how are you'}]
eg3: [{'role': 'user', 'contents': [{'type':'text', 'content': 'how are you'}, {'type':'image', 'content': '/mnt/a.git'}]]
"""
PromptStruct = Union[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]

ScoreUnit = Dict[str, Union[int, float]]


@dataclass
class EvalTaskCfg:
    dataset: str
    prompt: str
    model: str
    agg: str = "dump"
    evaluator: str = "dump"
    post_process: List[str] = field(default_factory=list)
