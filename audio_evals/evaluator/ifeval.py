from .base import Evaluator
from typing import Dict, List, Optional, Union
import dataclasses
import numpy as np
from .instruction_following_eval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


class IFEval(Evaluator):
    def __init__(self, strict: bool = True):
        self.strict = strict

    def _read_prompt_list(self, data: List[Dict]) -> List[InputExample]:
        """读取输入数据"""
        inputs = []
        for example in data:
            inputs.append(
                InputExample(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
            )
        return inputs

    def _read_prompt_to_response_dict(self, data: List[Dict]) -> Dict[str, str]:
        """创建 prompt 和 response 的映射字典"""
        return_dict = {}
        for example in data:
            if isinstance(example, list):
                example = example[0]
            tmp = example["response"]
            if tmp.startswith("<1>") or tmp.startswith("<2>") or tmp.startswith("<3>"):
                tmp = tmp[3:].strip()
            if tmp.endswith("<|user|>"):
                tmp = tmp[:-8].strip()
            return_dict[example["prompt"]] = tmp
        return return_dict

    def _test_instruction_following_strict(
        self, inp: InputExample, prompt_to_response: Dict[str, str]
    ) -> OutputExample:
        """严格测试指令遵循情况"""
        response = prompt_to_response[inp.prompt]
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)

        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=response,
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )

    def _test_instruction_following_loose(
        self, inp: InputExample, prompt_to_response: Dict[str, str]
    ) -> OutputExample:
        """宽松测试指令遵循情况"""
        response = prompt_to_response[inp.prompt]
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
        instruction_list = inp.instruction_id_list
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=inp.prompt)

            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list.append(is_following)

        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response=response,
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )

    def _print_report(self, outputs: List[OutputExample]) -> Dict[str, float]:
        """生成评测报告"""
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        for example in outputs:
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list

            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

        return {
            "prompt": prompt_correct / prompt_total,
            "instruction": instruction_correct / instruction_total,
        }

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label = str(label)

        # 准备输入数据
        data = [
            {
                "key": 0,
                "instruction_id_list": kwargs.get("instruction_id_list", []),
                "prompt": label,
                "kwargs": kwargs.get("kwargs", []),
                "response": pred,
            }
        ]

        # 读取输入数据
        inputs = self._read_prompt_list(data)
        prompt_to_response = self._read_prompt_to_response_dict(data)

        # 测试指令遵循情况
        if self.strict:
            outputs = [
                self._test_instruction_following_strict(inp, prompt_to_response)
                for inp in inputs
            ]
            results = self._print_report(outputs)
            return {
                "strict_prompt": results["prompt"],
                "strict_instruction": results["instruction"],
                "pred": pred,
                "ref": label,
            }
        else:
            outputs = [
                self._test_instruction_following_loose(inp, prompt_to_response)
                for inp in inputs
            ]
            results = self._print_report(outputs)
            return {
                "loose_prompt": results["prompt"],
                "loose_instruction": results["instruction"],
                "pred": pred,
                "ref": label,
            }
