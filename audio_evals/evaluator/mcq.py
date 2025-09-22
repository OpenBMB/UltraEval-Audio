from audio_evals.evaluator.base import Evaluator
import random
from typing import Dict


class MCQ(Evaluator):
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def _extract_answer(self, response: str) -> str:
        response = response.lower() if self.ignore_case else response
        if (
            response.startswith("<1>")
            or response.startswith("<2>")
            or response.startswith("<3>")
        ):
            response = response[3:].strip()

        # 定义答案模板
        templates = [
            "答案是[CHOICE]",
            "答案是:[CHOICE]",
            "答案是：[CHOICE]",
            "答案是 [CHOICE]",
            "答案是选项[CHOICE]",
            "答案应该是[CHOICE]",
            "答案应该是 [CHOICE]",
            "答案就是选项[CHOICE]",
            "答案是'[CHOICE]",
            "是[CHOICE]：",
            "答案选[CHOICE]",
            "[CHOICE]是正确",
            "选项[CHOICE]是最合适的",
            "answer is: **[CHOICE]",
            "answer is **[CHOICE]",
            "the answer to the question is: **[CHOICE]",
            "the answer to the multiple-choice question is **[CHOICE]",
            "the answer is '[CHOICE]'",
            "[CHOICE] is the best answer",
            "the answer is [CHOICE]",
            "the correct answer is [CHOICE]",
            "would select [CHOICE]",
            "would choose [CHOICE]",
            "would select option [CHOICE]",
            "would choose option [CHOICE]",
            'is "[CHOICE]"',
            'is "[CHOICE].',
            "is: **[CHOICE])",
            "is **[CHOICE],",
            "is **[CHOICE]:",
            "is **[CHOICE])",
            "is: **[CHOICE].",
            "is: **[CHOICE]:",
            "is **[CHOICE].",
            "be **[CHOICE],",
            "is: **[CHOICE]**",
            "is therefore option **[CHOICE]:",
            "is: \n\n**[CHOICE])",
            "as **[CHOICE]:",
            "be **[CHOICE])",
            "be **[CHOICE]:",
            "is: \n\n**[CHOICE]**",
            "suggests **[CHOICE])",
            "be option **[CHOICE]:",
            "with **[CHOICE])",
            'is typically "[CHOICE])',
            "be to **[CHOICE])",
            "is: \n\n[CHOICE])",
            "is likely to be: **[CHOICE].",
            "is **[CHOICE] (",
            "is option **[CHOICE]**",
            "is likely **[CHOICE]**",
            "is:\n**[CHOICE].",
            "is:\n\n**[CHOICE].",
            "would be [CHOICE]",
            "would be option [CHOICE]",
            "would be ([CHOICE])",
            "would be option ([CHOICE])",
            "is [CHOICE],",
            "is typically [CHOICE],",
            "is typically [CHOICE].",
            "i'd say [CHOICE].",
            "option [CHOICE].",
            "option [CHOICE]:",
            "option [CHOICE],",
            "the answer is:\n**[CHOICE]",
            "is [CHOICE]:",
            "is [CHOICE].",
            "is [CHOICE],",
            "is: [CHOICE].",
            "is ([CHOICE])",
            "is:\n**[CHOICE])",
            "is likely **[CHOICE]:",
            "is the **[CHOICE])",
            ":\n[CHOICE].",
            ":\n[CHOICE])",
            ":\n[CHOICE],",
            ": \n[CHOICE].",
            ":  \n[CHOICE].",
            ":\n\n[CHOICE].",
            ":\n\n[CHOICE])",
            "is most likely **[CHOICE]:",
            ":\n\n[CHOICE],",
            ": \n\n[CHOICE].",
            "is option [CHOICE],",
            "([CHOICE]) would be",
            "is ([CHOICE]).",
            "is [CHOICE])",
            "is: [CHOICE])",
            "is:\n\n[CHOICE]:",
            "is: **[CHOICE],",
            "(option [CHOICE])",
            "answer is ([CHOICE])",
            'select option "[CHOICE]"',
            "is: [CHOICE]",
            "is typically **[CHOICE],",
            "is **[CHOICE]**",
            "is likely '[CHOICE]'",
            "is option '[CHOICE]'",
            "is:\n**[CHOICE]:",
            "is \\( \\boxed{[CHOICE] ",
            "would be '[CHOICE]'",
            "is the **[CHOICE]** ",
            "question is [CHOICE] (",
            "is:\n\n**[CHOICE])",
            "closest to option **[CHOICE]**",
            "is most likely **[CHOICE])",
            "the answer to the question is '[CHOICE]'",
            "question is **[CHOICE]**",
            "known as '[CHOICE]'",
            "is '[CHOICE])",
            "is typically **[CHOICE]:",
            "is \\( \\boxed{\\text{[CHOICE]}} \\)",
            "is \\( \\text{[CHOICE]) }",
            "is \\( \\text{[CHOICE]} \\)",
            "is \\( \\text{[CHOICE]:",
            "is \\( \\text{[CHOICE])",
            "is \\(\\text{[CHOICE].",
            "is:\n\n**[CHOICE]",
            "is \\( \\text{[CHOICE].}",
            "is \\( \\text{[CHOICE].",
            "is \\( \\boxed{[CHOICE]}",
            "is:\n\\[ \\boxed{\\text{[CHOICE]}}",
            "is:\n\\[ \\text{[CHOICE])",
            "is:\n\n\\[ \\text{[CHOICE])",
            "is \\( \\textbf{[CHOICE])",
            "is \\( \\text{[CHOICE]}",
            "is: \\( \\text{[CHOICE].",
            "corresponds to:\n- **[CHOICE]:",
            "would be: **[CHOICE]**.",
            "is \\( [CHOICE] \\)",
            "is:\n**[CHOICE] ",
            "corresponds to option **[CHOICE]**",
            "be **[CHOICE]**",
            "be: \n\n[CHOICE])",
            "is:\n\\[ \\boxed{[CHOICE]}",
            "is:  \n**[CHOICE]:",
            "is: \\( \\text{[CHOICE])",
            "is likely: **[CHOICE],",
            "is } \\mathbf{[CHOICE].",
            "is \\( \\boxed{[CHOICE])",
            "is \\( \\textbf{[CHOICE]}",
            "is \\([CHOICE]\\)",
            "is:\n  \n**[CHOICE]:",
            "is option **[CHOICE] ",
            "is:\n\\( \\textbf{[CHOICE].",
            "is \\( \\mathbf{[CHOICE]}",
            "was option **[CHOICE]**",
            'is likely "[CHOICE])',
            "option **[CHOICE]:",
            'is "[CHOICE])',
            "is most likely **[CHOICE],",
            "is often **[CHOICE]:",
            "is:  \n[CHOICE])",
            " [CHOICE].",
            " [CHOICE],",
            " [CHOICE]:",
            " [CHOICE])",
            "**[CHOICE].",
            "**[CHOICE])",
            '"[CHOICE].',
            '"[CHOICE],',
            '"[CHOICE]:',
            "([CHOICE])",
            '"[CHOICE]"',
        ]

        for template in templates:
            for choice in ["a", "b", "c", "d"]:
                if template.replace("[CHOICE]", choice) in response:
                    return choice.upper()

        for choice in ["a", "b", "c", "d"]:
            if response == choice:
                return choice.upper()
            for punc in [".", ",", ":", ")"]:
                if response.startswith(choice + punc):
                    return choice.upper()

        if "would be a." in response:
            return "A"
        elif 'would be "a.' in response:
            return "A"
        elif (
            "the best option from the given choices would be a scorpion (a)" in response
        ):
            return "A"

        return None

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label = str(label)

        if self.ignore_case:
            pred = pred.lower()
            label = label.lower()

        extracted_answer = self._extract_answer(pred)
        if extracted_answer is None:
            return {
                "match": 0,
                "pred": extracted_answer,
                "ref": label,
                "extract_fail": 1,
            }

        return {
            "match": 1 if extracted_answer.lower() == label.lower() else 0,
            "pred": extracted_answer,
            "ref": label,
            "extract_fail": 0,
        }


if __name__ == "__main__":
    s = "The answer to the multiple choice question is: D. Lake Erie. A mola mola, also known as an ocean sunfish, is a large fish that is native to the Atlantic, Pacific, and Indian Oceans. It is not found in the Great Lakes, the Mississippi River, or the Bay of Bengal. Lake Erie is a large freshwater lake in North America, and it is the only one of the Great Lakes where mola mola might live."
    print(MCQ()._eval(s, "D"))
