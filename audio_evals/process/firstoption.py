import re

from audio_evals.process.base import Process


class FirstOption(Process):
    def __init__(self, options):
        self.options = options

    def __call__(self, answer: str):
        options = self.options
        patterns = [
            f"答案是?\s*([{options}])",
            f"答案是?\s*：\s*([{options}])",
            f"答案是?\s*:\s*([{options}])",
            f"答案选项应?该?是\s*([{options}])",
            f"答案选项应?该?为\s*([{options}])",
            f"答案应该?是\s*([{options}])",
            f"答案应该?选\s*([{options}])",
            f"答案选项为?\s*：\s*([{options}])",
            f"答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?",
            f"答案选项是?\s*:\s*([{options}])",
            f"答案为\s*([{options}])",
            f"答案选\s*([{options}])",
            f"选择?\s*([{options}])",
            f"故选?\s*([{options}])" f"只有选?项?\s?([{options}])\s?是?对",
            f"只有选?项?\s?([{options}])\s?是?错",
            f"只有选?项?\s?([{options}])\s?不?正确",
            f"只有选?项?\s?([{options}])\s?错误",
            f"说法不?对选?项?的?是\s?([{options}])",
            f"说法不?正确选?项?的?是\s?([{options}])",
            f"说法错误选?项?的?是\s?([{options}])",
            f"([{options}])\s?是正确的",
            f"([{options}])\s?是正确答案",
            f"选项\s?([{options}])\s?正确",
            f"所以答\s?([{options}])",
            f"所以\s?([{options}][.。$]?$)",
            f"所有\s?([{options}][.。$]?$)",
            f"[\s，：:,]([{options}])[。，,\.]?$",
            f"[\s，,：:][故即]([{options}])[。\.]?$",
            f"[\s，,：:]因此([{options}])[。\.]?$",
            f"[是为。]\s?([{options}])[。\.]?$",
            f"因此\s?([{options}])[。\.]?$",
            f"显然\s?([{options}])[。\.]?$",
            f"答案是\s?(\S+)(?:。|$)",
            f"答案应该是\s?(\S+)(?:。|$)",
            f"答案为\s?(\S+)(?:。|$)",
            f"(?i)ANSWER\s*:\s*([{options}])",
            f"[Tt]he answer is:?\s+\(?([{options}])\)?",
            f"[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?",
            f"[Tt]he answer is option:?\s+\(?([{options}])\)?",
            f"[Tt]he correct answer is:?\s+\(?([{options}])\)?",
            f"[Tt]he correct answer is option:?\s+\(?([{options}])\)?",
            f"[Tt]he correct answer is:?.*?boxed{{([{options}])}}",
            f"[Tt]he correct option is:?.*?boxed{{([{options}])}}",
            f"[Tt]he correct answer option is:?.*?boxed{{([{options}])}}",
            f"[Tt]he answer to the question is:?\s+\(?([{options}])\)?",
            f"^选项\s?([{options}])",
            f"^([{options}])\s?选?项",
            f"(\s|^)[{options}][\s。，,：:\.$]",
            f"1.\s?(.*?)$",
            f"1.\s?([{options}])[.。$]?$",
        ]
        cushion_patterns = [
            f"([{options}]):",
            f"([{options}])",
        ]

        patterns.extend(cushion_patterns)
        for pattern in patterns:
            answer = answer.strip()
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                if match.group(1) is not None and match.group(1) != "":
                    outputs = match.group(1)
                else:
                    outputs = match.group(0)
                for i in options:
                    if i.upper() in outputs.upper():
                        return i
        return ""
