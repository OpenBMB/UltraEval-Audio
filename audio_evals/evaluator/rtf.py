import json
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class RTFExtractor(Evaluator):
    """
    RTF (Real-Time Factor) 提取器
    从 JSON 字符串中提取 RTF 值
    """

    def __init__(self, rtf_key: str = "RTF"):
        """
        Args:
            rtf_key: JSON 中 RTF 值的键名，默认为 "RTF"
        """
        self.rtf_key = rtf_key

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        """
        从 pred (JSON 字符串) 中提取 RTF 值

        Args:
            pred: JSON 字符串，格式如 {"audio": "path", "RTF": 0.5}
            label: 未使用，保留接口兼容性

        Returns:
            包含 RTF 值的字典
        """
        # 如果 pred 已经是 dict，直接使用
        if isinstance(pred, dict):
            data = pred
        else:
            # 解析 JSON 字符串
            data = json.loads(pred)

        return {
            "RTF": data[self.rtf_key],
            "pred": pred,
        }
