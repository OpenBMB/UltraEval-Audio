from .base import Evaluator
import numpy as np
from typing import Dict, List, Optional, Union


class ChoicesEval(Evaluator):
    def __init__(self, choices_columns: Union[List[str], str], ignore_parse_error: bool = False, constant = False):
        self.choices_columns = choices_columns
        self.ignore_parse_error = ignore_parse_error
        self.constant = constant

    def _eval(self, pred: str, label: str, **kwargs) -> Dict[str, any]:
        pred = str(pred).strip()
        label = str(label).strip()
        
        # Valid choices
        if isinstance(self.choices_columns, str):
            choices = kwargs[self.choices_columns]
            choice_letters = [chr(65+i) for i in range(len(choices))]
            choices = {choice.lower().strip(): choice_letters[i] for i, choice in enumerate(choices)}
        else:
            if self.constant:
                choice_letters = [chr(65+i) for i in range(len(self.choices_columns))]
                choices = {item.lower().strip(): choice_letters[i] for i, item in enumerate(self.choices_columns)}

            else:
                choice_letters = [chr(65+i) for i in range(len(self.choices_columns))]
                choices = {str(kwargs.get(col, '')).lower().strip().replace('-', ' '): choice_letters[i] for i, col in enumerate(self.choices_columns)}

        
        # Extract model prediction from response
        model_predict = None
        is_format_error = False
        
        if pred and pred != 'None':
            # Check if first character is a valid choice
            if pred[0] in choice_letters:
                model_predict = pred[0]
            # This situation may occur when the answer given by model is "The answer is A."
            elif len(pred) > 1:
                if pred[-2] in choice_letters:
                    model_predict = pred[-2]
                else:
                    print(f'Wrong format response: {pred}')
                    is_format_error = True
            else:
                print(f'Wrong format response: {pred}')
                is_format_error = True
        else:
            print(f'Wrong format response: {pred}')
            is_format_error = True
        
        # Determine the result: 1 (correct), 0 (incorrect), None (format error)
        result: Optional[int] = None
        
        if is_format_error:
            if self.ignore_parse_error:
                result = 0
            else:
                result = None
        elif model_predict:
            # Get choices from kwargs
            result = 1 if model_predict.lower() == choices[label.lower().strip().replace('-', ' ')].lower() else 0
        
        return {
            "match": result,  # 1, 0, or None
            "is_format_error": 1 if is_format_error else 0,  # 1 for format error, 0 otherwise
            "pred": pred,
            "model_predict": model_predict,
            "ref": label,
        }
