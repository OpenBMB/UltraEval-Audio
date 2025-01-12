import json
import os
import typing


class Recorder:
    def __init__(self, f_name: str):
        self.name = f_name
        directory = os.path.dirname(f_name)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(f_name):
            print(f"File {f_name} already exists, overwriting it.")
            os.remove(f_name)

    def add(self, data: typing.Dict[str, typing.Any]):
        with open(self.name, "a+") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
