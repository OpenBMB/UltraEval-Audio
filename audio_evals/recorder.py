import json
import logging
import os
import time
import typing

logger = logging.getLogger(__name__)

# Shared filesystems (JuiceFS) intermittently return EIO on open/append for a
# few seconds. One such blip used to kill whole evaluation jobs, so retry
# writes with backoff before giving up.
_MAX_ATTEMPTS = 8
_BACKOFF_SECONDS = (1, 2, 5, 10, 20, 30, 60)


class Recorder:
    def __init__(self, f_name: str):
        self.name = f_name
        directory = os.path.dirname(f_name)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(f_name):
            print(f"File {f_name} already exists, overwriting it.")
            os.remove(f_name)

    def add(self, data: typing.Dict[str, typing.Any]):
        payload = json.dumps(data, ensure_ascii=False) + "\n"
        last_error = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with open(self.name, "a+") as f:
                    f.write(payload)
                return
            except OSError as error:
                last_error = error
                wait = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                logger.warning(
                    "recorder write to %s failed (attempt %d/%d): %r; retrying in %ds",
                    self.name,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    error,
                    wait,
                )
                time.sleep(wait)
        raise last_error
