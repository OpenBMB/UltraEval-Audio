"""
Functions to handle registration of evals. To add a new eval to the registry,
add an entry in one of the YAML files in the `../registry` dir.
By convention, every eval name should start with {base_eval}.{split}.
"""

import difflib
import functools
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml

from audio_evals.agg.base import AggPolicy
from audio_evals.base import EvalTaskCfg
from audio_evals.dataset.dataset import Dataset
from audio_evals.evaluator.base import Evaluator
from audio_evals.models.model import Model
from audio_evals.process.base import Process
from audio_evals.prompt.base import Prompt
from audio_evals.utils import make_object

logger = logging.getLogger(__name__)

SPEC_RESERVED_KEYWORDS = ["key", "group", "cls"]


T = TypeVar("T")
RawRegistry = Dict[str, Any]
DEFAULT_PATHS = [Path(__file__).parents[1].resolve() / "registry"]


class Registry:
    def __init__(self, registry_paths=None):
        if registry_paths is None:
            registry_paths = DEFAULT_PATHS
        self._registry_paths = [
            Path(p) if isinstance(p, str) else p for p in registry_paths
        ]

    def add_registry_paths(self, paths: Sequence[Union[str, Path]]) -> None:
        self._registry_paths.extend(
            [Path(p) if isinstance(p, str) else p for p in paths]
        )

    def _dereference(
        self, name: str, d: RawRegistry, object: str, **kwargs: dict
    ) -> Optional[T]:
        if name not in d:
            logger.warning(
                (
                    f"{object} '{name}' not found. "
                    f"Closest matches: {difflib.get_close_matches(name, d.keys(), n=5)}"
                )
            )
            return None

        logger.debug(f"Looking for {name}")

        spec = d[name]
        if kwargs:  # specified parameters take precedence over default one
            spec["args"].update(kwargs)

        try:
            return make_object(spec["cls"], **spec["args"])
        except TypeError as e:
            raise TypeError(f"Error while processing {object} '{name}': {e}")

    def get_model(self, name: str, **kwargs) -> Optional[Model]:
        return self._dereference(name, self._model, "model", **kwargs)

    def get_evaluator(self, name: str, **kwargs) -> Optional[Evaluator]:
        return self._dereference(name, self._evaluator, "evaluator", **kwargs)

    def get_agg(self, name: str, **kwargs) -> Optional[AggPolicy]:
        return self._dereference(name, self._agg, "agg", **kwargs)

    def get_eval_task(self, name: str, **kwargs) -> Optional[EvalTaskCfg]:
        return self._dereference(name, self._eval_task, "eval task", **kwargs)

    def get_dataset(self, name: str, **kwargs) -> Optional[Dataset]:
        return self._dereference(name, self._dataset, "dataset", **kwargs)

    def get_prompt(self, name: str, **kwargs) -> Optional[Prompt]:
        return self._dereference(name, self._prompt, "prompt", **kwargs)

    def get_process(self, name: str, **kwargs) -> Optional[Process]:
        return self._dereference(name, self._process, "process", **kwargs)

    def _load_file(self, path: Path) -> Generator[Tuple[str, Path, dict], None, None]:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f.read())

        if d is None or not isinstance(d, dict):
            # no entries in the file
            return

        for name, spec in d.items():
            yield name, path, spec

    def _load_directory(
        self, path: Path
    ) -> Generator[Tuple[str, Path, dict], None, None]:
        files = Path(path).glob("*.yaml")
        for file in files:
            yield from self._load_file(file)

    def _load_resources(
        self, registry_path: Path, resource_type: str
    ) -> Generator[Tuple[str, Path, dict], None, None]:
        path = registry_path / resource_type
        logging.info(f"Loading registry from {path}")

        if os.path.exists(path):
            if os.path.isdir(path):
                yield from self._load_directory(path)
            else:
                yield from self._load_file(path)

    @staticmethod
    def _validate_reserved_keywords(spec: dict, name: str, path: Path) -> None:
        for reserved_keyword in SPEC_RESERVED_KEYWORDS:
            if reserved_keyword in spec:
                raise ValueError(
                    f"{reserved_keyword} is a reserved keyword, but was used in {name} from {path}"
                )

    def _load_registry(
        self, registry_paths: Sequence[Path], resource_type: str
    ) -> RawRegistry:
        """Load registry from a list of regstry paths and a specific resource type

        Each path includes yaml files which are a dictionary of name -> spec.
        """

        registry: RawRegistry = {}

        for registry_path in registry_paths:
            for name, path, spec in self._load_resources(registry_path, resource_type):
                assert name not in registry, f"duplicate entry: {name} from {path}"
                self._validate_reserved_keywords(spec, name, path)

                spec["key"] = name
                spec["group"] = str(os.path.basename(path).split(".")[0])
                spec["registry_path"] = registry_path

                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]

                registry[name] = spec

        return registry

    @functools.cached_property
    def _model(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "model")

    @functools.cached_property
    def _dataset(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "dataset")

    @functools.cached_property
    def _evaluator(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "evaluator")

    @functools.cached_property
    def _prompt(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "prompt")

    @functools.cached_property
    def _eval_task(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "eval_task")

    @functools.cached_property
    def _agg(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "agg")

    @functools.cached_property
    def _process(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "process")


registry = Registry()
