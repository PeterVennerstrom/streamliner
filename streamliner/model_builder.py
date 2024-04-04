import importlib
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from .registry import register as REGISTER
from .registry import streamliner_registry


def build_object_by_name(object_name, *args, **kwargs):
    obj = streamliner_registry.get(object_name)
    if obj is not None:
        return obj(*args, **kwargs)
    else:
        raise ValueError(f"Object not found: {object_name}")


class LazyObject:
    def __init__(self, class_name, *args, **kwargs):
        self._object = None  # Placeholder for the actual object
        self._class_name = class_name
        self._init_args = args
        self._init_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self._object is None:
            # Use the existing function to build the object.
            self._object = build_object_by_name(
                self._class_name, *self._init_args, **self._init_kwargs
            )

        # Now that the object is built, forward the call.
        return self._object(*args, **kwargs)


def import_to_register(config):
    try:
        if "custom_import" in config:
            module_name = config["custom_import"]
            # Dynamically import the module, which will execute the decorator and register the class
            importlib.import_module(module_name)
    except ImportError as e:
        print(f"An error occurred while trying to import '{module_name}': {e}")
        sys.exit(1)


@REGISTER
class LocalBuilder:
    def __init__(self, config, device=0):
        if isinstance(config, str):
            with open(config, "r") as file:
                self.config = json.load(file)
        else:
            self.config = config

        self.device = device

    def build(self, model_name):
        if model_name not in self.config["models"]:
            raise ValueError(f"Model '{model_name}' not found in configurations.")

        model_cfg = self.config["models"][model_name]
        import_to_register(model_cfg)
        model_class_name = model_cfg["model_class"]
        init_params = model_cfg.get("init_params", {})
        init_params["device"] = self.device

        init_files = model_cfg.get("init_files", {})
        full_init_files = self.join_paths(model_name, init_files)
        init_params = {**full_init_files, **init_params}

        return build_object_by_name(model_class_name, **init_params)

    def join_paths(self, model_name, init_files):
        full_paths = {}
        for key, file_name in init_files.items():
            full_path = Path(self.config["model_dir"]) / model_name / file_name
            if not full_path.exists():
                raise FileNotFoundError(f"Required file not found: {full_path}")
            full_paths[key] = str(full_path)
        return full_paths


class RemoteBuilder(LocalBuilder, ABC):

    @abstractmethod
    def _acquire_files(self, model_name, files):
        # Logic to download files from remote source
        pass

    def build(self, model_name):
        model_cfg = self.config["models"][model_name]
        init_files = model_cfg.get("init_files", {})

        files_to_acquire = {
            key: val
            for key, val in init_files.items()
            if not (Path(self.config["model_dir"]) / model_name / val).exists()
        }

        self._acquire_files(model_name, files_to_acquire)

        return super().build(model_name)
