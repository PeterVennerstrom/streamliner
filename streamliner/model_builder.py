import importlib
import json
import sys

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
    def __init__(self, path_to_cfg, device=0):
        self.path_to_cfg = path_to_cfg
        self.device = device
        self.models = {}

        with open(path_to_cfg, "r") as file:
            self.models = json.load(file)

    def build(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configurations.")

        model_cfg = self.models[model_name]
        import_to_register(model_cfg)
        model_class_name = model_cfg["model_class"]
        init_params = model_cfg.get("init_params", {})

        init_params["device"] = self.device

        model = build_object_by_name(model_class_name, **init_params)

        return model
