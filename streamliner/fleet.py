import json
from multiprocessing import Array, Lock, Manager, Pipe, Process, Value

from .model_builder import (
    LocalBuilder,
    RemoteBuilder,
    build_object_by_name,
    streamliner_registry,
)


class SingleDeviceFleet:
    def __init__(self, model_builder):
        if not isinstance(model_builder, LocalBuilder):
            raise ValueError(
                "model_builder must be an instance of LocalBuilder or its subclasses."
            )
        self.model_builder = model_builder
        self.loaded_models = {}

    def __getitem__(self, model_name):
        return self.load_model(model_name)

    def __getattr__(self, model_name):
        if model_name not in self.loaded_models:
            self.loaded_models[model_name] = self.model_builder.build(model_name)
        model = self.loaded_models[model_name]

        def model_accessor(*args, **kwargs):
            if not args and not kwargs:
                return model
            else:
                return model(*args, **kwargs)

        return model_accessor

    def load_model(self, model_name):
        if model_name not in self.loaded_models:
            self.loaded_models[model_name] = self.model_builder.build(model_name)
        return self.loaded_models[model_name]


class ModelProxy:
    def __init__(self, model_config, fleet_callable):
        self.model_config = model_config
        self.fleet_callable = fleet_callable

    def __getattr__(self, model_name):
        if model_name in self.model_config:
            return ModelMethodProxy(
                self.fleet_callable, model_name, None, self.model_config[model_name]
            )
        else:
            raise AttributeError(
                f"Model '{model_name}' not found in model configuration."
            )

    def __getitem__(self, model_name):
        return self.__getattr__(model_name)

    def __getstate__(self):
        return self.model_config, self.fleet_callable

    def __setstate__(self, state):
        self.model_config, self.fleet_callable = state


class ModelMethodProxy:
    def __init__(self, fleet_callable, model_name, method_name, model_details):
        self.fleet_callable = fleet_callable
        self.model_name = model_name
        self.method_name = method_name
        self.model_details = model_details

    def __call__(self, *args, **kwargs):
        call_dict = {
            "model_name": self.model_name,
            "method_name": self.method_name,
            "args": args,
            "kwargs": kwargs,
        }

        results = self.fleet_callable(call_dict)
        return results

    def __getattr__(self, method_name):
        if (
            "proxy_methods" in self.model_details
            and method_name in self.model_details["proxy_methods"]
        ):
            return ModelMethodProxy(
                self.fleet_callable, self.model_name, method_name, self.model_details
            )
        else:
            raise AttributeError(
                f"Method '{method_name}' not configured for proxy on model '{self.model_name}'."
            )


class DeviceLoadBalancer:
    def __init__(
        self,
        main_pipes,
        task_completion_events,
        device_access_locks,
        device_indices,
        model_build_tracker,
    ):
        self.main_pipes = main_pipes
        self.task_completion_events = task_completion_events
        self.device_access_locks = device_access_locks
        self.device_indices = device_indices
        self.model_build_tracker = model_build_tracker

        self.round_robin_index = Value("i", 0)
        self.round_robin_lock = Lock()
        self.device_availability = Array("b", [True] * len(device_indices))
        self.availability_lock = Lock()

    def _select_device(self):
        with self.availability_lock, self.round_robin_lock:
            available_devices = [
                i for i, available in enumerate(self.device_availability) if available
            ]
            if available_devices:
                selected_index = available_devices[
                    self.round_robin_index.value % len(available_devices)
                ]
            else:
                selected_index = self.round_robin_index.value % len(self.device_indices)

            self.round_robin_index.value += 1
            self.device_availability[selected_index] = False

        return self.device_indices[selected_index]

    def _mark_device_available(self, device):
        with self.availability_lock:
            self.device_availability[self.device_indices.index(device)] = True

    def _ensure_model_ready(self, model_name):
        if model_name not in self.model_build_tracker:
            self.model_build_tracker[model_name] = False
        else:
            while not self.model_build_tracker[model_name]:
                pass

    def __call__(self, call_dict):
        if self.model_build_tracker:
            model_name = call_dict["model_name"]
            self._ensure_model_ready(model_name)

        device = self._select_device()
        self.device_access_locks[device].acquire()
        try:
            self.main_pipes[device].send(call_dict)
            self.task_completion_events[device].wait()
            results = self.main_pipes[device].recv()
            if self.model_build_tracker:
                self.model_build_tracker[model_name] = True
        finally:
            self.task_completion_events[device].clear()
            self.device_access_locks[device].release()
            self._mark_device_available(device)

        return results


class MultiDeviceFleet:
    def __init__(self, device_indices, model_builder_config):
        self.device_indices = device_indices
        self.model_builder_config = model_builder_config
        self.per_device_workers = []
        self.main_pipes = []
        self.task_completion_events = []
        self.process_manager = Manager()
        self.device_access_locks = [
            self.process_manager.Lock() for _ in self.device_indices
        ]
        self.initialize_per_device_workers()
        remote_builder = isinstance(
            streamliner_registry.get(model_builder_config["class"]), RemoteBuilder
        )
        self.model_build_tracker = (
            self.process_manager.dict() if remote_builder else None
        )

    def initialize_per_device_workers(self):
        for device_id in self.device_indices:
            main_conn, worker_conn = Pipe()
            event = self.process_manager.Event()
            worker = Process(
                target=self.per_device_worker_routine,
                args=(device_id, worker_conn, event, self.model_builder_config),
            )
            worker.start()
            self.per_device_workers.append(worker)
            self.main_pipes.append(main_conn)
            self.task_completion_events.append(event)

    @staticmethod
    def per_device_worker_routine(device_id, worker_conn, event, model_builder_config):
        model_builder = build_object_by_name(
            model_builder_config["class"],
            **model_builder_config["init_params"],
            device=device_id,
        )
        fleet = SingleDeviceFleet(model_builder)

        while True:
            call_dict = worker_conn.recv()
            if call_dict is None:
                break
            results = MultiDeviceFleet._reconstruct_and_call(fleet, call_dict)
            worker_conn.send(results)
            event.set()

    @staticmethod
    def _reconstruct_and_call(fleet, call_dict):
        model_name = call_dict["model_name"]
        method_name = call_dict["method_name"]
        args = call_dict["args"]
        kwargs = call_dict["kwargs"]

        if method_name:
            result = getattr(fleet[model_name], method_name)(*args, **kwargs)
        else:
            result = fleet[model_name](*args, **kwargs)

        return result

    def stop(self):
        for pipe in self.main_pipes:
            pipe.send(None)
        for worker in self.per_device_workers:
            worker.join()

    @property
    def fleet_callable(self):
        device_load_balancer = DeviceLoadBalancer(
            self.main_pipes,
            self.task_completion_events,
            self.device_access_locks,
            self.device_indices,
            self.model_build_tracker,
        )
        return device_load_balancer

    @property
    def model_proxy(self):
        model_config_path = self.model_builder_config["init_params"]["path_to_cfg"]
        with open(model_config_path, "r") as file:
            model_config = json.load(file)

        proxy = ModelProxy(model_config, self.fleet_callable)
        return proxy
