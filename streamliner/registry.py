streamliner_registry = {}


def register(cls):
    streamliner_registry[cls.__name__] = cls
    return cls
