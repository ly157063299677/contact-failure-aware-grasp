"""Load an explicitly selected local simulator adapter; no automatic downloads."""
import importlib


def load_factory(spec):
    module, separator, symbol = spec.partition(":")
    if not separator or not module or not symbol:
        raise ValueError("Backend must be specified as module:factory")
    factory = getattr(importlib.import_module(module), symbol)
    if not callable(factory):
        raise TypeError("Backend factory is not callable")
    return factory
