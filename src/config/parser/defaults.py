import inspect
import logging

logger = logging.getLogger(__name__)

DEFAULTS = {}


def record_defaults(prefix, ignore=None):
    """
    Decorator for recording default values from keyword-arguments in classes and functions.
    """
    if prefix != "":
        prefix += "."
    ignore = ignore or []

    def inner(obj):
        sign = inspect.signature(obj)
        for name, param in sign.parameters.items():
            if (param.default is not param.empty) and (name not in ignore):
                key = f"{prefix}{name}"
                logger.debug(f"Recording default argument: {key}={param.default}")
                if key in DEFAULTS:
                    raise RuntimeError(f"Illegal duplicate config key. '{key}'.")
                DEFAULTS[key] = param.default
        return obj
    return inner
