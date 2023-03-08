"""
Custom "type" parsers for arguments
"""
from functools import partial
from pathlib import Path


def _type_or_none(arg, typ=str):
    if (arg is None) or (arg.lower() == "none"):
        return None
    return typ(arg)


str_or_none = partial(_type_or_none, typ=str)
int_or_none = partial(_type_or_none, typ=int)
float_or_none = partial(_type_or_none, typ=float)
path_or_none = partial(_type_or_none, typ=Path)


def str_upper(arg):
    if arg is None:
        return None
    return arg.upper()


def str_lower(arg):
    if arg is None:
        return None
    return arg.lower()


def str_to_bool(arg):
    arg = str_lower(arg)
    if arg in {"true", "t", "yes", "y"}:
        return True
    return False