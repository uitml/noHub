import yaml
import logging
from enum import Enum
from argparse import ArgumentParser
from copy import deepcopy

from .defaults import DEFAULTS
from .nested_namespace import NestedNamespace
from .types import path_or_none
import helpers

logger = logging.getLogger(__name__)


class ArgStatus(Enum):
    """Single value Enum to indicate when an argument does not have a default."""
    NO_DEFAULT = 0


NO_DEFAULT = ArgStatus.NO_DEFAULT


class ArgumentParserWithDefaults(ArgumentParser):
    """Subclass ArgumentParser to allow for automatic defaults from dictionary."""
    def __init__(self, *args, defaults=None, parse_config_file=True, **kwargs):
        super(ArgumentParserWithDefaults, self).__init__(*args, **kwargs)

        if defaults is None:
            defaults = DEFAULTS

        self.defaults = deepcopy(defaults)

        if parse_config_file:
            self.add_argument("-c", "--config", dest="config", default=None, type=path_or_none)
            config_file = super(ArgumentParserWithDefaults, self).parse_known_args()[0].config
            self.file_args = self.load_file_args(config_file)
            self.update_defaults(self.file_args)
        else:
            self.file_args = {}

    def update_defaults(self, dct):
        self.defaults.update(dct)

    def add_argument_with_default(self, *args, default=ArgStatus.NO_DEFAULT, dest=None, **kwargs):
        assert len(args) == 1, "More than one argument alias is not supported by ArgumentParserWithDefaults."
        assert "required" not in kwargs, "ArgumentParserWithDefault does not support the 'required' kwarg."
        assert "action" not in kwargs, "ArgumentParserWithDefault does not support the 'action' kwarg."

        # Use the 'dest' kwarg as a key if it was specified
        # Otherwise, use the arg itself (without leading dashes)
        key = dest or args[0].lstrip("-")

        # Values are prioritized as follows:
        #   1. Value from 'self.defaults'
        #   2. Value from kwarg 'default'
        default_value = self.defaults.get(key, default)
        self.add_argument(*args, default=default_value, **kwargs)

    @staticmethod
    def load_file_args(config_file):
        if config_file is None:
            return {}

        logger.info(f"Loading args from file: '{config_file}'.")
        with open(config_file, "r") as f:
            file_args_nested = yaml.safe_load(f)

        file_args_flat = {}
        helpers.flatten_nested_dict(dct=file_args_nested, out=file_args_flat)
        return file_args_flat

    @staticmethod
    def check_required_args(args_dict):
        missing = []
        for key, value in args_dict.items():
            if value is NO_DEFAULT:
                missing.append(key)

        if len(missing) > 0:
            raise RuntimeError(f"Missing required argument(s): {', '.join(missing)}.")

    @staticmethod
    def check_file_args(args_dict, file_args_dict):
        extra = set(file_args_dict.keys()) - set(args_dict.keys())
        if len(extra) > 0:
            raise RuntimeError(f"Got unknown argument(s) in config file: {', '.join(extra)}.")

    def parse_args(self, *args, check=True, **kwargs):
        args_dict = vars(super(ArgumentParserWithDefaults, self).parse_args(*args, **kwargs))

        if check:
            self.check_required_args(args_dict)
            self.check_file_args(args_dict, self.file_args)

        args = NestedNamespace(dct=args_dict)
        return args


