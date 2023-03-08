import torch as th
import numpy as np
import pandas as pd
from torch.profiler import record_function


def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.
    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def dict_means(dicts):
    """
    Compute the mean value of keys in a list of dicts
    :param dicts: Input dicts
    :type dicts: List[dict]
    :return: Mean values
    :rtype: dict
    """
    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def transpose_list_dict(dicts):
    """
    Convert a list of dicts to a dict of lists. Assumes keys are the same in all dicts.
    """
    return {key: [dct[key] for dct in dicts] for key in dicts[0].keys()}


def dict_cat(dicts):
    """
    Converts a list of dicts with np.array values to a dict, where values are the concatenated arrays for each key in
    dicts[0]. Assumes all dicts in `dicts` have the same set of keys.
    """
    return {key: np.concatenate([dct[key] for dct in dicts], axis=0) for key in dicts[0].keys()}


def dict_cat_tensor(dicts):
    """
    Converts a list of dicts with th.tensor values to a dict, where values are the concatenated tensors for each key in
    dicts[0]. Assumes all dicts in `dicts` have the same set of keys.
    """
    return {key: th.cat([dct[key] for dct in dicts], dim=0) for key in dicts[0].keys()}


def flatten_nested_dict(dct, out, base_key="", sep="."):
    if base_key != "":
        base_key += sep

    for key, value in dct.items():
        full_key = base_key + key
        if isinstance(value, dict):
            flatten_nested_dict(value, out, base_key=full_key, sep=sep)
        else:
            out[full_key] = value


def record_function_decorator(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            with record_function(name):
                out = func(*args, **kwargs)
            return out
        return wrapper
    return inner


def versions():
    import re
    import torch as th
    from subprocess import run, CalledProcessError

    # CUDA version
    try:
        output = run(["nvidia-smi"], check=True, capture_output=True)
        match = re.findall(r"CUDA Version: (\d+\.\d+)", output.stdout.decode("utf8"))
        cuda_version = match[0] if match else None
    except (CalledProcessError, FileNotFoundError):
        cuda_version = None

    out = {
        "cuda_version": cuda_version,
        "torch_version": th.__version__
    }
    return out
