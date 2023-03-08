"""
Hubness metrics. Adapted from https://scikit-hubness.readthedocs.io/en/latest/
"""
import torch as th


def skewness(k_occ):
    centered = k_occ - k_occ.mean(dim=1, keepdims=True)
    m3 = (centered**3).mean(dim=1)
    m2 = (centered**2).mean(dim=1)
    skew = m3 / (m2 ** (3/2))
    return skew


def hub_occurrence(k_occ, k, hub_size=2):
    is_hub = (k_occ >= (hub_size * k)).float()
    hub_occ = (k_occ * is_hub).mean(dim=1) / k
    return hub_occ


def antihub_occurrence(k_occ):
    is_antihub = (k_occ == 0).float()
    return is_antihub.mean(dim=1)


def robinhood_index(k_occ):
    numerator = .5 * th.sum(th.abs(k_occ - k_occ.mean(dim=1, keepdims=True)), dim=1)
    denominator = th.sum(k_occ, dim=1)
    return numerator / denominator


def score(k_occ, k):
    k_occ = k_occ.float()
    return {
        "skewness": skewness(k_occ),
        "hub_occurrence": hub_occurrence(k_occ, k),
        "antihub_occurrence": antihub_occurrence(k_occ),
        "robinhood_index": robinhood_index(k_occ),
    }
