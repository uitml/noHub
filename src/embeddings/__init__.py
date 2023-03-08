from config.parser.types import *

from .embeddings import L2Norm, CL2Norm, PCANorm, ZScoreNorm
from .no_hub import train_no_hub, NoHub, NoHubS
from .re_representation import ReRepresentation
from .tcpr import TCPR
from .ease import EASE


# ======================================================================================================================
# Argument functions
# ======================================================================================================================

def _add_base_no_hub_args(parser, prefix):
    parser.add_argument_with_default(f"--{prefix}.init", type=str_lower)
    parser.add_argument_with_default(f"--{prefix}.out_dims", type=int_or_none)
    parser.add_argument_with_default(f"--{prefix}.initial_dims", type=int_or_none)
    parser.add_argument_with_default(f"--{prefix}.kappa", type=float)
    parser.add_argument_with_default(f"--{prefix}.perplexity", type=float)
    parser.add_argument_with_default(f"--{prefix}.re_norm", type=str_to_bool)
    parser.add_argument_with_default(f"--{prefix}.eps", type=float)
    parser.add_argument_with_default(f"--{prefix}.p_sim", type=str_lower)
    parser.add_argument_with_default(f"--{prefix}.p_rel_tol", type=float_or_none)
    parser.add_argument_with_default(f"--{prefix}.p_abs_tol", type=float_or_none)
    parser.add_argument_with_default(f"--{prefix}.p_betas", type=float_or_none, nargs=2)
    parser.add_argument_with_default(f"--{prefix}.pca_mode", type=str, choices=["episode", "base"])
    parser.add_argument_with_default(f"--{prefix}.n_iter", type=int)
    parser.add_argument_with_default(f"--{prefix}.learning_rate", type=float)


def add_no_hub_args(parser):
    _add_base_no_hub_args(parser, prefix="nohub")
    parser.add_argument_with_default("--nohub.loss_weights", type=float, nargs="+")


def add_no_hub_s_args(parser):
    _add_base_no_hub_args(parser, "nohubs")
    parser.add_argument_with_default("--nohubs.exaggeration_mode", type=str, choices=["linear", "exp", "hyperbolic"])
    parser.add_argument_with_default("--nohubs.different_label_exaggeration", type=float)
    parser.add_argument_with_default("--nohubs.loss_weights", type=float, nargs="+")
    parser.add_argument_with_default("--nohubs.modify_align_loss", type=str_to_bool)
    parser.add_argument_with_default("--nohubs.modify_uniform_loss", type=str_to_bool)


def add_pca_args(parser):
    parser.add_argument_with_default("--pca.out_dims", type=int_or_none)
    parser.add_argument_with_default("--pca.mode", type=str_lower, choices=["base", "episode"])


def add_cl2_args(parser):
    parser.add_argument_with_default("--cl2.dim", type=int)
    parser.add_argument_with_default("--cl2.center_mode", choices=["base", "episode"], type=str_lower)
    parser.add_argument_with_default("--cl2.out_dims", type=int_or_none)
    parser.add_argument_with_default("--cl2.pca_mode", type=str_lower)


def add_l2_args(parser):
    parser.add_argument_with_default("--l2.out_dims", type=int_or_none)
    parser.add_argument_with_default("--l2.pca_mode", type=str_lower, choices=["base", "episode"])


def add_ease_args(parser):
    parser.add_argument_with_default("--ease.k", type=int)
    parser.add_argument_with_default("--ease.lam", type=float)
    parser.add_argument_with_default("--ease.low_rank", type=int)
    parser.add_argument_with_default("--ease.l2_normalize_outputs", type=str_to_bool)


def add_zn_args(parser):
    pass


def add_rr_args(parser):
    parser.add_argument_with_default("--rr.alpha_1", type=float)
    parser.add_argument_with_default("--rr.alpha_2", type=float)
    parser.add_argument_with_default("--rr.tau", type=float)


def add_tcpr_args(parser):
    parser.add_argument_with_default("--tcpr.k", type=int)


def add_none_args(parser):
    pass


EMBEDDING_ARG_ADDERS = {
    "nohub": add_no_hub_args,
    "nohubs": add_no_hub_s_args,
    "pca": add_pca_args,
    "cl2": add_cl2_args,
    "l2": add_l2_args,
    "ease": add_ease_args,
    "zn": add_zn_args,
    "rr": add_rr_args,
    "tcpr": add_tcpr_args,
    "none": add_none_args,
}

# ======================================================================================================================
# Embedding functions
# ======================================================================================================================

def embed_nohub(features, support_labels, extra_tensors, episode, embedding_args):
    no_hub = NoHub(features, pca_weights=extra_tensors["train_pca_weights"], **embedding_args)
    embeddings, _ = train_no_hub(no_hub, global_step=episode)
    return embeddings


def embed_nohubs(features, support_labels, extra_tensors, episode, embedding_args):
    no_hub_s = NoHubS(features, support_labels, pca_weights=extra_tensors["train_pca_weights"], **embedding_args)
    embeddings, _ = train_no_hub(no_hub_s, global_step=episode)
    return embeddings


def embed_pca(features, support_labels, extra_tensors, episode, embedding_args):
    pca = PCANorm(weights=extra_tensors["train_pca_weights"], **embedding_args)
    embeddings = pca(features)
    return embeddings


def embed_l2(features, support_labels, extra_tensors, episode, embedding_args):
    l2 = L2Norm(pca_weights=extra_tensors["train_pca_weights"], **embedding_args)
    embeddings = l2(features)
    return embeddings


def embed_cl2(features, support_labels, extra_tensors, episode, embedding_args):
    cl2 = CL2Norm(center=extra_tensors["train_mean"], pca_weights=extra_tensors["train_pca_weights"], **embedding_args)
    embeddings = cl2(features)
    return embeddings


def embed_ease(features, support_labels, extra_tensors, episode, embedding_args):
    ease = EASE(**embedding_args)
    embeddings = ease(features)
    return embeddings


def embed_zn(features, support_labels, extra_tensors, episode, embedding_args):
    zn = ZScoreNorm()
    embeddings = zn(features)
    return embeddings


def embed_rr(features, support_labels, extra_tensors, episode, embedding_args):
    rr = ReRepresentation(**embedding_args)
    embeddings = rr(features, support_labels)
    return embeddings


def embed_tcpr(features, support_labels, extra_tensors, episode, embedding_args):
    tcpr = TCPR(**embedding_args)
    embeddings = tcpr(features=features, support_labels=support_labels,
                      base_features=extra_tensors["train_features"])
    return embeddings


def embed_none(features, support_labels, extra_tensors, episode, embedding_args):
    return features


EMBEDDINGS = {
    "nohub": embed_nohub,
    "nohubs": embed_nohubs,
    "pca": embed_pca,
    "l2": embed_l2,
    "cl2": embed_cl2,
    "ease": embed_ease,
    "zn": embed_zn,
    "rr": embed_rr,
    "tcpr": embed_tcpr,
    "none": embed_none,
}
