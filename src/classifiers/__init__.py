from config.parser.types import *

from .simpleshot import SimpleShot
from .tim.tim import TIMGD
from .laplacianshot import LaplacianShot
from .tim.alpha_tim import AlphaTIM
from .siamese import SIAMESE
from .ilpc.ilpc import ILPC
from .oblique_manifold import ObliqueManifoldClassifier


# ======================================================================================================================
# Argument functions
# ======================================================================================================================

def add_args_simpleshot(parser):
    pass


def add_args_tim(parser):
    parser.add_argument_with_default("--tim.temp", type=float)
    parser.add_argument_with_default("--tim.loss_weights", type=float, nargs=3)
    parser.add_argument_with_default("--tim.n_iter", type=int)
    parser.add_argument_with_default("--tim.lr", type=float)


def add_args_laplacianshot(parser):
    parser.add_argument_with_default("--laplacianshot.lam", type=float)
    parser.add_argument_with_default("--laplacianshot.knn", type=int)
    parser.add_argument_with_default("--laplacianshot.max_iter", type=int)
    parser.add_argument_with_default("--laplacianshot.rel_tol", type=float)
    parser.add_argument_with_default("--laplacianshot.rectify_prototypes", type=str_to_bool)
    parser.add_argument_with_default("--laplacianshot.symmetrize_affinity", type=str_to_bool)
    parser.add_argument_with_default("--laplacianshot.normalize_prototypes", type=str_to_bool)


def add_args_alpha_tim(parser):
    parser.add_argument_with_default("--alpha_tim.temp", type=float)
    parser.add_argument_with_default("--alpha_tim.loss_weights", type=float, nargs=3)
    parser.add_argument_with_default("--alpha_tim.n_iter", type=int)
    parser.add_argument_with_default("--alpha_tim.lr", type=float)
    parser.add_argument_with_default("--alpha_tim.entropies", type=str_lower, nargs=3)
    parser.add_argument_with_default("--alpha_tim.alpha_values", type=str_or_none, nargs=3)
    parser.add_argument_with_default("--alpha_tim.use_tuned_alpha_values", type=str_to_bool)


def add_args_siamese(parser):
    parser.add_argument_with_default("--siamese.lam", type=float)
    parser.add_argument_with_default("--siamese.alpha", type=float)
    parser.add_argument_with_default("--siamese.epsilon", type=float)
    parser.add_argument_with_default("--siamese.optimal_transport_max_iter", type=int)
    parser.add_argument_with_default("--siamese.n_iter", type=int)


def add_args_ilpc(parser):
    parser.add_argument_with_default("--ilpc.k", type=int)
    parser.add_argument_with_default("--ilpc.alpha", type=float)
    parser.add_argument_with_default("--ilpc.best_samples", type=int)
    parser.add_argument_with_default("--ilpc.denoising_iterations", type=int)
    parser.add_argument_with_default("--ilpc.tau", type=float)
    parser.add_argument_with_default("--ilpc.sinkhorn_iter", type=int)


def add_args_om(parser):
    parser.add_argument_with_default("--om.t", type=int)
    parser.add_argument_with_default("--om.lr_weights", type=float)
    parser.add_argument_with_default("--om.lr_anchors", type=float)
    parser.add_argument_with_default("--om.train_meta_epochs", type=int)
    parser.add_argument_with_default("--om.loss_weights", type=float, nargs=3)
    parser.add_argument_with_default("--om.scale_factor", type=float)


CLASSIFIER_ARG_ADDERS = {
    "simpleshot": add_args_simpleshot,
    "tim": add_args_tim,
    "laplacianshot": add_args_laplacianshot,
    "alpha_tim": add_args_alpha_tim,
    "siamese": add_args_siamese,
    "ilpc": add_args_ilpc,
    "om": add_args_om,
}


# ======================================================================================================================
# Classification functions
# ======================================================================================================================

def classify_simpleshot(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways,
                        n_queries):
    clf = SimpleShot(support_features=support_features, support_labels=support_labels)
    query_predictions = clf(query_features)
    return query_predictions


def classify_tim(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways, n_queries):
    tim = TIMGD(support=support_features, query=query_features, y_s=support_labels, **clf_args)
    tim.run_adaptation(support=support_features, y_s=support_labels, query=query_features, global_step=episode)
    query_predictions = tim(query_features)
    return query_predictions


def classify_laplacianshot(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways,
                           n_queries):
    laplacian_shot = LaplacianShot(**clf_args)
    query_predictions = laplacian_shot.fit_predict(
        support_features=support_features, support_labels=support_labels, query_features=query_features,
        global_step=episode
    )
    return query_predictions


def classify_alpha_tim(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways, n_queries):
    alpha_tim = AlphaTIM(support=support_features, query=query_features, y_s=support_labels, shot=n_shots,
                         **clf_args)
    alpha_tim.run_adaptation(support=support_features, y_s=support_labels, query=query_features,
                             global_step=episode)
    query_predictions = alpha_tim(query_features)
    return query_predictions


def classify_siamese(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways, n_queries):
    siamese = SIAMESE(support_features, support_labels, n_ways, n_shots, n_queries, **clf_args)
    query_predictions = siamese.fit_predict(support_features, support_labels, query_features)
    return query_predictions


def classify_ilpc(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways, n_queries):
    ilpc = ILPC(n_queries=n_queries, n_ways=n_ways, **clf_args)
    query_predictions = ilpc.fit_predict(support_features=support_features, query_features=query_features,
                                         support_labels=support_labels)
    return query_predictions


def classify_om(support_features, support_labels, query_features, clf_args, episode, n_shots, n_ways, n_queries):
    om = ObliqueManifoldClassifier(n_ways=n_ways, n_shots=n_shots, **clf_args)
    query_predictions = om.fit_predict(support_features=support_features, query_features=query_features,
                                       support_labels=support_labels)
    return query_predictions


CLASSIFIERS = {
    "simpleshot": classify_simpleshot,
    "tim": classify_tim,
    "laplacianshot": classify_laplacianshot,
    "alpha_tim": classify_alpha_tim,
    "siamese": classify_siamese,
    "ilpc": classify_ilpc,
    "om": classify_om,
}
