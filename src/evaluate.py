import logging
import numpy as np
import torch as th
import time
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
import helpers
from args import parse_args
from models import get_model
from wandb_utils import wandb_logger
from data.dataset import get_dataset
from data.episode_sampler import EpisodeSampler
from features import get_features
from pca import get_pca_weights
from classifiers import CLASSIFIERS
from embeddings import EMBEDDINGS
from hubness.metrics import score as score_hubness
from hubness.knn import k_occurrence

logger = logging.getLogger(__name__)


def embed_features(support_features, support_labels, query_features, args, extra_tensors, episode=0):
    features = th.cat((support_features, query_features), dim=1)

    # Get embedding arguments
    embedding_args = getattr(args, args.embedding, None)
    if embedding_args is not None:
        embedding_args = embedding_args.to_dict()

    # Embed features
    if args.embedding in EMBEDDINGS:
        embeddings = EMBEDDINGS[args.embedding](
            features=features, support_labels=support_labels, extra_tensors=extra_tensors, episode=episode,
            embedding_args=embedding_args
        )
    else:
        raise RuntimeError(f"Unknown embedding '{args.embedding}'.")
    # Separate embeddings back into support and query.
    n_supp = support_features.size(1)
    support_features = embeddings[:, :n_supp]
    query_features = embeddings[:, n_supp:]
    return support_features, query_features


def classify_features(support_features, support_labels, query_features, args, episode=0):
    # Get the classifier args
    clf_args = getattr(args, args.classifier, None)
    if clf_args is not None:
        clf_args = clf_args.to_dict()

    # Classify query features
    if args.classifier in CLASSIFIERS:
        query_predictions = CLASSIFIERS[args.classifier](
            support_features=support_features, support_labels=support_labels, query_features=query_features,
            clf_args=clf_args, episode=episode, n_shots=args.n_shots, n_ways=args.n_ways, n_queries=args.n_queries
        )
    else:
        raise RuntimeError(f"Unknown classifier: '{args.classifier}'.")

    return query_predictions


def compute_classification_metrics(query_labels, query_predictions):
    return {
        "accuracy": (query_labels == query_predictions).float().mean(dim=1)
    }


def compute_hubness_metrics(args, support_features, support_labels, query_features, query_labels):
    if not args.compute_hubness:
        return {}

    features = th.cat((support_features, query_features), dim=1)
    k_occ = k_occurrence(features, k=args.k_occ.k, metric=args.k_occ.metric)
    scores = score_hubness(k_occ, k=args.k_occ.k)
    return scores


def aggregate_metrics(mtc, name):
    if not isinstance(mtc, np.ndarray):
        mtc = np.array(mtc)
    return {
        f"{name}.mean": mtc.mean(),
        f"{name}.std": mtc.std(),
        f"{name}.conf": (1.96 * (mtc.std() / np.sqrt(mtc.shape[0]))),
        f"{name}.max": mtc.max(),
        f"{name}.min": mtc.min(),
    }


def evaluate_episode(support, query, args, extra_tensors, episode=0):
    support_features, support_labels = support
    query_features, query_labels = query

    support_features, query_features = embed_features(support_features, support_labels, query_features, args,
                                                      extra_tensors, episode=episode)

    query_predictions = classify_features(support_features, support_labels, query_features, args, episode=episode)
    metrics = compute_classification_metrics(query_labels, query_predictions)
    hub_metrics = compute_hubness_metrics(args, support_features, support_labels, query_features, query_labels)
    metrics.update(hub_metrics)
    return metrics


def evaluate(features, labels, args, extra_tensors):
    logger.info(f"Evaluating {args.n_episodes} episodes.")
    episode_sampler = EpisodeSampler(
        features=features, labels=labels, n_ways=args.n_ways, n_shots=args.n_shots, n_queries=args.n_queries,
        n_episodes=args.n_episodes, batch_size=args.episode_batch_size,
    )

    metrics = []
    for episode, (support, query) in tqdm(enumerate(episode_sampler), total=len(episode_sampler)):
        episode_metrics = evaluate_episode(support, query, args, extra_tensors, episode=episode)
        metrics.append(helpers.npy(episode_metrics))

    return metrics


@th.no_grad()
def get_extra_tensors(args, model):
    if args.use_cached:
        # No need to create a train loader when we're using cached features
        train_loader = None
        train_dataset = None
    else:
        train_dataset = get_dataset(name=args.dataset, split="train")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, drop_last=False
        )

    requires_pca_weights = \
        ((args.embedding == "nohub") and (args.nohub.pca_mode == "base")) \
        or ((args.embedding == "nohubs") and (args.nohubs.pca_mode == "base")) \
        or ((args.embedding == "pca") and (args.pca.mode == "base")) \
        or ((args.embedding == "l2") and (args.l2.pca_mode == "base")) \
        or ((args.embedding == "cl2") and (args.cl2.pca_mode == "base"))

    requires_mean = (args.embedding == "cl2") and (args.cl2.center_mode == "base")

    requires_features = (args.embedding == "tcpr")

    if any([requires_pca_weights, requires_mean, requires_features]):
        train_features, train_labels = get_features(model, train_loader, args.cache_dir, args.use_cached, split="train")
    else:
        train_features = train_labels = None

    extra_tensors = {
        "train_mean": train_features.mean(dim=args.cl2.dim, keepdims=True)[None] if requires_mean else None,
        "train_pca_weights": get_pca_weights(train_features)[None] if requires_pca_weights else None,
        "train_features": train_features
    }
    return extra_tensors


def log_metrics(metrics):
    # Log metrics (losses, etc) accumulated during evaluations
    wandb_logger.log_accumulated()

    # Log aggregated metrics
    for metric_name, values in metrics.items():
        wandb.summary.update(aggregate_metrics(mtc=values, name=metric_name))

    # Log metric histogram
    data = np.stack(list(metrics.values()), axis=1)
    # Jitter the data a little to avoid duplicate values being filtered out by WandB
    data += np.random.normal(0, 1e-4, size=data.shape)
    # Create table and log histogram
    table = wandb.Table(data=data.tolist(), columns=list(metrics.keys()))
    for metric_name in metrics.keys():
        wandb.log({
            f"{metric_name}.histogram": wandb.plot.histogram(table, metric_name,
                                                             title=f"{metric_name.capitalize()} histogram")
        })


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=config.LOG_FORMAT)
    logger.info(f"Running evaluation with config:\n{str(args)}")
    wandb_logger.init(args)

    if args.use_cached:
        # No need to load backbone model when we're using cached features.
        model = None
    else:
        model = get_model(arch=args.arch, checkpoint_file=args.checkpoint, dataset_name=args.dataset)

    # Compute additional tensors required for the evaluation.
    extra_tensors = get_extra_tensors(args, model)

    # Get test features
    if args.use_cached:
        # No need to create a test loader when we're using cached features
        test_loader = None
    else:
        test_dataset = get_dataset(args.dataset, split=args.eval_split)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, drop_last=False
        )
    features, labels = get_features(model, test_loader, args.cache_dir, args.use_cached, args.eval_split)

    start = time.time()
    metrics = evaluate(features, labels, args, extra_tensors)

    episode_time = (time.time() - start) / args.n_episodes
    logger.info(f"Evaluation finished. {round(episode_time, 4)} seconds/episode")
    wandb.summary["episode_time"] = episode_time

    metrics = helpers.dict_cat(metrics)
    log_metrics(metrics)


if __name__ == '__main__':
    main()
