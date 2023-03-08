import torch as th

import config


@config.record_defaults(prefix="k_occ")
def k_occurrence(features, k=5, metric="sqeuclidean"):
    # Compute distances
    if metric == "cosine":
        normed = th.nn.functional.normalize(features, dim=2, p=2)
        dist = - (normed @ normed.transpose(1, 2))
    elif metric == "sqeuclidean":
        dist = th.cdist(features, features)
    else:
        raise RuntimeError(f"Unknown k_occurrence metric: '{metric}'.")

    # Find distance to k-th neighbor for each sample (within each episode)
    dist_to_kth_neighbor = th.topk(dist, k=k+1, dim=2, largest=False)[0][:, :, -1]
    # Count number of times each point is closer to another point than its k-th neighbor
    p = (dist <= dist_to_kth_neighbor[:, :, None])
    k_occ = p.sum(axis=1)
    # Subtract one to account for self-neighbors
    k_occ -= 1
    return k_occ
