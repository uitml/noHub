import logging
import torch as th

logger = logging.getLogger(__name__)


@th.no_grad()
def get_pca_weights(inputs):
    logger.debug(f"Running PCA on inputs with shape = {inputs.size()}.")

    if inputs.ndim == 2:
        drop_first_dim = True
        # Insert a dummy first dimension to emulate batched computation with batch size = 1
        inputs = inputs[None, :, :]
    else:
        drop_first_dim = False

    # Center data
    inputs -= inputs.mean(dim=1, keepdim=True)
    # Compute eigenvectors
    _, eig_vec = th.linalg.eigh(inputs.transpose(1, 2) @ inputs)

    # Eigenvectors are ordered by ascending eigenvalues, so we flip it to descending. This way the principal components
    # that explain the most variance are first in the array.
    eig_vec = th.flip(eig_vec, dims=[2])

    if drop_first_dim:
        eig_vec = eig_vec[0]

    return eig_vec
