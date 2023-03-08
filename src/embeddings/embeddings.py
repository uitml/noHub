from abc import abstractmethod, ABC
from torch import nn

import config
from pca import get_pca_weights


class BaseL2Norm(ABC, nn.Module):
    def __init__(self, out_dims, pca_mode, pca_weights):
        super(BaseL2Norm, self).__init__()
        self.pca_weights = pca_weights
        self.out_dims = out_dims

        if out_dims is None:
            # No PCA preprocessing
            self._pca_features = self._no_pca_features
        else:
            if pca_mode == "base":
                assert pca_weights is not None, "L2 norm with pca_mode='base' requires 'pca_weights' to be not None."
                self._pca_features = self._pca_features_base
            elif pca_mode == "episode":
                self._pca_features = self._pca_features_episode
            else:
                raise RuntimeError(f"Invalid PCA mode for BaseL2Norm: '{pca_mode}'")

    @staticmethod
    def _pca_transform(features, weights, out_dims, center=None):
        _weights = weights[:, :, :out_dims]
        out = features @ _weights
        if center is None:
            return out
        return out, center @ _weights

    def _pca_features_base(self, features, center=None):
        weights = self.pca_weights[:, :, :self.out_dims]
        return self._pca_transform(features, weights, out_dims=self.out_dims, center=center)

    def _pca_features_episode(self, features, center=None):
        weights = get_pca_weights(features)
        return self._pca_transform(features, weights, out_dims=self.out_dims, center=center)

    @staticmethod
    def _no_pca_features(features, center=None):
        if center is None:
            return features
        return features, center

    @staticmethod
    def _normalize(features):
        return nn.functional.normalize(features, dim=-1, p=2)

    @abstractmethod
    def forward(self, features):
        pass


@config.record_defaults(prefix="l2", ignore=["pca_weights"])
class L2Norm(BaseL2Norm):
    def __init__(self, out_dims=None, pca_mode="base", pca_weights=None):
        super(L2Norm, self).__init__(out_dims=out_dims, pca_mode=pca_mode, pca_weights=pca_weights)
    
    def forward(self, features):
        features = self._pca_features(features)
        features = self._normalize(features)
        return features


@config.record_defaults(prefix="cl2", ignore=["center", "pca_weights"])
class CL2Norm(BaseL2Norm):
    def __init__(self, dim=0, center_mode="base", center=None, out_dims=None, pca_mode="base", pca_weights=None):
        super(CL2Norm, self).__init__(out_dims=out_dims, pca_mode=pca_mode, pca_weights=pca_weights)

        assert dim in {0, 1}, "CL2Norm requires 'dim' to be either 0 (sample mean) or 1 (instance mean)."
        # Offset dim by one because of episode-axis
        self.dim = dim + 1
        self.center_mode = center_mode

        if center_mode == "base":
            assert center is not None, f"CL2 with center_mode='base' requires a pre-computed center. " \
                                       f"Got: '{center_mode}'."
            assert self.dim == 1, f"CL2 with center_mode='base' can only be done when dim=0"
            self.register_buffer(name="center", tensor=center)

        elif center_mode == "episode":
            pass

        else:
            raise RuntimeError(f"Invalid center_mode for CL2: '{center_mode}'.")

    def forward(self, features):
        if self.center_mode == "base":
            features, center = self._pca_features(features, self.center)
        else:
            features = self._pca_features(features)
            center = features.mean(dim=self.dim, keepdims=True)

        centered = features - center
        normed = self._normalize(centered)
        return normed


@config.record_defaults(prefix="pca", ignore=["weights"])
class PCANorm(BaseL2Norm):
    def __init__(self, mode="base", out_dims=64, weights=None):
        super(PCANorm, self).__init__(pca_mode=mode, out_dims=out_dims, pca_weights=weights)

    def forward(self, features):
        return self._pca_features(features)
    

class ZScoreNorm(nn.Module):
    @staticmethod
    def forward(features):
        means = features.mean(dim=2, keepdims=True)
        stds = features.std(dim=2, keepdims=True)
        normed = (features - means) / stds
        return normed
