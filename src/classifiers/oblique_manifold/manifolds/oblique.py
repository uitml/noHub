import itertools

import torch

from classifiers.oblique_manifold.manifolds.base import Manifold

EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class Oblique(Manifold):
    def __init__(self):
        super().__init__()
        self.name = 'Oblique'

    def proj(self, p):
        return p / p.norm(dim=-1, keepdim=True)

    def proj_tan(self, u, p):
        u = u - (p * u).sum(dim=-1, keepdim=True) * p
        return u

    def expmap(self, u, p):
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = p * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.proj(p + u)
        cond = norm_u > EPS[p.dtype]
        return torch.where(cond, exp, retr)

    def logmap(self, p1, p2):
        u = self.proj_tan(p1 - p2, p2)
        dist = self.dist(p2, p1, keepdim=True)
        cond = dist.gt(EPS[p1.dtype])
        result = torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[p1.dtype]), u)
        return result

    def dist(self, p1, p2, keepdim=False):
        inner = self.inner(p2, p2, p1, keepdim=keepdim).clamp(-1 + EPS[p1.dtype], 1 - EPS[p1.dtype])
        return torch.acos(inner)

    def inner(self, p, u, v=None, keepdim=False):
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        target_shape = self.broadcast_shapes(p.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)

    def broadcast_shapes(self, *shapes):
        """Apply numpy broadcasting rules to shapes."""
        result = []
        for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
            dim: int = 1
            for d in dims:
                if dim != 1 and d != 1 and d != dim:
                    raise ValueError("Shapes can't be broadcasted")
                elif d > dim:
                    dim = d
            result.append(dim)
        return tuple(reversed(result))

    def ptransp(self, x, y, u):
        v_trans = self.proj_tan(u, y)
        return v_trans

    def egrad2rgrad(self, p, dp):
        return self.proj_tan(dp, p)

    def retr(self, x, u):
        # return self.proj(x + u)
        return self.expmap(u, x)

    def retr_transp(self, x, u, v):
        y = self.retr(x, u)
        v_transp = self.ptransp(x, y, v)
        return y, v_transp

