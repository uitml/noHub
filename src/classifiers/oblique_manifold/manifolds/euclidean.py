"""Euclidean manifold."""
from classifiers.oblique_manifold.manifolds.base import Manifold


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp):
        return dp

    def proj(self, p):
        return p

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p):
        return p + u

    def logmap(self, p1, p2):
        return p2 - p1

    def expmap0(self, u):
        return u

    def logmap0(self, p):
        return p

    def mobius_add(self, x, y, dim=-1):
        return x + y

    def mobius_matvec(self, m, x):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v):
        return v

    def ptransp0(self, x, v):
        return x + v

    def retr(self, x, u):
        return self.expmap(u, x)

    def retr_transp(self, x, u, v):
        y = self.retr(x, u)
        v_transp = self.ptransp(x, y, v)
        return y, v_transp
