"""Base manifold."""

from torch.nn import Parameter


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def init_weights(self, w, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError

    def retr(self, x, u):
        """Perform a retraction from point x with given direction u"""
        raise NotImplementedError

    def retr_transp(self, x, u, v):
        """Perform a retraction + vector transport at once."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """

    def __new__(cls, data, requires_grad, manifold):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold):
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
