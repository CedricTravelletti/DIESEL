"""Lazy implementation (dask) of some usual covariance kernels. All kernels
allow selecting between standard euclidean metric or haversine metric
(great circle distance on the sphere).

Availables kernels are:
- `Matern32`
- `SquaredExponential`

Our implementation of the pairwise distances is inspired by [the one
is Dask ML](https://github.com/dask/dask-ml/blob/main/dask_ml/metrics/pairwise.py)
"""

import dask.array as da
import numpy as np
from scipy.special import kv, gamma  # For the general Matérn kernel


class MaternKernel:
    """Base class for Matérn kernels."""

    def __init__(self, sigma=1.0, l=1.0):
        """
        Args:
            - sigma: float, variance (scaling factor) of the kernel.
            - l: float or array-like, lengthscale of the kernel. If a float, the same lengthscale
                 is used for all dimensions. If an array-like, it must have length d, and each
                 element specifies the lengthscale for the corresponding dimension.
        """
        self.sigma = sigma
        self.l = l

    def _validate_inputs(self, x, y):
        """Validate inputs and ensure they are Dask arrays."""
        if not isinstance(x, da.Array) or not isinstance(y, da.Array):
            raise ValueError("Inputs x and y must be Dask arrays.")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same number of dimensions.")

    def _normalize_inputs(self, x, y):
        """Normalize inputs by the lengthscale."""
        d = x.shape[1]
        if np.isscalar(self.l):
            l = np.full(d, self.l)  # Convert scalar to array
        elif isinstance(self.l, (list, np.ndarray)) and len(self.l) != d:
            raise ValueError(f"Lengthscale l must be a scalar or an array-like of length {d}.")
        else:
            l = np.asarray(self.l)

        l = da.from_array(l, chunks=l.shape)
        return x / l, y / l

    def _compute_distance(self, x, y):
        """Compute pairwise Euclidean distance between x and y."""
        x_norm = da.sum(x**2, axis=1)[:, None]  # Shape (n, 1)
        y_norm = da.sum(y**2, axis=1)[None, :]  # Shape (1, m)
        xy_dot = x @ y.T  # Shape (n, m)
        distance_sq = x_norm + y_norm - 2 * xy_dot  # Shape (n, m)
        return da.sqrt(distance_sq)  # Shape (n, m)

    def __call__(self, x, y):
        """Compute the covariance matrix between x and y."""
        self._validate_inputs(x, y)
        x_scaled, y_scaled = self._normalize_inputs(x, y)
        distance = self._compute_distance(x_scaled, y_scaled)
        return self._compute_kernel(distance)

    def _compute_kernel(self, distance):
        """Compute the kernel-specific function. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class Matern32(MaternKernel):
    """Matérn 3/2 kernel."""

    def _compute_kernel(self, distance):
        sqrt3 = np.sqrt(3)
        return (self.sigma**2) * (1 + sqrt3 * distance) * da.exp(-sqrt3 * distance)


class Matern52(MaternKernel):
    """Matérn 5/2 kernel."""

    def _compute_kernel(self, distance):
        sqrt5 = np.sqrt(5)
        return (
            (self.sigma**2)
            * (1 + sqrt5 * distance + (5 / 3) * distance**2)
            * da.exp(-sqrt5 * distance)
        )


class ExponentialKernel(MaternKernel):
    """Exponential kernel (Matérn 1/2)."""

    def _compute_kernel(self, distance):
        return (self.sigma**2) * da.exp(-distance)


class GaussianKernel(MaternKernel):
    """Gaussian kernel (squared exponential)."""

    def _compute_kernel(self, distance):
        return (self.sigma**2) * da.exp(-0.5 * distance**2)


class MaternGeneral(MaternKernel):
    """General Matérn kernel for any nu."""

    def __init__(self, nu, sigma=1.0, l=1.0):
        """
        Args:
            - nu: float, parameter of the Matérn kernel (must be positive).
            - sigma: float, variance (scaling factor) of the kernel.
            - l: float or array-like, lengthscale of the kernel.
        """
        super().__init__(sigma, l)
        self.nu = nu

    def _compute_kernel(self, distance):
        nu = self.nu
        sqrt2nu = np.sqrt(2 * nu)
        distance = da.where(distance == 0, 1e-10, distance)  # Avoid division by zero
        term1 = (2 ** (1 - nu)) / gamma(nu)
        term2 = (sqrt2nu * distance) ** nu
        term3 = kv(nu, sqrt2nu * distance)  # Modified Bessel function of the second kind
        return (self.sigma**2) * term1 * term2 * term3
