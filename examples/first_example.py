"""
first_example.py

This example demonstrates how to use the DIESEL package for distributed estimation of ensemble covariance matrices.
It includes the following steps:
1. Instantiate a local Dask cluster.
2. Build a square grid.
3. Construct a lazy covariance matrix using the Matern 3/2 kernel.
4. Compute a compressed SVD of the covariance matrix.
5. Sample ensemble members from the SVD.
6. Plot one ensemble member.
7. Estimate the covariance using the empirical covariance of the ensemble.
8. Compute the Frobenius norm distance between the true and estimated covariance matrices.

Usage:
    python first_example.py
"""

import dask.array as da
from dask.distributed import Client
import numpy as np
import diesel as ds
import matplotlib.pyplot as plt


def main():
    """
    Main function to run the example.
    """
    # Instantiate a local cluster, to mimic distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
     # Add to builtins so we have one global client.
    __builtins__.CLIENT = client

    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=30)
    grid_pts = grid.grid_pts

    # Construct (lazy) covariance matrix using the Matern 3/2 kernel.
    kernel = ds.covariance.matern32(lengthscales=np.array([0.1, 0.1]))
    lazy_covariance_matrix = kernel.covariance_matrix(grid_pts)

    # Compute compressed SVD.
    svd_rank = 900  # Since our matrix is 900 * 900 this will be a full SVD.
    u, s, v = da.linalg.svd_compressed(
        lazy_covariance_matrix, k=svd_rank, compute=False
    )

    # Construct sampler from the SVD of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)

    # Sample 16 ensemble members.
    ensembles = sampler.sample(16)  # Note this is still lazy.

    # Plot one ensemble member.
    plt.imshow(grid.list_to_mesh(ensembles[0]), cmap="jet")
    plt.title("Sampled Ensemble Member")
    plt.colorbar()
    plt.show()

    # Estimate covariance using empirical covariance of the ensemble.
    estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)

    # Compute distance in Frobenius norm between true covariance and estimated covariance.
    dist = da.linalg.norm(lazy_covariance_matrix - estimated_cov_lazy, ord="fro")
    dist = client.compute(dist).result()
    print(
        "Frobenius distance between true covariance matrix and sample covariance: {}.".format(
            dist
        )
    )


if __name__ == "__main__":
    main()
