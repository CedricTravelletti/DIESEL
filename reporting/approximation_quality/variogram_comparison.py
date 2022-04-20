""" Compare true variogram with ensemble estimated one. 

"""
import numpy as np
import pandas as pd
import dask.array as da
from dask.distributed import Client
import diesel as ds


def main():
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(30)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.2)
    lazy_dist_matrix = ds.covariance.distance_matrix(grid_pts)
    
    # Compute compressed SVD.
    svd_rank = 900
    u, s, v = da.linalg.svd_compressed(
                    lazy_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)

    # Alternative sampler.
    chol_sampler = ds.sampling.CholeskySampler(lazy_covariance_matrix)

    # 

    # Sample 30 ensemble members.
    ens_size = 30
    ensembles = sampler.sample(ens_size)
    ensembles = client.compute(ensembles).result()

    chol_ensembles = chol_sampler.sample(ens_size)
    chol_ensembles = client.compute(chol_ensembles).result()
    
    # Estimate covariance using empirical covariance of the ensemble.
    lazy_ens_cov = ds.estimation.empirical_covariance(ensembles)
    chol_lazy_ens_cov = ds.estimation.empirical_covariance(chol_ensembles)

    # Compute variograms.
    true_variog_bins, true_variog_means, true_variog_stds = ds.plotting.compute_variogram(
            lazy_dist_matrix, lazy_covariance_matrix, 20)
    ens_variog_bins, ens_variog_means, ens_variog_stds = ds.plotting.compute_variogram(
            lazy_dist_matrix, lazy_ens_cov, 20)
    chol_variog_bins, chol_variog_means, chol_variog_stds = ds.plotting.compute_variogram(
            lazy_dist_matrix, chol_lazy_ens_cov, 20)

    import matplotlib.pyplot as plt
    plt.plot(true_variog_bins, true_variog_means, c='b')
    plt.plot(ens_variog_bins, ens_variog_means, c='r')
    plt.plot(chol_variog_bins, chol_variog_means, c='g')

    plt.fill_between(
            true_variog_bins, true_variog_means - 3*true_variog_stds,
            true_variog_means + 3*true_variog_stds, color='b', alpha=.2)
    plt.fill_between(ens_variog_bins, ens_variog_means - 3*ens_variog_stds,
            ens_variog_means + 3*ens_variog_stds, color='r', alpha=.2)
    plt.fill_between(chol_variog_bins, chol_variog_means - 3*chol_variog_stds,
            chol_variog_means + 3*chol_variog_stds, color='r', alpha=.2)

    plt.savefig('variogram_comparison.png', bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.show()
    

if __name__ == "__main__":
    main()
