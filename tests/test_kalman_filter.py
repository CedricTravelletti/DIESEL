import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client
import diesel as ds
from diesel.kalman_filtering import EnsembleKalmanFilter
from diesel.utils import compute_RE_score


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=60)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.1)
    
    # Compute compressed SVD.
    svd_rank = 900 # Since our matrix is 900 * 900 this will be a full SVD.
    u, s, v = da.linalg.svd_compressed(
                    lazy_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)
    
    # Sample 30 ensemble members.
    n_ensembles = 30
    ensembles = sampler.sample(n_ensembles + 1) # Note this is still lazy.

    # Use the first sample as ground truth.
    ground_truth = ensembles[0]
    ensembles = ensembles[1:]

    # Trigger computations.
    ground_truth = ground_truth.compute()
    ensembles = [ensemble.compute() for ensemble in ensembles]

    # Estimate covariance using empirical covariance of the ensemble.
    estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)

    # Persist the covariance on the cluster.
    estimated_cov = client.persist(estimated_cov_lazy)

    # Prepare some data by randomly selecting some points.
    n_data = 60
    data_inds = np.random.choice(ground_truth.shape[0], n_data, replace=False)  

    #  Built observation operator.
    G = np.zeros((data_inds.shape[0], ground_truth.shape[0]))
    G[range(data_inds.shape[0]), data_inds] = 1
    G = da.from_array(G)

    data_std = 0.01
    y = G @ ground_truth

    # Plot data location.
    fig, ax = plt.subplots()
    grid.plot_vals(ground_truth, ax, points=grid_pts[data_inds])
    plt.show()

    # Compute ensemble mean.
    mean = da.mean(da.stack(ensembles, axis=1), axis=1)

    # Run data assimilation using an ensemble Kalman filter.
    my_filter = EnsembleKalmanFilter()
    mean_updated = my_filter.update_mean(mean, G, y, data_std, estimated_cov)

    fig, axs = plt.subplots(1, 2)
    grid.plot_vals(ground_truth, axs[0], points=grid_pts[data_inds])
    grid.plot_vals(mean_updated.compute(), axs[1], points=grid_pts[data_inds])
    plt.savefig("kalman_filter_comparison", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()

    fig, ax = plt.subplots()
    RE_score = compute_RE_score(mean, mean_updated, ground_truth)
    ax = grid.plot_vals(RE_score.compute(), ax, points=grid_pts[data_inds],
            vmin=-10, vmax=1)
    plt.show()

    


if __name__ == "__main__":
    main()
