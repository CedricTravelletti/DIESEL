""" Compare the performance of raw covariance estimation with localization 
on a simple 2D example. Comparison done with the RE score.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client
import diesel as ds
from diesel.kalman_filtering import EnsembleKalmanFilter
from diesel.utils import compute_RE_score
from diesel.estimation import localize_covariance


results_folder ="/home/cedric/PHD/Dev/DIESEL/reporting/toy_example/results/"
# results_folder ="/storage/homefs/ct19x463/Dev/DIESEL/reporting/toy_example/results/"


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=30)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lambda0=0.1
    lengthscales = da.from_array([lambda0])
    kernel = ds.covariance.matern32(lengthscales)
    lazy_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts)
    
    # Compute compressed SVD.
    svd_rank = 900 # Since our matrix is 900 * 900 this will be a full SVD.
    u, s, v = da.linalg.svd_compressed(
                    lazy_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)
    
    n_rep = 30
    for rep in range(n_rep):
        print("Repetition {} / {}.".format(rep, n_rep))
        # Sample 30 ensemble members.
        n_ensembles = 240
        ensembles = sampler.sample(n_ensembles + 1) # Note this is still lazy.
    
        # Use the first sample as ground truth.
        ground_truth = ensembles[0]
        ensembles = ensembles[1:]
    
        # Trigger computations.
        ground_truth = client.persist(ground_truth)
        np.save(os.path.join(results_folder, "ground_truth_{}.npy".format(rep)), ground_truth.compute())
        ensembles = [client.compute(ensemble).result() for ensemble in ensembles]
    
        # Estimate covariance using empirical covariance of the ensemble.
        raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)
    
        # Persist the covariance on the cluster.
        raw_estimated_cov = client.persist(raw_estimated_cov_lazy)

        # Perform covariance localization (use scaled version of base covariance to localize).
        # Maybe should persist here.
        scaled_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts, 
                lengthscales=da.from_array([0.5 * lambda0]))
        loc_estimated_cov = localize_covariance(raw_estimated_cov, lazy_covariance_matrix)
    
        # Prepare some data by randomly selecting some points.
        n_data = 500
        data_inds = np.random.choice(ground_truth.shape[0], n_data, replace=False)  
    
        #  Built observation operator.
        G = np.zeros((data_inds.shape[0], ground_truth.shape[0]))
        G[range(data_inds.shape[0]), data_inds] = 1
        G = da.from_array(G)
    
        data_std = 0.01
        y = G @ ground_truth
    
        # Compute ensemble mean.
        mean = da.mean(da.stack(ensembles, axis=1), axis=1)
    
        # Run data assimilation using an ensemble Kalman filter.
        my_filter = EnsembleKalmanFilter()
        mean_updated_raw = my_filter.update_mean(mean, G, y, data_std, raw_estimated_cov)
        mean_updated_loc = my_filter.update_mean(mean, G, y, data_std, loc_estimated_cov)

        
    
        RE_score_raw = compute_RE_score(mean, mean_updated_raw, ground_truth)
        RE_score_loc = compute_RE_score(mean, mean_updated_loc, ground_truth)

        print("RE score raw: {}.".format(da.median(RE_score_raw, axis=0).compute()))
        print("RE score localization: {}.".format(da.median(RE_score_loc, axis=0).compute()))

        fig, ax = plt.subplots()
        grid.plot_vals(ground_truth, ax)
        plt.savefig("ground_truth", bbox_inches="tight", pad_inches=0.1, dpi=400)

        fig, ax = plt.subplots()
        grid.plot_vals(mean_updated_raw.compute(), ax)
        plt.savefig("mean_updated_raw", bbox_inches="tight", pad_inches=0.1, dpi=400)

        fig, ax = plt.subplots()
        grid.plot_vals(mean_updated_loc.compute(), ax)
        plt.savefig("mean_updated_loc", bbox_inches="tight", pad_inches=0.1, dpi=400)

        fig, ax = plt.subplots()
        grid.plot_vals(mean, ax)
        plt.savefig("mean", bbox_inches="tight", pad_inches=0.1, dpi=400)

        fig, ax = plt.subplots()
        grid.plot_vals(RE_score_raw.compute(), ax, points=grid_pts[data_inds],
                vmin=-10, vmax=1,
                fig=fig, colorbar=True)
        plt.savefig("re_score_raw", bbox_inches="tight", pad_inches=0.1, dpi=400)

        fig, ax = plt.subplots()
        grid.plot_vals(RE_score_loc.compute(), ax, points=grid_pts[data_inds],
                vmin=-10, vmax=1,
                fig=fig, colorbar=True)
        plt.savefig("re_score_loc", bbox_inches="tight", pad_inches=0.1, dpi=400)


if __name__ == "__main__":
    main()
