""" Check the effect of observation ordering on the sequential assimilation.

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, wait, progress
import diesel as ds
from diesel.scoring import compute_RE_score, compute_energy_score
from diesel.estimation import localize_covariance


import time


# results_folder ="/home/cedric/PHD/Dev/DIESEL/reporting/toy_example/results/"
results_folder ="/storage/homefs/ct19x463/Dev/DIESEL/reporting/toy_example/results_paper/synthetic_ordering/"


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    # cluster = ds.cluster.LocalCluster()
    cluster = ds.cluster.UbelixCluster(n_nodes=12, mem_per_node=64, cores_per_node=3,
            partition="gpu", qos="job_gpu")
    cluster.scale(10)
    client = Client(cluster)

    # Add to builtins so we have one global client.
    __builtins__.CLIENT = client
    from diesel.kalman_filtering import EnsembleKalmanFilter
    
    # Build a square grid with 80^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=80)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lambda0=0.1
    lengthscales = da.from_array([lambda0])
    kernel = ds.covariance.matern32(lengthscales)
    true_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts)
    
    # Compute compressed SVD.
    svd_rank = 900 # Since our matrix is 900 * 900 this will be a full SVD.
    u, s, v = da.linalg.svd_compressed(
                    true_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)
    
    # Repeat the whole experiment several time for statistical analysis.
    ES_prior, ES_aao_loc, ES_seq_loc, ES_aao_truecov = [], [], [], []
    RE_aao_loc, RE_seq_loc, RE_aao_truecov = [], [], []
    RMSE_prior, RMSE_aao_loc, RMSE_seq_loc, RMSE_aao_truecov = [], [], [], []

    rep = 0
    # Sample 30 ensemble members.
    n_ensembles = 30
    ensembles = sampler.sample(n_ensembles + 1) # Note this is still lazy.

    # Use the first sample as ground truth.
    ground_truth = ensembles[0]
    ensembles = ensembles[1:]

    # Compute ensemble mean.
    mean = da.mean(da.stack(ensembles, axis=1), axis=1)
    
    # Trigger computations.
    ground_truth = client.persist(ground_truth)
    ensembles = [client.persist(ensemble) for ensemble in ensembles]

    # Stack ensembles so are in the format required later.
    ensembles = da.stack(ensembles)

    # Save for later.
    np.save(os.path.join(
        results_folder, "ground_truth_{}.npy".format(rep)), ground_truth.compute())
    np.save(os.path.join(
            results_folder, "ensemble_{}.npy".format(rep)), ensembles.compute())
    np.save(os.path.join(
        results_folder, "mean_{}.npy".format(rep)), mean.compute())
    
    # Estimate covariance using empirical covariance of the ensemble.
    raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)
    raw_estimated_cov = client.persist(raw_estimated_cov_lazy)

    # Perform covariance localization (use scaled version of base covariance to localize).
    # Maybe should persist here.
    localization_matrix = kernel.covariance_matrix(grid_pts, grid_pts, 
            lengthscales=da.from_array([2 * lambda0]))
    localization_matrix = client.persist(localization_matrix)
    loc_estimated_cov = localize_covariance(raw_estimated_cov, localization_matrix)
    loc_estimated_cov = client.persist(loc_estimated_cov)

    # Prepare some data by randomly selecting some points.
    n_data = 500
    data_inds = np.random.choice(ground_truth.shape[0], n_data, replace=False)  
    np.save(os.path.join(
        results_folder, "data_inds_{}.npy".format(rep)), data_inds)

    n_rep = 20
    for rep in range(n_rep):
        # Shuffle data ordering.
        np.random.shuffle(data_inds)

        print("Repetition {} / {}.".format(rep, n_rep))
        #  Built observation operator.
        G = np.zeros((data_inds.shape[0], ground_truth.shape[0]))
        G[range(data_inds.shape[0]), data_inds] = 1
        G = da.from_array(G)
    
        data_std = 0.01
        y = G @ ground_truth
    
        # Run data assimilation using an ensemble Kalman filter.
        my_filter = EnsembleKalmanFilter()

        # ------------------------
        # Sequential assimilation.
        # ------------------------
        mean_updated_seq_loc, ensemble_updated_seq_loc = my_filter.update_ensemble_sequential_nondask(
                mean, ensembles, G, y, data_std, localization_matrix)

        np.save(os.path.join(
                results_folder, "mean_updated_seq_loc_{}.npy".format(rep)),
                mean_updated_seq_loc)
        np.save(os.path.join(
                results_folder, "ensemble_updated_seq_loc_{}.npy".format(rep)),
                ensemble_updated_seq_loc)

        # Compute scores and save.
        ES, _, _ = compute_energy_score(ensembles.compute(), ground_truth.compute())
        ES_prior.append(ES)
        ES, _, _ = compute_energy_score(ensemble_updated_seq_loc, ground_truth.compute())

        RE = np.median(compute_RE_score(mean.compute(), mean_updated_seq_loc, ground_truth.compute()))
        RE_seq_loc.append(RE)

        RMSE_prior.append(np.sqrt(np.mean((mean.compute().reshape(-1, 1) - ground_truth.compute().reshape(-1, 1))**2)))
        RMSE_seq_loc.append(np.sqrt(np.mean((mean_updated_seq_loc.reshape(-1, 1) - ground_truth.compute().reshape(-1, 1))**2)))

        df_results = pd.DataFrame({
            'RMSE prior': RMSE_prior, 'RMSE seq loc': RMSE_seq_loc,
            'ES prior': ES_prior, 'ES seq loc': ES_seq_loc,
            'RE seq loc': RE_seq_loc, )
        df_results.to_pickle(os.path.join(results_folder, 'scores.pkl'))


if __name__ == "__main__":
    main()
