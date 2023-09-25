""" 22.09.2023 

Report memory usage for Kalman all-at-once paper.

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, wait, progress
import diesel.covariance, diesel.cluster, diesel.gridding, diesel.sampling
from diesel.scoring import compute_RE_score, compute_energy_score
from dask.distributed.diagnostics import MemorySampler

from dask.distributed.client import futures_of

import time


# results_folder ="/home/cedric/PHD/Dev/DIESEL/reporting/toy_example/results/"
results_folder ="/storage/homefs/ct19x463/Dev/DIESEL/reporting/toy_example/results_paper/report_mem_usage/"


CHUNK_SIZE = 3000
n_pts_1d = 300
n_data = 10000


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    # cluster = ds.cluster.LocalCluster()
    cluster = diesel.cluster.UbelixCluster(n_nodes=15, mem_per_node=64, cores_per_node=4,
            partition="gpu", qos="job_gpu")
    cluster.scale(20)
    client = Client(cluster)

    # Add to builtins so we have one global client.
    __builtins__.CLIENT = client

    # This has to be imported later, otherwise we do not know 
    # which client to use.
    from diesel.kalman_filtering import EnsembleKalmanFilter
    from diesel.estimation import localize_covariance, empirical_covariance

    # ----------------
    # Start profiling.
    # ----------------
    ms = MemorySampler()
    with ms.sample("collection 1"):
        
        # Build a square grid with 80^2 elements.
        grid = diesel.gridding.SquareGrid(n_pts_1d=n_pts_1d)
        grid_pts = grid.grid_pts.astype('float32')

        # Chunk it so that localization matrices built out of the coordinates 
        # are chunked too.
        grid_pts = grid_pts.rechunk((CHUNK_SIZE, -1))
        wait(grid_pts)
        print(grid_pts)

        # Construct (lazy) covariance matrix.
        lambda0 = 0.05
        lengthscales = da.from_array([lambda0])
        kernel = diesel.covariance.matern32(lengthscales)
        true_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts)

        print(true_covariance_matrix)
        
        # Compute compressed SVD.
        svd_rank = 1000 # Since our matrix is 900 * 900 this will be a full SVD.
        u, s, v = da.linalg.svd_compressed(
                        true_covariance_matrix, k=svd_rank, compute=False) 
        u = client.persist(u)
        s = client.persist(s)
        wait(u)
        wait(s)
        # Aggressively clean memory.
        client.cancel(true_covariance_matrix)
        print("Finished waiting.")
    
        # Save for later.
        u_res = client.compute(u)
        s_res = client.compute(s)
        np.save(os.path.join(
                results_folder, "svd_u.npy"), u_res)
        np.save(os.path.join(
                results_folder, "svd_s.npy"), s_res)
        print("Saving SVD done.")
        
        # Construct sampler from the svd of the covariance matrix.
        sampler = diesel.sampling.SvdSampler(u, s)
        
        # Repeat the whole experiment several time for statistical analysis.
        ES_prior, ES_aao_loc = [], []
        RE_aao_loc = []
        RMSE_prior, RMSE_aao_loc = [], []
        n_rep = 1
        for rep in range(n_rep):
            print("Repetition {} / {}.".format(rep, n_rep))
            # Sample 30 ensemble members.
            n_ensembles = 30
            ensembles = sampler.sample(n_ensembles + 1) # Note this is still lazy.
            print("Sampling done.")
        
            # Use the first sample as ground truth.
            ground_truth = ensembles[0]
            ensembles = ensembles[1:]

            # Compute ensemble mean.
            mean = da.mean(da.stack(ensembles, axis=1), axis=1)
        
            # Trigger computations.
            ground_truth = client.persist(ground_truth)
            mean = client.persist(mean)

            # Save ensembles locally so they survive cleaning.
            ensembles_local = [client.compute(ensemble.astype('float32')).result()
                    for ensemble in ensembles]
            # Aggressively clear memory.
            client.cancel(s)
            client.cancel(u)
            # Persist on cluster and stack.
            ensembles_cluster = [client.persist(
                da.from_array(ensemble).astype('float32'))
                    for ensemble in ensembles_local]
            ensembles_cluster = da.stack(ensembles_cluster)

            # Chunk so later computations fit in memory.
            ensembles_cluster = client.persist(ensembles_cluster.rechunk((-1, CHUNK_SIZE)))
            print(ensembles_cluster.chunks)

            wait(ensembles_cluster)
            wait(mean)
            print("Finished waiting on ensembles")
    
            # Stack ensembles so are in the format required later.
    
            # Save for later.
            np.save(os.path.join(
                results_folder, "ground_truth_n1d_{}.npy".format(n_pts_1d)), ground_truth.compute())
            np.save(os.path.join(
                results_folder, "ensemble_n1d_{}.npy".format(n_pts_1d)), ensembles_cluster.compute())
            np.save(os.path.join(
                results_folder, "mean_n1d_{}.npy".format(n_pts_1d)), mean.compute())
            print("Saving starting conditions done.")
        
            # Estimate covariance using empirical covariance of the ensemble.
            raw_estimated_cov_lazy = empirical_covariance(ensembles_cluster)
            # raw_estimated_cov = client.persist(raw_estimated_cov_lazy)
    
            # Perform covariance localization (use scaled version of base covariance to localize).
            # Maybe should persist here.
            # grid_pts = client.persist(grid_pts_local)
            localization_matrix = kernel.covariance_matrix(grid_pts, grid_pts, 
                    lengthscales=da.from_array([2 * lambda0]))
            # TODO: delete once bug found.
            # TODO: Result of this comand is found to increase memory footprint 
            # but drastically increase speed.
            localization_matrix_pers = client.persist(localization_matrix)

            # localization_matrix = client.persist(localization_matrix)
            loc_estimated_cov = localize_covariance(raw_estimated_cov_lazy, localization_matrix_pers)
            loc_estimated_cov = client.persist(loc_estimated_cov)
    
            # Wait for the localized estimated covariance 
            # to be loaded in distributed memory.
            wait(loc_estimated_cov)
            del localization_matrix_pers

            # client.cancel(raw_estimated_cov_lazy)
            # client.cancel(localization_matrix)
            print("Finished waiting on localized estimated covariance.")
        
            # Prepare some data by randomly selecting some points.
            data_inds = np.random.choice(ground_truth.shape[0], n_data, replace=False)  
            np.save(os.path.join(
                results_folder, "data_inds_n1d_{}.npy".format(n_pts_1d)), data_inds)
        
            #  Built observation operator.
            G = np.zeros((data_inds.shape[0], ground_truth.shape[0]))
            G[range(data_inds.shape[0]), data_inds] = 1
            G = da.from_array(G)
        
            data_std = 0.01
            y = G @ ground_truth
        
            # Run data assimilation using an ensemble Kalman filter.
            my_filter = EnsembleKalmanFilter()
    
            # -------------------------
            # All-at-once assimilation.
            # -------------------------
            # Localized version.
            # We have to re-persist the ensembles, since cleaning 
            # of the covariances wipes them from the cluster.
            ensembles_cluster = [client.persist(
                da.from_array(ensemble).astype('float32'))
                    for ensemble in ensembles_local]
            ensembles_cluster = da.stack(ensembles_cluster)

            print("Starting assimilation.")
            mean_updated_aao_loc, ensemble_updated_aao_loc = my_filter.update_ensemble(
                    mean, ensembles_cluster, G, y, data_std, loc_estimated_cov)
            mean_updated_aao_loc, ensemble_updated_aao_loc = (
                    client.persist(mean_updated_aao_loc),
                    client.persist(ensemble_updated_aao_loc))
            progress(ensemble_updated_aao_loc)
            progress(mean_updated_aao_loc)
            wait(ensemble_updated_aao_loc)
            wait(mean_updated_aao_loc)
            print("Finished assimilation.")
            print(mean_updated_aao_loc)
    
            np.save(os.path.join(
                results_folder, "mean_updated_aao_loc_n1d_{}.npy".format(n_pts_1d)),
                mean_updated_aao_loc.compute())
            np.save(os.path.join(
                results_folder, "ensemble_updated_aao_loc_n1d_{}.npy".format(n_pts_1d)),
                ensemble_updated_aao_loc.compute())

    # ----------------
    # End profiling.
    # ----------------
    # Save memory usage dataframe.
    df_memory = ms.to_pandas(align=True)
    df_memory_path = os.path.join(results_folder, "mem_use_df_n1d_{}.pkl".format(n_pts_1d))
    df_memory.to_pickle(df_memory_path)


if __name__ == "__main__":
    main()
