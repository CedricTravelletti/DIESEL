""" Compare the performance of the sequential Ensemble Kalman Filter with 
the version that assimilates all data points in one go.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, wait, progress
import diesel as ds
from diesel.scoring import compute_RE_score, compute_CRPS, compute_energy_score
from diesel.estimation import localize_covariance


import time


# results_folder ="/home/cedric/PHD/Dev/DIESEL/reporting/toy_example/results/"
results_folder ="/storage/homefs/ct19x463/Dev/DIESEL/reporting/toy_example/results/"


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    # cluster = ds.cluster.LocalCluster()
    cluster = ds.cluster.UbelixCluster(n_nodes=12, mem_per_node=32, cores_per_node=2,
            partition="gpu", qos="job_gpu")
    cluster.scale(12)
    client = Client(cluster)

    # Add to builtins so we have one global client.
    __builtins__.CLIENT = client
    from diesel.kalman_filtering import EnsembleKalmanFilter
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=120)
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
    
    n_rep = 1
    for rep in range(n_rep):
        print("Repetition {} / {}.".format(rep, n_rep))
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
    
        # Persist the covariance on the cluster.
        raw_estimated_cov = client.persist(raw_estimated_cov_lazy)

        # Perform covariance localization (use scaled version of base covariance to localize).
        # Maybe should persist here.
        scaled_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts, 
                lengthscales=da.from_array([10 * lambda0]))
        loc_estimated_cov = localize_covariance(raw_estimated_cov, scaled_covariance_matrix)
        loc_estimated_cov = client.persist(loc_estimated_cov)
    
        # Prepare some data by randomly selecting some points.
        n_data = 500
        data_inds = np.random.choice(ground_truth.shape[0], n_data, replace=False)  
    
        #  Built observation operator.
        G = np.zeros((data_inds.shape[0], ground_truth.shape[0]))
        G[range(data_inds.shape[0]), data_inds] = 1
        G = da.from_array(G)
    
        data_std = 0.01
        y = G @ ground_truth
    
        # Run data assimilation using an ensemble Kalman filter.
        my_filter = EnsembleKalmanFilter()

        mean_updated_one_go_raw, ensemble_updated_one_go_raw = my_filter.update_ensemble(mean, ensembles, G, y, data_std, raw_estimated_cov)

        # Trigger computations and block. Otherwise will clutter the scheduler.
        mean_updated_one_go_raw, ensemble_updated_one_go_raw = (
                client.persist(mean_updated_one_go_raw),
                client.persist(ensemble_updated_one_go_raw))
        progress(ensemble_updated_one_go_raw)

        np.save(os.path.join(
            results_folder, "mean_updated_one_go_raw_{}.npy".format(rep)),
            mean_updated_one_go_raw.compute())
        np.save(os.path.join(
            results_folder, "ensemble_updated_one_go_raw_{}.npy".format(rep)),
            ensemble_updated_one_go_raw.compute())

        mean_updated_one_go_loc, ensemble_updated_one_go_loc = my_filter.update_ensemble(
                mean, ensembles, G, y, data_std, loc_estimated_cov)

        # Trigger computations and block. Otherwise will clutter the scheduler.
        mean_updated_one_go_loc, ensemble_updated_one_go_loc = (
                client.persist(mean_updated_one_go_loc),
                client.persist(ensemble_updated_one_go_loc))
        progress(ensemble_updated_one_go_loc)

        np.save(os.path.join(
            results_folder, "mean_updated_one_go_loc_{}.npy".format(rep)),
            mean_updated_one_go_loc.compute())
        np.save(os.path.join(
            results_folder, "ensemble_updated_one_go_loc_{}.npy".format(rep)),
            ensemble_updated_one_go_loc.compute())


        localizer_loc = lambda x: localize_covariance(ds.estimation.empirical_covariance(x), scaled_covariance_matrix)
        localizer_raw = lambda x: ds.estimation.empirical_covariance(x)

        # Stupid, but have to transfer to local node 
        # in order to preserve from cancelling.
        mean_store = mean.compute()
        ensemble_store = ensembles.compute()

        chunk_size = 1
        mean_updated_seq_raw = client.persist(da.from_array(mean_store))
        ensemble_updated_seq_raw = client.persist(da.from_array(ensemble_store))
        running_estimated_cov = raw_estimated_cov
        # for i in range(0, G.shape[0], chunk_size):
        for i in range(0, 50, chunk_size):
            print(i)
            running_estimated_cov_lazy = ds.estimation.empirical_covariance(ensemble_updated_seq_raw)
            running_estimated_cov = client.persist(running_estimated_cov_lazy)
            print("Estimating cov.")
            progress(running_estimated_cov)

            G_chunk = G[i:i+chunk_size].reshape(chunk_size, -1)
            y_chunk = y[i:i+chunk_size].reshape(chunk_size, -1)

            mean_updated_seq_raw, ensemble_updated_seq_raw = my_filter.update_ensemble(
                        mean_updated_seq_raw, ensemble_updated_seq_raw,
                        G_chunk, y_chunk, data_std, running_estimated_cov
                )
            mean_updated_seq_raw = client.persist(mean_updated_seq_raw)
            ensemble_updated_seq_raw = client.persist(ensemble_updated_seq_raw)
            print("Start progress.")
            progress(ensemble_updated_seq_raw)
            print("End progress.")
            ensemble_tmp = ensemble_updated_seq_raw.compute()
            mean_tmp = mean_updated_seq_raw.compute()
            client.cancel(ensemble_updated_seq_raw)
            client.cancel(mean_updated_seq_raw)
            client.cancel(running_estimated_cov_lazy)
            client.cancel(running_estimated_cov)
            mean_updated_seq_raw = client.persist(da.from_array(mean_tmp))
            ensemble_updated_seq_raw = client.persist(da.from_array(ensemble_tmp))

            np.save(os.path.join(
                results_folder, "mean_updated_seq_raw_{}.npy".format(i)),
                mean_updated_seq_raw.compute())
            np.save(os.path.join(
                results_folder, "ensemble_updated_seq_raw_{}.npy".format(i)),
                ensemble_updated_seq_raw.compute())

        print(ensembles.compute())
        mean_updated_seq_loc = client.persist(da.from_array(mean_store))
        ensemble_updated_seq_loc = client.persist(da.from_array(ensemble_store))
        running_estimated_cov = raw_estimated_cov
        # for i in range(0, G.shape[0], chunk_size):
        for i in range(0, 50, chunk_size):
            print(i)
            running_estimated_cov_lazy = localizer_loc(ensemble_updated_seq_loc)
            running_estimated_cov = client.persist(running_estimated_cov_lazy)
            print("Estimating cov.")
            progress(running_estimated_cov)

            G_chunk = G[i:i+chunk_size].reshape(chunk_size, -1)
            y_chunk = y[i:i+chunk_size].reshape(chunk_size, -1)

            mean_updated_seq_loc, ensemble_updated_seq_loc = my_filter.update_ensemble(
                        mean_updated_seq_loc, ensemble_updated_seq_loc,
                        G_chunk, y_chunk, data_std, running_estimated_cov
                )
            mean_updated_seq_loc = client.persist(mean_updated_seq_loc)
            ensemble_updated_seq_loc = client.persist(ensemble_updated_seq_loc)
            print("Start progress.")
            progress(ensemble_updated_seq_loc)
            print("End progress.")
            ensemble_tmp = ensemble_updated_seq_loc.compute()
            mean_tmp = mean_updated_seq_loc.compute()
            client.cancel(ensemble_updated_seq_loc)
            client.cancel(mean_updated_seq_loc)
            client.cancel(running_estimated_cov_lazy)
            client.cancel(running_estimated_cov)
            mean_updated_seq_loc = client.persist(da.from_array(mean_tmp))
            ensemble_updated_seq_loc = client.persist(da.from_array(ensemble_tmp))

            np.save(os.path.join(
                results_folder, "mean_updated_seq_loc_{}.npy".format(i)),
                mean_updated_seq_loc.compute())
            np.save(os.path.join(
                results_folder, "ensemble_updated_seq_loc_{}.npy".format(i)),
                ensemble_updated_seq_loc.compute())

        # Compare sequential and one-go.
        fig, axs = plt.subplots(2, 3)
        grid.plot_vals(ground_truth, axs[0, 0], 
                vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])

        grid.plot_vals(client.compute(mean_updated_one_go_raw).result(), axs[0, 1],
                vmin=-3, vmax=3)
        axs[0, 1].title.set_text('all-at-once (no localization)')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        grid.plot_vals(client.compute(mean_updated_one_go_loc).result(), axs[0, 2],
                vmin=-3, vmax=3)
        axs[0, 2].title.set_text('all-at-once (localization)')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        grid.plot_vals(client.compute(mean).result(), axs[1, 0], vmin=-3, vmax=3)
        axs[1, 0].title.set_text('prior mean')
        axs[1, 0].set_xticks([])

        grid.plot_vals(client.compute(mean_updated_seq_raw).result(), axs[1, 1],
                vmin=-3, vmax=3)
        axs[1, 1].title.set_text('sequential (no localization)')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        grid.plot_vals(mean_updated_seq_loc.compute(), axs[1, 2],
                vmin=-3, vmax=3)
        axs[1, 2].title.set_text('sequential (localization)')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        plt.savefig("sequential_vs_one_go_bigdata", bbox_inches="tight", pad_inches=0.1, dpi=400)
        # plt.show()

        # Now compare scores.
        RE_score_one_go_raw = compute_RE_score(mean, mean_updated_one_go_raw, ground_truth)
        RE_score_one_go_loc = compute_RE_score(mean, mean_updated_one_go_loc, ground_truth)
        CRPS_one_go_raw, misfit_one_go_raw, spread_one_go_raw = compute_CRPS(
                ensemble_updated_one_go_raw, ground_truth)
        CRPS_one_go_loc, misfit_one_go_loc, spread_one_go_loc = compute_CRPS(
                ensemble_updated_one_go_loc, ground_truth)
        ES_one_go_raw, ES_misfit_one_go_raw, ES_spread_one_go_raw = compute_energy_score(
                ensemble_updated_one_go_raw, ground_truth)
        ES_one_go_loc, ES_misfit_one_go_loc, ES_spread_one_go_loc = compute_energy_score(
                ensemble_updated_one_go_loc, ground_truth)


        fig, axs = plt.subplots(3, 3)
        grid.plot_vals(ground_truth, axs[0, 0], vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])

        grid.plot_vals(mean_updated_one_go_raw.compute(), axs[0, 1],
                vmin=-3, vmax=3)
        axs[0, 1].title.set_text('all-at-once (no localization)')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        grid.plot_vals(mean_updated_one_go_loc.compute(), axs[0, 2],
                vmin=-3, vmax=3)
        axs[0, 2].title.set_text('all-at-once (localization)')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        grid.plot_vals(CRPS_one_go_raw.compute(), axs[1, 0], vmin=0, vmax=3)
        axs[1, 0].title.set_text('CRPS raw (ES: {})'.format(ES_one_go_raw.compute()))
        axs[1, 0].set_xticks([])

        grid.plot_vals(misfit_one_go_raw.compute(), axs[1, 1],
                points=grid_pts[data_inds],
                vmin=0, vmax=2.5,
                points_color='magenta')
        axs[1, 1].title.set_text('misfit raw')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_xticks([])

        grid.plot_vals(spread_one_go_raw.compute(), axs[1, 2],
                points=grid_pts[data_inds],
                points_color='magenta')
        axs[1, 2].title.set_text('spread raw')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        grid.plot_vals(CRPS_one_go_loc.compute(), axs[2, 0], vmin=0, vmax=3)
        axs[2, 0].title.set_text('CRPS loc (ES: {})'.format(ES_one_go_loc.compute()))
        axs[2, 0].set_xticks([])

        grid.plot_vals(misfit_one_go_loc.compute(), axs[2, 1],
                points=grid_pts[data_inds],
                vmin=0, vmax=2.5,
                points_color="magenta")
        axs[2, 1].title.set_text('misfit loc')
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])

        grid.plot_vals(spread_one_go_loc.compute(), axs[2, 2],
                points=grid_pts[data_inds],
                points_color='magenta')
        axs[2, 2].title.set_text('spread loc')
        axs[2, 2].set_xticks([])
        axs[2, 2].set_yticks([])

        plt.savefig("scores_sequential_vs_one_go_bigdata",
            bbox_inches="tight", pad_inches=0.1, dpi=400)
        # plt.show()

        # Plot members.
        fig, axs = plt.subplots(3, 4)

        grid.plot_vals(ground_truth, axs[0, 0], vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])

        grid.plot_vals(ensembles[0, :].compute(), axs[0, 1],
                vmin=-3, vmax=3)
        axs[0, 1].title.set_text('ensemble 0')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        grid.plot_vals(ensembles[1, :].compute(), axs[0, 2],
                vmin=-3, vmax=3)
        axs[0, 2].title.set_text('ensemble 1')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        grid.plot_vals(ensembles[2, :].compute(), axs[0, 3],
                vmin=-3, vmax=3)
        axs[0, 3].title.set_text('ensemble 2')
        axs[0, 3].set_xticks([])
        axs[0, 3].set_yticks([])

        grid.plot_vals(mean_updated_one_go_raw, axs[1, 0],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[1, 0].title.set_text('mean updated raw')
        axs[1, 0].set_xticks([])

        grid.plot_vals(ensemble_updated_one_go_raw[0, :].compute(), axs[1, 1],
                vmin=-3, vmax=3)
        axs[1, 1].title.set_text('ensemble 0 raw')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        grid.plot_vals(ensemble_updated_one_go_raw[1, :].compute(), axs[1, 2],
                vmin=-3, vmax=3)
        axs[1, 2].title.set_text('ensemble 1 raw')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        grid.plot_vals(ensemble_updated_one_go_raw[2, :].compute(), axs[1, 3],
                vmin=-3, vmax=3)
        axs[1, 3].title.set_text('ensemble 2 raw')
        axs[1, 3].set_xticks([])
        axs[1, 3].set_yticks([])

        grid.plot_vals(mean_updated_one_go_loc, axs[2, 0],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[2, 0].title.set_text('mean updated loc')
        axs[2, 0].set_xticks([])

        grid.plot_vals(ensemble_updated_one_go_loc[0, :].compute(), axs[2, 1],
                vmin=-3, vmax=3)
        axs[2, 1].title.set_text('ensemble 0 loc')
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])

        grid.plot_vals(ensemble_updated_one_go_loc[1, :].compute(), axs[2, 2],
                vmin=-3, vmax=3)
        axs[2, 2].title.set_text('ensemble 1 loc')
        axs[2, 2].set_xticks([])
        axs[2, 2].set_yticks([])

        grid.plot_vals(ensemble_updated_one_go_loc[2, :].compute(), axs[2, 3],
                vmin=-3, vmax=3)
        axs[2, 3].title.set_text('ensemble 2 loc')
        axs[2, 3].set_xticks([])
        axs[2, 3].set_yticks([])

        plt.savefig("ensembles_sequential_vs_one_go_bigdata",
            bbox_inches="tight", pad_inches=0.1, dpi=400)
        # plt.show()


if __name__ == "__main__":
    main()
