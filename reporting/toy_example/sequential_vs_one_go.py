""" Compare the performance of the sequential Ensemble Kalman Filter with 
the version that assimilates all data points in one go.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client
import diesel as ds
from diesel.kalman_filtering import EnsembleKalmanFilter
from diesel.scoring import compute_RE_score, compute_CRPS
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
    
    n_rep = 1
    for rep in range(n_rep):
        print("Repetition {} / {}.".format(rep, n_rep))
        # Sample 30 ensemble members.
        n_ensembles = 30
        ensembles = sampler.sample(n_ensembles + 1) # Note this is still lazy.
    
        # Use the first sample as ground truth.
        ground_truth = ensembles[0]
        ensembles = ensembles[1:]
    
        # Trigger computations.
        ground_truth = client.persist(ground_truth)
        # np.save(os.path.join(results_folder, "ground_truth_{}.npy".format(rep)), ground_truth.compute())
        ensembles = [client.compute(ensemble).result() for ensemble in ensembles]

        # Stack ensembles so are in the format required later.
        ensembles = da.stack(ensembles)
    
        # Estimate covariance using empirical covariance of the ensemble.
        raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)
    
        # Persist the covariance on the cluster.
        raw_estimated_cov = client.persist(raw_estimated_cov_lazy)

        # Perform covariance localization (use scaled version of base covariance to localize).
        # Maybe should persist here.
        scaled_covariance_matrix = kernel.covariance_matrix(grid_pts, grid_pts, 
                lengthscales=da.from_array([10 * lambda0]))
        loc_estimated_cov = localize_covariance(raw_estimated_cov, scaled_covariance_matrix)
    
        # Prepare some data by randomly selecting some points.
        n_data = 20
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

        mean_updated_one_go_raw, ensemble_updated_one_go_raw = my_filter.update_ensemble(mean, ensembles, G, y, data_std, raw_estimated_cov)
        mean_updated_one_go_loc, ensemble_updated_one_go_loc = my_filter.update_ensemble(mean, ensembles, G, y, data_std, loc_estimated_cov)

        localizer_loc = lambda x: localize_covariance(ds.estimation.empirical_covariance(x), scaled_covariance_matrix)
        localizer_raw = lambda x: ds.estimation.empirical_covariance(x)

        mean_updated_seq_raw, ensemble_updated_seq_raw = my_filter.update_ensemble_sequential(mean, ensembles, G, y, data_std, raw_estimated_cov,
                covariance_estimator=localizer_raw
                )
        mean_updated_seq_loc, ensemble_updated_seq_loc = my_filter.update_ensemble_sequential(mean, ensembles, G, y, data_std, raw_estimated_cov,
                covariance_estimator=localizer_loc
                )

        # Compare sequential and one-go.
        fig, axs = plt.subplots(2, 3)
        grid.plot_vals(ground_truth, axs[0, 0], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])

        grid.plot_vals(mean_updated_one_go_raw.compute(), axs[0, 1], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 1].title.set_text('all-at-once (no localization)')
        axs[0, 1].set_xticks([])

        grid.plot_vals(mean_updated_one_go_loc.compute(), axs[0, 2], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 2].title.set_text('all-at-once (localization)')
        axs[0, 2].set_xticks([])

        grid.plot_vals(mean.compute(), axs[1, 0], vmin=-3, vmax=3)
        axs[1, 0].title.set_text('prior mean')

        grid.plot_vals(mean_updated_seq_raw.compute(), axs[1, 1], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[1, 1].title.set_text('sequential (no localization)')

        grid.plot_vals(mean_updated_seq_loc.compute(), axs[1, 2], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[1, 2].title.set_text('sequential (localization)')

        plt.savefig("sequential_vs_one_go", bbox_inches="tight", pad_inches=0.1, dpi=400)
        plt.show()

        # Now compare scores.
        RE_score_one_go_raw = compute_RE_score(mean, mean_updated_one_go_raw, ground_truth)
        RE_score_one_go_loc = compute_RE_score(mean, mean_updated_one_go_loc, ground_truth)
        CRPS_one_go_raw, acc_one_go_raw, spread_one_go_raw = compute_CRPS(
                ensemble_updated_one_go_raw, ground_truth)
        CRPS_one_go_loc, acc_one_go_loc, spread_one_go_loc = compute_CRPS(
                ensemble_updated_one_go_loc, ground_truth)

        # Normalize.
        acc_one_go_raw, acc_one_go_loc = (1 / n_ensembles) * acc_one_go_raw, (1 / n_ensembles) * acc_one_go_loc
        spread_one_go_raw, spread_one_go_loc = (1 / (2 * n_ensembles**2)) * spread_one_go_raw, (1 / (2 * n_ensembles**2)) * spread_one_go_loc

        fig, axs = plt.subplots(3, 3)
        grid.plot_vals(ground_truth, axs[0, 0], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])

        grid.plot_vals(mean_updated_one_go_raw.compute(), axs[0, 1],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 1].title.set_text('all-at-once (no localization)')
        axs[0, 1].set_xticks([])

        grid.plot_vals(mean_updated_one_go_loc.compute(), axs[0, 2],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 2].title.set_text('all-at-once (localization)')
        axs[0, 2].set_xticks([])

        grid.plot_vals(CRPS_one_go_raw.compute(), axs[1, 0], vmin=0, vmax=3)
        axs[1, 0].title.set_text('CRPS raw')

        grid.plot_vals(acc_one_go_raw.compute(), axs[1, 1],
                points=grid_pts[data_inds],
                vmin=0, vmax=2.5,
                points_color='magenta')
        axs[1, 1].title.set_text('acc raw')

        grid.plot_vals(spread_one_go_raw.compute(), axs[1, 2],
                points=grid_pts[data_inds],
                points_color='magenta')
        axs[1, 2].title.set_text('spread raw')

        grid.plot_vals(CRPS_one_go_loc.compute(), axs[2, 0], vmin=0, vmax=3)
        axs[2, 0].title.set_text('CRPS loc')

        grid.plot_vals(acc_one_go_loc.compute(), axs[2, 1],
                points=grid_pts[data_inds],
                vmin=0, vmax=2.5,
                points_color="magenta")
        axs[2, 1].title.set_text('accuray loc')

        grid.plot_vals(spread_one_go_loc.compute(), axs[2, 2],
                points=grid_pts[data_inds],
                points_color='magenta')
        axs[2, 2].title.set_text('spread loc')

        plt.savefig("scores_sequential_vs_one_go", bbox_inches="tight", pad_inches=0.1, dpi=400)
        plt.show()

        # Plot members.
        fig, axs = plt.subplots(3, 4)

        grid.plot_vals(ground_truth, axs[0, 0], points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[0, 0].title.set_text('ground truth')
        axs[0, 0].set_xticks([])
        grid.plot_vals(ensembles[0, :].compute(), axs[0, 1],
                vmin=-3, vmax=3)
        axs[0, 1].title.set_text('ensemble 0')
        axs[0, 1].set_xticks([])
        grid.plot_vals(ensembles[1, :].compute(), axs[0, 2],
                vmin=-3, vmax=3)
        axs[0, 2].title.set_text('ensemble 1')
        axs[0, 2].set_xticks([])
        grid.plot_vals(ensembles[2, :].compute(), axs[0, 3],
                vmin=-3, vmax=3)
        axs[0, 3].title.set_text('ensemble 2')
        axs[0, 3].set_xticks([])

        grid.plot_vals(mean_updated_one_go_raw, axs[1, 0],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[1, 0].title.set_text('mean updated raw')
        axs[1, 0].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_raw[0, :].compute(), axs[1, 1],
                vmin=-3, vmax=3)
        axs[1, 1].title.set_text('ensemble 0 raw')
        axs[1, 1].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_raw[1, :].compute(), axs[1, 2],
                vmin=-3, vmax=3)
        axs[1, 2].title.set_text('ensemble 1 raw')
        axs[1, 2].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_raw[2, :].compute(), axs[1, 3],
                vmin=-3, vmax=3)
        axs[1, 3].title.set_text('ensemble 2 raw')
        axs[1, 3].set_xticks([])

        grid.plot_vals(mean_updated_one_go_loc, axs[2, 0],
                points=grid_pts[data_inds], vmin=-3, vmax=3)
        axs[2, 0].title.set_text('mean updated loc')
        axs[2, 0].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_loc[0, :].compute(), axs[2, 1],
                vmin=-3, vmax=3)
        axs[2, 1].title.set_text('ensemble 0 loc')
        axs[2, 1].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_loc[1, :].compute(), axs[2, 2],
                vmin=-3, vmax=3)
        axs[2, 2].title.set_text('ensemble 1 loc')
        axs[2, 2].set_xticks([])
        grid.plot_vals(ensemble_updated_one_go_loc[2, :].compute(), axs[2, 3],
                vmin=-3, vmax=3)
        axs[2, 3].title.set_text('ensemble 2 loc')
        axs[2, 3].set_xticks([])

        plt.savefig("ensembles_sequential_vs_one_go", bbox_inches="tight", pad_inches=0.1, dpi=400)
        plt.show()


if __name__ == "__main__":
    main()
