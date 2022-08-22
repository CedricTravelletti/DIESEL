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


results_folder ="/home/cedric/PHD/Dev/DIESEL/reporting/toy_example/ubelix_results/"


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)

    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=120)
    grid_pts = grid.grid_pts

    ground_truth = da.from_array(
            np.load(os.path.join(results_folder, "ground_truth_0.npy")))
    ensemble_updated_one_go_raw = da.from_array(
            np.load(os.path.join(results_folder, "ensemble_updated_one_go_raw_0.npy")))
    ensemble_updated_one_go_loc = da.from_array(
            np.load(os.path.join(results_folder, "ensemble_updated_one_go_loc_0.npy")))
    ensemble_updated_seq_raw = da.from_array(
            np.load(os.path.join(results_folder, "ensemble_updated_seq_raw_49.npy")))
    ensemble_updated_seq_loc = da.from_array(
            np.load(os.path.join(results_folder, "ensemble_updated_seq_loc_49.npy")))

    # Now compare scores.
    # RE_score_one_go_raw = compute_RE_score(mean, mean_updated_one_go_raw, ground_truth)
    # RE_score_one_go_loc = compute_RE_score(mean, mean_updated_one_go_loc, ground_truth)

    CRPS_one_go_raw, misfit_one_go_raw, spread_one_go_raw = compute_CRPS(
                ensemble_updated_one_go_raw, ground_truth)
    CRPS_one_go_loc, misfit_one_go_loc, spread_one_go_loc = compute_CRPS(
            ensemble_updated_one_go_loc, ground_truth)

    CRPS_seq_raw, misfit_seq_raw, spread_seq_raw = compute_CRPS(
                ensemble_updated_seq_raw, ground_truth)
    CRPS_seq_loc, misfit_seq_loc, spread_seq_loc = compute_CRPS(
            ensemble_updated_seq_loc, ground_truth)

    ES_one_go_raw, ES_misfit_one_go_raw, ES_spread_one_go_raw = compute_energy_score(
            ensemble_updated_one_go_raw, ground_truth)
    ES_one_go_loc, ES_misfit_one_go_loc, ES_spread_one_go_loc = compute_energy_score(
            ensemble_updated_one_go_loc, ground_truth)

    ES_seq_raw, ES_misfit_seq_raw, ES_spread_seq_raw = compute_energy_score(
            ensemble_updated_seq_raw, ground_truth)
    ES_seq_loc, ES_misfit_seq_loc, ES_spread_seq_loc = compute_energy_score(
            ensemble_updated_seq_loc, ground_truth)


    fig, axs = plt.subplots(3, 3)
    grid.plot_vals(ground_truth, axs[0, 0], vmin=-3, vmax=3)
    axs[0, 0].title.set_text('ground truth')
    axs[0, 0].set_xticks([])

    grid.plot_vals(ensemble_updated_one_go_raw.mean(axis=0).compute(), axs[0, 1],
            vmin=-3, vmax=3)
    axs[0, 1].title.set_text('all-at-once (no localization)')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    grid.plot_vals(ensemble_updated_one_go_loc.mean(axis=0).compute(), axs[0, 2],
            vmin=-3, vmax=3)
    axs[0, 2].title.set_text('all-at-once (localization)')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    grid.plot_vals(CRPS_one_go_loc.compute(), axs[1, 0], vmin=0, vmax=3)
    axs[1, 0].title.set_text('CRPS all-at-once (ES: {})'.format(ES_one_go_loc.compute()))
    axs[1, 0].set_xticks([])

    grid.plot_vals(misfit_one_go_loc.compute(), axs[1, 1],
            vmin=0, vmax=2.5,
            )
    axs[1, 1].title.set_text('misfit all-at-once')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_xticks([])

    grid.plot_vals(spread_one_go_loc.compute(), axs[1, 2],
            )
    axs[1, 2].title.set_text('spread all-at-once')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    grid.plot_vals(CRPS_seq_loc.compute(), axs[2, 0], vmin=0, vmax=3)
    axs[2, 0].title.set_text('CRPS sequential (ES: {})'.format(ES_seq_loc.compute()))
    axs[2, 0].set_xticks([])

    grid.plot_vals(misfit_seq_loc.compute(), axs[2, 1],
            vmin=0, vmax=2.5,
            )
    axs[2, 1].title.set_text('misfit sequential')
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    grid.plot_vals(spread_seq_loc.compute(), axs[2, 2],
            )
    axs[2, 2].title.set_text('spread sequential')
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])

    plt.savefig("scores_sequential_vs_one_go_bigdata_2",
        bbox_inches="tight", pad_inches=0.1, dpi=400)
    # plt.show()


if __name__ == "__main__":
    main()
