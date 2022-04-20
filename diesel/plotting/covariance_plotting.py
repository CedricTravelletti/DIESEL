""" Module for graphic representation of covariance matrices. 

"""
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt


def compute_variogram(dist_mat, cov_mat, n_bins):
    """ compute binned variogram (covariance as function of distance).

    """
    min_dist = dist_mat.min().compute()
    max_dist = dist_mat.max().compute()
    bins = np.linspace(min_dist, max_dist, n_bins)

    bins_midpts = bins[:-1] + (bins[1] - bins[0])/2

    bin_means = []
    bin_stds = []
    for i in range(bins.shape[0] - 1):
        bin_low, bins_high = bins[i], bins[i + 1]

        # Find indices where condition is satisfied.
        inds_i, inds_j = da.where((dist_mat >= bin_low) & (dist_mat < bins_high))

        # Compute mean and std over bin.
        cov_mat_bin = cov_mat.vindex[inds_i, inds_j]
        bin_means.append(cov_mat_bin.mean().compute())
        bin_stds.append(cov_mat_bin.std().compute())

    return bins_midpts, np.array(bin_means), np.array(bin_stds)

def plot_variogram(dist_mat, cov_mat, n_bins, outfile=None):
    bins_midpts, bins_means, bins_stds = compute_variogram(
            dist_mat, cov_mat, n_bins)
    plt.plot(bins_midpts, bins_means)
    plt.fill_between(bins_midpts, bins_means - 3*bins_stds, bins_means + 3*bins_stds, alpha=.2)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.show()

    return bins_midpts, bins_means, bins_stds
