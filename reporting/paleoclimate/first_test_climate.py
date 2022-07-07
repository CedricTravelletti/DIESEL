""" Try to run DIESEL on climate data (Valler, Franke et al).

"""
import numpy as np
import dask.array as da
import diesel as ds
from dask.distributed import Client, wait, progress
from climate.kalman_filter import EnsembleKalmanFilterScatter
from climate.utils import load_dataset


data_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
results_folder = "/storage/homefs/ct19x463/Dev/DIESEL/reporting/paleoclimate/results/"
# data_folder = "/home/cedric/PHD/Dev/Climate/Data/"
# results_folder = "/home/cedric/PHD/Dev/DIESEL/reporting/paleoclimate/results/"



 cluster = ds.cluster.UbelixCluster(n_nodes=8, mem_per_node=24, cores_per_node=4,
         partition="gpu", qos="job_gpu")
# cluster = ds.cluster.LocalCluster()
client = Client(cluster)

# The loading function returns 4 datasets: the ensemble members, the ensemble
# mean, the instrumental data and the reference dataset.
TOT_ENSEMBLES_NUMBER = 30
(dataset_mean, dataset_members,
        dataset_instrumental, dataset_reference,
        dataset_members_zarr)= load_dataset(
        data_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)
dataset_instrumental = dataset_instrumental.chunk()
print("Loading done.")

# Extract one window vector.
# Use a helper Kalman filter for simplicity.
helper_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr,
        dataset_instrumental, client)
time_begin, time_end = '1961-01-16', '1961-06-16'
n_months = 6
# First get the mean vector and data vector (stacked for the window).
mean = helper_filter.dataset_mean.get_window_vector(time_begin, time_end)
ensemble = helper_filter.dataset_members.get_window_vector(time_begin, time_end)
y = helper_filter.dataset_instrumental.get_window_vector(time_begin, time_end)

# Get rid of the Nans.
print(y.shape)
y = y[np.logical_not(np.isnan(y))]
print("After removing NaNs")
print(y.shape)

# Get forward.
G = helper_filter.get_forward_for_window(time_begin, time_end, n_months)
G = da.from_array(G)
G = client.persist(G)
print(G.shape)


# Estimate covariance using empirical covariance of the ensemble.
raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)
    
# Persist the covariance on the cluster.
raw_estimated_cov = client.persist(raw_estimated_cov_lazy)

data_std = np.sqrt(0.9)
mean_updated_one_go_raw, ensemble_updated_one_go_raw = my_filter.update_ensemble(
        mean, ensemble, G, y, data_std, raw_estimated_cov)
