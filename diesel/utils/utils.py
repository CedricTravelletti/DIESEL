"""Helper functions for the DIESEL package."""

import numpy as np
from sklearn.neighbors import BallTree


def build_forward_mean_per_cell(mean_ds, data_ds):
    """Build the forward operator corresponding to a given
    model grid and data point cloud.
    This function only assimilated the mean observed value in each cell.

    Parameters
    ----------
    mean_ds: xr.DataArray
    data_ds: xr.DataArray

    Returns
    -------
    G_mean: (n_data_mean, n_cells)
        Forward operator for assimilation of mean data in each cell.
    mean_datas (n_data_mean)
        Vector of mean observed data in each cell.


    """
    # Get the model cell index corresponding to each observations.
    matched_inds = match_vectors_indices(mean_ds, data_ds)

    # Get unique indices. For the ones that appear several time,
    # we will assimilat the mean. I.e. we assimilat the mean observed data
    # in each cell where we have observations.
    unique_indices = np.unique(matched_inds)
    mean_datas = [np.mean(data_ds.values[matched_inds == i]) for i in unique_indices]
    median_datas = [np.median(data_ds.values[matched_inds == i]) for i in unique_indices]
    std_datas = [np.std(data_ds.values[matched_inds == i]) for i in unique_indices]
    n_datas = [len(data_ds.values[matched_inds == i]) for i in unique_indices]

    mean_datas, median_datas, std_datas, n_datas = (
        np.array(mean_datas),
        np.array(median_datas),
        np.array(std_datas),
        np.array(n_datas),
    )

    data_lats = mean_ds.latitude[unique_indices]
    data_lons = mean_ds.longitude[unique_indices]

    G = np.zeros((unique_indices.shape[0], mean_ds.shape[0]))
    for i in range(unique_indices.shape[0]):
        G[i, unique_indices[i]] = 1.0
    return G, mean_datas, std_datas, median_datas, n_datas, data_lons, data_lats


def match_vectors_indices(base_vector, vector_to_match):
    """ " Given two stacked datasets (vectors), for each element in the dataset_tomatch,
    find the index of the element in the base dataset that is closest.

    Note that the base dataset should contain only one element at each spatial locaiton,
    so that the matched index is unique.

    Parameters
    ----------
    base_vector: xarray.DataArray
        Stacked dataset.
    vector_to_match: xarray.DataArray
        Stacked dataset.

    Returns
    -------
    Array[int] (vector_to_match.shape[0])
        Indices in the base dataset of closest element for each
        element of the dataset_tomatch.

    """
    # Convert to radians.
    lat_rad = np.deg2rad(base_vector.latitude.values.astype(np.float32))
    lon_rad = np.deg2rad(base_vector.longitude.values.astype(np.float32))

    # Build a ball tree to make nearest neighbor queries faster.
    ball = BallTree(np.vstack([lat_rad, lon_rad]).T, metric="haversine")

    # Define grid to be matched.
    lon_tomatch = np.deg2rad(vector_to_match.longitude.values.astype(np.float32))
    lat_tomatch = np.deg2rad(vector_to_match.latitude.values.astype(np.float32))
    coarse_grid_list = np.vstack([lat_tomatch.T, lon_tomatch.T]).T

    distances, index_array_1d = ball.query(coarse_grid_list, k=1)

    # Convert back to kilometers.
    distances_km = 6371 * distances
    # Sanity check.
    print(f"Maximal distance to matched point: {np.max(distances_km)} km.")

    return index_array_1d.squeeze()
