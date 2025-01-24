""" Plot variogram.

"""
from dask.distributed import Client

import diesel as ds


def main():
    cluster = ds.LocalCluster()
    client = Client(cluster)
     # Add to builtins so we have one global client.
    __builtins__.CLIENT = client
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(30)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lazy_dist_matrix = ds.covariance.distance_matrix(grid_pts)
    lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.2)

    variog_bins, variog_means, variog_stds = ds.plotting.plot_variogram(
            lazy_dist_matrix, lazy_covariance_matrix, 20)

if __name__ == "__main__":
    main()
