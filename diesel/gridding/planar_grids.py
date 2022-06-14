""" Module for building standard planar grids.

"""
import numpy as np
import dask.array as da


class SquareGrid:
    """ Build a 2D regular square grid with n_pts_1d^2 points.
    The points are returned as dask chunked arrays.

    Parameters
    ----------
    n_pts_1d: int
        Number of grid points along 1 dimension.
    block_size: int
        Maximal size of the chunks for the chunked array.

    Returns
    -------
    grid_pts: array [n_pts, 2]

    """
    def __init__(self, n_pts_1d, block_size=1e4):
        # Size of corresponding covariance matrix in GB.
        cov_size = 4 * n_pts_1d**4
        print("Builing grid with {} GB covariance matrix.".format(cov_size/1e9))
        self.X, self.Y = np.meshgrid(
                np.linspace(0, 1, n_pts_1d), np.linspace(0, 1, n_pts_1d), indexing='ij')
        grid_pts = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        grid_pts = np.squeeze(grid_pts)
    
        grid_pts = da.from_array(grid_pts)
        grid_pts = grid_pts.rechunk(block_size_limit=block_size)
        self.grid_pts = grid_pts

    def mesh_to_list(self, mesh_vals):
        """ Flatten 2D meshed values into 1D list.

        Parameters
        ----------
        mesh_vals: array[dim_x, dim_y]

        Returns
        -------
        list_vals: array[dim_x * dim_y]

        """
        return mesh_vals.ravel()

    def list_to_mesh(self, list_vals):
        return list_vals.reshape(self.X.shape[0], self.Y.shape[0])
