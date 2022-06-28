""" Module for building standard planar grids.

"""
import numpy as np
import dask.array as da
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    def plot_vals(self, vals_list, ax, points=None, vmin=None, vmax=None,
            fig=None, colorbar=False):
        dx = (self.X[1, 0]-self.X[0, 0])/2.
        dy = (self.Y[0, 1]-self.Y[0, 0])/2
        extent = extent = [
                self.X[0, 0]-dx, self.X[-1, 0]+dx,
                self.Y[0, -1]+dy, self.Y[0, 0]-dy]

        im = ax.imshow(self.list_to_mesh(vals_list).T,
                cmap='jet', extent=extent,
                vmin=vmin, vmax=vmax)

        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], c='black', s=10, marker='*')
        if colorbar is True:
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        return ax
