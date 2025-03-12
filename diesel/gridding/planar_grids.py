"""Module for building standard planar grids."""

import dask.array as da
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.family"] = "serif"
plot_params = {
    "font.size": 18,
    "font.style": "normal",
    "axes.labelsize": "x-small",
    "axes.titlesize": "x-small",
    "legend.fontsize": "x-small",
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "x-small",
}
plt.rcParams.update(plot_params)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)


class SquareGrid:
    """Build a 2D regular square grid with n_pts_1d^2 points.
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
        print(f"Builing grid with {cov_size / 1e9} GB covariance matrix.")
        self.X, self.Y = np.meshgrid(
            np.linspace(0, 1, n_pts_1d), np.linspace(0, 1, n_pts_1d), indexing="ij"
        )
        grid_pts = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        grid_pts = np.squeeze(grid_pts)

        grid_pts = da.from_array(grid_pts)
        grid_pts = grid_pts.rechunk(block_size_limit=block_size)
        self.grid_pts = grid_pts

    def mesh_to_list(self, mesh_vals):
        """Flatten 2D meshed values into 1D list.

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

    def plot_vals(
        self,
        vals_list,
        ax,
        points=None,
        points_color="black",
        vmin=None,
        vmax=None,
        fig=None,
        colorbar=False,
        cmap="jet",
    ):
        sns.set()
        sns.set_style("white")
        dx = (self.X[1, 0] - self.X[0, 0]) / 2.0
        dy = (self.Y[0, 1] - self.Y[0, 0]) / 2
        extent = extent = [
            self.X[0, 0] - dx,
            self.X[-1, 0] + dx,
            self.Y[0, -1] + dy,
            self.Y[0, 0] - dy,
        ]

        im = ax.imshow(
            self.list_to_mesh(vals_list).T,
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], c=points_color, s=3, marker="*")
        if colorbar is True:
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
        return ax


def unflatten_to_grid(flat_data, original_grid, active_dims):
    """
    Unflatten a 1D or 2D array back into a multi-dimensional grid using the original grid.

    Args:
        - flat_data: dask.array or numpy.ndarray, the flattened data.
                     Shape should be (n_points,) or (n_points, n_vars).
        - original_grid: xarray.Dataset or xarray.DataArray, the original grid used for stacking.
        - active_dims: list of str, the dimensions that are active in the flattened data.

    Returns:
        - xarray.DataArray or xarray.Dataset, the unflattened grid.
    """
    # Ensure flat_data is a Dask array
    if not isinstance(flat_data, da.Array):
        flat_data = da.from_array(flat_data)

    # Extract coordinate names and original dimensions from the grid
    if isinstance(original_grid, xr.DataArray):
        coord_names = list(original_grid.dims)
        original_dims = {dim: original_grid[dim].values for dim in coord_names}
    elif isinstance(original_grid, xr.Dataset):
        if not original_grid.data_vars:
            # If the dataset has no data variables, use the coordinates directly
            coord_names = list(original_grid.dims)
            original_dims = {dim: original_grid[dim].values for dim in coord_names}
        else:
            # Use the first DataArray in the Dataset to extract dimensions
            first_var = next(iter(original_grid.data_vars.values()))
            coord_names = list(first_var.dims)
            original_dims = {dim: original_grid[dim].values for dim in coord_names}
    else:
        raise ValueError("original_grid must be an xarray.Dataset or xarray.DataArray.")

    # Drop extra dimensions that are not active in the flattened data
    active_dims = list(active_dims)  # Ensure active_dims is a list
    extra_dims = [dim for dim in coord_names if dim not in active_dims]

    # Compute the total number of points in the active dimensions
    n_points = np.prod([len(original_dims[dim]) for dim in active_dims])

    # Ensure the flattened data has the correct shape
    if flat_data.shape[0] != n_points:
        raise ValueError(
            f"flat_data has {flat_data.shape[0]} points, but the active grid has {n_points} points."
        )

    # Reshape the flattened data to match the active grid's shape
    if flat_data.ndim == 1:
        # If flat_data is 1D, reshape it to match the active grid
        reshaped_data = flat_data.reshape([len(original_dims[dim]) for dim in active_dims])
    elif flat_data.ndim == 2:
        # If flat_data is 2D, reshape it to match the active grid + variables
        reshaped_data = flat_data.reshape(
            [len(original_dims[dim]) for dim in active_dims] + [flat_data.shape[1]]
        )
    else:
        raise ValueError("flat_data must be 1D or 2D.")

    # Create a new DataArray with the reshaped data and active coordinates
    if flat_data.ndim == 1:
        unflattened = xr.DataArray(
            reshaped_data,
            dims=active_dims,
            coords={dim: original_dims[dim] for dim in active_dims},
        )
    elif flat_data.ndim == 2:
        # Add a 'variable' dimension for 2D data
        unflattened = xr.DataArray(
            reshaped_data,
            dims=active_dims + ["variable"],
            coords={
                **{dim: original_dims[dim] for dim in active_dims},
                "variable": range(flat_data.shape[1]),
            },
        )

    return unflattened


def flatten_grid(dataset, coord_names):
    """
    Flatten a multi-dimensional grid into a 2D Dask array of points.

    Args:
        - dataset: xarray.Dataset or xarray.DataArray, the input grid.
        - coord_names: list of str, names of the coordinates to flatten.

    Returns:
        - points: dask.array, a 2D array of shape (n_points, n_coords),
                  where each row represents a point in the grid.
    """
    # Stack the specified coordinates into a single dimension
    stacked = dataset.stack(point=coord_names)

    # Extract the flattened coordinates as a list of Dask arrays
    coord_arrays = [stacked[coord].data for coord in coord_names]

    # Stack the Dask arrays vertically and transpose to get shape (n_points, n_coords)
    points = da.vstack(coord_arrays).T

    return points
