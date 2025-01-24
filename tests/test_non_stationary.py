import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client
import diesel as ds


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    __builtins__.CLIENT = client
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=60)
    grid_pts = grid.grid_pts

    lengthscales = da.from_array([0.1, 0.4])
    kernel = ds.covariance.matern32(lengthscales)

    cov_mat = kernel.covariance_matrix(grid_pts, grid_pts)

    # Plot covariance to verify everything works.
    fig, ax = plt.subplots()
    grid.plot_vals(cov_mat[1000, :], ax, points=grid_pts[1000].reshape(1, -1))
    plt.show()

    global_lengthscales = da.from_array([1 / np.sqrt(2.44)])
    local_lengthscales = da.from_array([1 / np.sqrt(578.09)])

    dat_pts = np.array([0.93, 0.95,
        1.05,
        1.34, 1.355,
        1.265,
        1.45,
        2.2, 2.3, 2.4, 2.5]).reshape(-1, 1)
    def true_fun(x): np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4
    y = true_fun(dat_pts)

    myGP = ds.BaCompositeGP(
            global_covariance=ds.covariance.matern32(global_lengthscales),
            local_covariance=ds.covariance.matern32(local_lengthscales))

    lmbda = 0.019
    b = 1

    pred_pts = np.linspace(0.5, 2.5, 200).reshape(-1, 1)
    def true_fun(x): np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4
    preds_global, preds_local = myGP.predict(pred_pts, dat_pts, y, lmbda, b)

    plt.plot(pred_pts, true_fun(pred_pts))
    plt.scatter(dat_pts, true_fun(dat_pts))
    plt.plot(pred_pts, preds_global)
    plt.plot(pred_pts, preds_global + preds_local, color="red")
    plt.show()

if __name__ == "__main__":
    main()
