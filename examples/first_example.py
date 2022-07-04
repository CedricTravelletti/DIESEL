import dask.array as da
from dask.distributed import Client
import diesel as ds
import matplotlib.pyplot as plt


def main():
    # Instantiate a local cluster, to mimick distributed computations, but on a single machine.
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(n_pts_1d=30)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.1)
    
    # Compute compressed SVD.
    svd_rank = 900 # Since our matrix is 900 * 900 this will be a full SVD.
    u, s, v = da.linalg.svd_compressed(
                    lazy_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)
    
    # Sample 16 ensemble members.
    ensembles = sampler.sample(16) # Note this is still lazy.

    # Plot one ensemble.
    plt.imshow(grid.list_to_mesh(ensembles[0]), cmap='jet')
    plt.show()
    
    # Estimate covariance using empirical covariance of the ensemble.
    estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)
    
    # Compute distance in Frobenius norm between true covariance and estimated covariance.
    dist = da.linalg.norm(lazy_covariance_matrix - estimated_cov_lazy, ord='fro')
    dist = client.compute(dist).result()
    print("Frobenius distance between true covariance matrix and sample covariance: {}.".format(dist))


if __name__ == "__main__":
    main()
