import dask.array as da
from dask.distributed import Client
import diesel as ds


cluster = ds.cluster.LocalCluster()
client = Client(cluster)

# Build a square grid with 30^2 elements.
grid = ds.gridding.SquareGrid(30)
grid_pts = grid.grid_pts

# Construct (lazy) covariance matrix.
lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.2)

# Compute compressed SVD.
svd_rank = 900
u, s, v = da.linalg.svd_compressed(
                lazy_covariance_matrix, k=svd_rank, compute=False) 

# Construct sampler from the svd of the covariance matrix.
sampler = ds.sampling.SvdSampler(u, s)

# Sample 16 ensemble members.
ensembles = sampler.sample(16)

ensembles = client.compute(ensembles).result()

# Estimate covariance using empirical covariance of the ensemble.
estimated_cov_lazy = ds.estimation.empirical_covariance(ensembles)

# Compute distance in Frobenius norm between true covariance and estimated covariance.
dist = da.linalg.norm(lazy_covariance_matrix - estimated_cov_lazy, ord='fro')
dist = client.compute(dist).result()
