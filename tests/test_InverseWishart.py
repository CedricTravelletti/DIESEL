""" Tests for diesel.estimation.bayesian.InverseWishartPrior

"""
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

# Build a simple inverse wishart prior.
dof = 10
scale_matrix = da.eye(ensembles.shape[1])
prior = ds.estimation.InverseWishartPrior(scale_matrix, dof)

# Compute posterior mean given the data.
lazy_post_cov = prior.posterior_mean(ensembles)
post_cov = client.compute(lazy_post_cov).result()
