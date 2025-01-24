""" Tests for diesel.sampling.SvdSampler

"""
import dask.array as da
from dask.distributed import Client
from diesel.gridding import SquareGrid
from diesel.cluster import LocalCluster
from diesel.covariance import matern32
from diesel.sampling import SvdSampler
import matplotlib.pyplot as plt


cluster = LocalCluster()
client = Client(cluster)

# Build a square grid with 30^2 elements.
grid = SquareGrid(30)
grid_pts = grid.grid_pts

# Construct (lazy) covariance matrix.
lazy_covariance_matrix = matern32(grid_pts, lambda0=0.2)

# Compute compressed SVD.
svd_rank = 900
u, s, v = da.linalg.svd_compressed(
                lazy_covariance_matrix, k=svd_rank, compute=False) 

# Construct sampler from the svd of the covariance matrix.
sampler = SvdSampler(u, s)

# Sample 16 ensemble members.
ensembles = sampler.sample(16)

ensembles = client.compute(ensembles).result()

# Plot results
fig, axs = plt.subplots(4, 4)

for i, sample in enumerate(ensembles):
    axs.flatten()[i].imshow(grid.list_to_mesh(sample))

plt.show()
