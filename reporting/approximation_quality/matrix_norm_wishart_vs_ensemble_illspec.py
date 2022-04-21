""" Compare covariance matrix reconstruction between a bayesian approach 
(inverse wishart prior) and a purely empirical estimate.

This one considers an ill-specified case for the prior, where we have the wrong lenght scale.

Script will plot Frobenius norm of error for both approaches.

"""
import numpy as np
import pandas as pd
import dask.array as da
from dask.distributed import Client
import diesel as ds


def main():
    cluster = ds.cluster.LocalCluster()
    client = Client(cluster)
    
    # Build a square grid with 30^2 elements.
    grid = ds.gridding.SquareGrid(30)
    grid_pts = grid.grid_pts
    
    # Construct (lazy) covariance matrix.
    dim = grid_pts.shape[0]
    lazy_covariance_matrix = ds.covariance.matern32(grid_pts, lambda0=0.2)
    
    # Compute compressed SVD.
    svd_rank = 900
    u, s, v = da.linalg.svd_compressed(
                    lazy_covariance_matrix, k=svd_rank, compute=False) 
    
    # Construct sampler from the svd of the covariance matrix.
    sampler = ds.sampling.SvdSampler(u, s)
    
    # Ill-specified prior.
    scale_matrix = ds.covariance.matern32(grid_pts, lambda0=1.0)

    results= pd.DataFrame(columns=['Degrees of Freedom','Repetition', 
        'Error (empirical)', 'Error (bayesian)'])
    
    # Replicate analysis 50 times.
    n_reps = 50
    dofs = np.linspace(900, 1500, 20)
    for dof in dofs:
        # Create inverse Wishart prior.
        scale_factor = dof - dim - 1 # Scale so that the mean is always equal to the scale matrix. 
        prior = ds.estimation.InverseWishartPrior(scale_factor * scale_matrix, dof)

        for rep in range(n_reps):
            print("repetition: {}".format(rep))
            # Sample ensemble.
            ensembles = sampler.sample(20)
            ensembles = client.compute(ensembles).result()

            # Estimate covariance using both approaches
            lazy_empirical_cov = ds.estimation.empirical_covariance(ensembles)
            lazy_bayesian_cov = prior.posterior_mean(ensembles)
    
            # Compute distance in Frobenius norm between true covariance and estimated covariance.
            dist_empirical = da.linalg.norm(lazy_covariance_matrix - lazy_empirical_cov, ord='fro')
            error_empirical = client.compute(dist_empirical).result()
            dist_bayesian = da.linalg.norm(lazy_covariance_matrix - lazy_bayesian_cov, ord='fro')
            error_bayesian = client.compute(dist_bayesian).result()
    
            results = results.append({'Degrees of Freedom': dof,
                            'Repetition': rep,
                            'Error (empirical)': error_empirical,
                            'Error (bayesian)': error_bayesian,
                            }, ignore_index=True)
    
    # Save at the end.
    results.to_pickle("error_empirical_vs_bayesian_results_ill_spec.pkl")
    
    # Plot results.
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.set_style("white")
    plt.rcParams["font.family"] = "Times New Roman"
    plot_params = {
            'font.size': 16, 'font.style': 'oblique',
            'axes.labelsize': 'small',
            'axes.titlesize':'small',
            'legend.fontsize': 'small'
            }
    plt.rcParams.update(plot_params)
    
    fig, ax = plt.subplots(figsize=(8,6))
    fig.set_size_inches(6, 6)
    
    my_palette = sns.color_palette("RdBu", 6)
    my_palette = my_palette[0:2] + [my_palette[-1]]

    mean_empirical_error = results['Error (empirical)'].mean()
    std_empirical_error = results['Error (empirical)'].std()
    
    ax = sns.lineplot('Degrees of Freedom', 'Error (bayesian)', data=results, 
            palette=my_palette)
    ax.axhline(mean_empirical_error, color='r')
    ax.fill_between(results['Degrees of Freedom'], mean_empirical_error - 2*std_empirical_error, mean_empirical_error + 2*std_empirical_error,
            color='r', alpha=.2)
    
    plt.savefig("error_empirical_vs_bayesian_ill_spec", bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.show()

if __name__ == "__main__":
    main()
