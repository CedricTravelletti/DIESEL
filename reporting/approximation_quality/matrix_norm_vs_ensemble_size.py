""" Study covariance matrix reconstruction quality as a function of the ensemble size. 

Script will plot Frobenius norm of error matrix as a function of the ensemble size.

"""
def main():
    import numpy as np
    import pandas as pd
    import dask.array as da
    from dask.distributed import Client
    from diesel.gridding import SquareGrid
    from diesel.cluster import LocalCluster
    from diesel.covariance import matern32
    from diesel.sampling import SvdSampler
    import diesel.estimation


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
    
    
    results= pd.DataFrame(columns=['Ensemble size','Repetition', 'Error (Frobenius norm)'])
    
    n_reps = 50
    sizes = np.arange(10, 1000, 50)
    sizes = np.concatenate(sizes, [1500, 2000, 3000, 10000])
    for ens_size in sizes:
        for rep in range(n_reps):
            print("repetition: {}".format(rep))
            # Sample ensemble.
            ensembles = sampler.sample(ens_size)
            ensembles = client.compute(ensembles).result()
    
            # Estimate covariance using empirical covariance of the ensemble.
            estimated_cov_lazy = diesel.estimation.empirical_covariance(ensembles)
    
            # Compute distance in Frobenius norm between true covariance and estimated covariance.
            dist = da.linalg.norm(lazy_covariance_matrix - estimated_cov_lazy, ord='fro')
            error = client.compute(dist).result()
    
            results = results.append({'Ensemble size': ens_size,
                            'repetition': rep,
                            'Error (Frobenius norm)': error
                            }, ignore_index=True)
    
    # Save at the end.
    results.to_pickle("error_vs_ens_size_results.pkl")
    
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
    
    ax = sns.lineplot('Ensemble size', 'Error (Frobenius norm)', data=results, 
            palette=my_palette)
    
    plt.savefig("error_vs_ens_size", bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.show()

if __name__ == "__main__":
    main()
