#!/usr/bin/env python
# coding: utf-8

# # Assimilate GLSD data with DIESEL for 20th century.
# 
# This notebook runs assimilation of GLSD data using the DIESEL version of the Ensemble Kalman filter. It compares sequential and all-at-once assimilation on the whole 20th century.

# In[1]:
import os
import numpy as np
import dask
import pandas as pd
import dask.array as da
import xarray as xr
from climate.utils import load_dataset, match_vectors_indices


from dask.distributed import Client, wait, progress                             
import diesel as ds                                                             
from diesel.scoring import compute_RE_score, compute_CRPS, compute_energy_score, compute_RMSE
from diesel.estimation import localize_covariance 


# In[2]:


n_data = 1200

base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
results_folder = "/storage/homefs/ct19x463/Dev/DIESEL/reporting/paleoclimate/results/twentieth_century/n{}/".format(n_data)


# ## Build Cluster

# In[3]:


cluster = ds.cluster.UbelixCluster(n_nodes=12, mem_per_node=64, cores_per_node=3,
            partition="gpu", qos="job_gpu")                                     
cluster.scale(18)                                                           
client = Client(cluster)                                                    
                                                                                
# Add to builtins so we have one global client.
# Note that this is necessary before importing the EnsembleKalmanFilter module, so that the module is aware of the cluster.
__builtins__.CLIENT = client                                                


# In[4]:


from diesel.kalman_filtering import EnsembleKalmanFilter 
from dask.diagnostics import ProgressBar
ProgressBar().register()


# In[5]:


cluster


# In[6]:


TOT_ENSEMBLES_NUMBER = 30
(dataset_mean, dataset_members,
    dataset_instrumental, dataset_reference,
    dataset_members_zarr)= load_dataset(
    base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)
print("Loading done.")


# In[7]:


from climate.kalman_filter import EnsembleKalmanFilterScatter
helper_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr, dataset_instrumental, client)


# In[8]:


my_filter = EnsembleKalmanFilter()                                      
data_std = 0.1


# ## Run Assimilation.

# In[9]:


# Construct localization matrix.                                      
lambda0 = 1500 # Localization in kilometers.
lengthscales = da.from_array([lambda0])   
kernel = ds.covariance.squared_exponential(lengthscales)
    
# Build localization matrix.
mean_dummy = helper_filter.dataset_mean.get_window_vector('1961-01-16', '1961-01-16', variable='temperature') # Dummy, just to get the grid.

grid_pts = da.vstack([mean_dummy.latitude, mean_dummy.longitude]).T
grid_pts = client.persist(grid_pts.rechunk((1800, 2)))
localization_matrix = kernel.covariance_matrix(grid_pts, grid_pts, metric='haversine') 
localization_matrix = client.persist(localization_matrix)
progress(localization_matrix)


# In[ ]:


ES_prior, ES_aao_loc, ES_seq_loc = [], [], []        
RE_aao_loc, RE_seq_loc = [], []                       
RMSE_prior, RMSE_aao_loc, RMSE_seq_loc = [], [], []

dates, months, years = [], [], []


# Loop over years.
for year in range(1902, 2000):
## Loop over months.
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        # Prepare vectors.
        assimilation_date = '{}-{}-16'.format(year, month)
        mean_ds = helper_filter.dataset_mean.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
        ensemble_ds = helper_filter.dataset_members.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
    
        mean_ds, ensemble_ds = client.persist(mean_ds), client.persist(ensemble_ds)
        
        """
        # Load data.
        data_df = pd.read_csv(os.path.join(base_folder, "Instrumental/GLSD/yearly_csv/temperature_{}.csv".format(year)), index_col=0)
        data_ds = xr.Dataset.from_dataframe(data_df)

        # Rename the date variable and make latitude/longitude into coordinates.
        data_ds = data_ds.rename({'date': 'time'})
        data_ds = data_ds.set_coords(['time', 'latitude', 'longitude'])
        data_ds = data_ds['temperature']
    
        # Prepare forward.
        date= '{}-{}-01'.format(year, month)
        data_month_ds = data_ds.where(data_ds.time==date, drop=True)

        # Need to clean data since dataset contains erroneous measurements, i.e. 
        # either extreme values (10^30) or values that are exactly zero for a given station across time.
        data_month_ds = data_month_ds.where((data_month_ds > -100.0) & (data_month_ds < 100.0) & (da.abs(data_month_ds) > 0.0001), drop=True)
        data_vector = client.persist(da.from_array(data_month_ds.data))
        """
        # TODO: Here there is a change. We instead try to assimilate a randomly chosen subset of the reference as data.
        date = assimilation_date
        ref = dataset_reference.temperature.sel(time=assimilation_date)
        stacked_ref = ref.stack( stacked_dim=('latitude', 'longitude'))
        data_ref = stacked_ref.values
        data_ref_lat = stacked_ref.latitude.values
        data_ref_lon = stacked_ref.longitude.values
        # Get rid of NaN's.
        data_ref_lat = data_ref_lat[~np.isnan(data_ref)]
        data_ref_lon = data_ref_lon[~np.isnan(data_ref)]
        data_ref = data_ref[~np.isnan(data_ref)]
        # Select a random subset.
        data_inds = np.random.choice(data_ref.shape[0], n_data, replace=False)
        np.save(os.path.join(results_folder, "data_inds_{}_n{}.npy".format(date, n_data)), data_inds)
        data = data_ref[data_inds]
        data_lat = data_ref_lat[data_inds]
        data_lon = data_ref_lon[data_inds]
        # Put into a dataframe.
        data_df = pd.DataFrame({'temperature': data, 'latitude': data_lat, 'longitude': data_lon})
        data_ds = xr.Dataset.from_dataframe(data_df)
        data_ds = xr.Dataset.from_dataframe(data_df)
        data_month_ds = data_ds.set_coords(['latitude', 'longitude'])['temperature']
        data_vector = client.persist(da.from_array(data_month_ds.data))
        
    
        # TODO: here back to traditional.
        # Get the model cell index corresponding to each observations.
        matched_inds = match_vectors_indices(mean_ds, data_month_ds)

        # WARNING: Never try to execute bare loops in DASK, it will exceed the maximal graph depth.
        G = np.zeros((data_month_ds.shape[0], mean_ds.shape[0]))
        for obs_nr, model_cell_ind in enumerate(matched_inds):
            G[obs_nr, model_cell_ind] = 1.0

        G = da.from_array(G)
        G = client.persist(G)
    
        # Estimate covariance.
        raw_estimated_cov_lazy = ds.estimation.empirical_covariance(ensemble_ds.chunk((1, 1800)))                                                                               
        # Persist the covariance on the cluster.                                
        raw_estimated_cov = client.persist(raw_estimated_cov_lazy) 
        progress(raw_estimated_cov)
    
        # Localize covariance.
        loc_estimated_cov = localize_covariance(raw_estimated_cov, localization_matrix)
        loc_estimated_cov = client.persist(loc_estimated_cov)
        progress(loc_estimated_cov)
    
        # Assimilate all-at-once.
        # -----------------------
        mean_updated_aao_loc, ensemble_updated_aao_loc = my_filter.update_ensemble(
            mean_ds.data, ensemble_ds.data, G,
            data_vector, data_std, loc_estimated_cov)

        # Trigger computations and block. Otherwise will clutter the scheduler. 
        mean_updated_aao_loc = client.persist(mean_updated_aao_loc)                
        ensemble_updated_aao_loc = client.persist(ensemble_updated_aao_loc)
        progress(ensemble_updated_aao_loc) # Block till end of computations.        
    
        # Save data.
        np.save(os.path.join(results_folder, "mean_updated_aao_loc_{}_n{}.npy".format(date, n_data)),
            mean_updated_aao_loc.compute())
        np.save(os.path.join(results_folder, "ensemble_updated_aao_loc_{}_n{}.npy".format(date, n_data)),
            ensemble_updated_aao_loc.compute())
        
        # Assimilate sequential.
        # ----------------------
        mean_updated_seq_loc, ensemble_updated_seq_loc = my_filter.update_ensemble_sequential_nondask(
            mean_ds.data, ensemble_ds.data, G,
            data_vector, data_std, localization_matrix)
    
        # Save data.
        np.save(os.path.join(results_folder, "mean_updated_seq_loc_{}_n{}.npy".format(date, n_data)),
            mean_updated_seq_loc)
        np.save(os.path.join(results_folder, "ensemble_updated_seq_loc_{}_n{}.npy".format(date, n_data)),
            ensemble_updated_seq_loc)
        
        # Compute scores. 
        # Before computing, have to put into unstacked form.
        unstacked_updated_mean_aao_loc = helper_filter.dataset_mean.unstack_window_vector(mean_updated_aao_loc.compute(), time=assimilation_date, variable_name='temperature')
        unstacked_updated_mean_seq_loc = helper_filter.dataset_mean.unstack_window_vector(mean_updated_seq_loc, time=assimilation_date, variable_name='temperature')
        # Clip to common extent, since reference does not contain the sea.
        ref = dataset_reference.temperature.sel(time=assimilation_date)
        unstacked_updated_mean_seq_loc = unstacked_updated_mean_aao_loc.where(
            xr.ufuncs.logical_not(xr.ufuncs.isnan(ref)))    
        unstacked_updated_mean_seq_loc = unstacked_updated_mean_aao_loc.where(
            xr.ufuncs.logical_not(xr.ufuncs.isnan(ref)))    

        stacked_ref = ref.stack( stacked_dim=('latitude', 'longitude'))
        ES, _, _ = compute_energy_score(ensemble_ds.compute().values, stacked_ref.data)
        ES_prior.append(ES)                                                     
                                                                                
        ES, _, _ = compute_energy_score(ensemble_updated_aao_loc.compute(), stacked_ref.data)
        ES_aao_loc.append(ES)                                                   
                                                                                
        ES, _, _ = compute_energy_score(ensemble_updated_seq_loc, stacked_ref.data)
        ES_seq_loc.append(ES)                                                   
                                                                                                                                                                                                          
        RE = np.median(compute_RE_score(mean_ds.data, mean_updated_aao_loc.compute(), stacked_ref.data).compute())
        RE_aao_loc.append(RE)                                                   
                                                                                
        RE = np.median(compute_RE_score(mean_ds.data, mean_updated_seq_loc, stacked_ref.data).compute())
        RE_seq_loc.append(RE)                                                                                       
                                                                                
        RMSE_prior.append(compute_RMSE(mean_ds.values, stacked_ref.values))
        RMSE_aao_loc.append(compute_RMSE(mean_updated_aao_loc.compute(), stacked_ref.values))
        RMSE_seq_loc.append(compute_RMSE(mean_updated_seq_loc, stacked_ref.values))
        
        dates.append(date), months.append(month), years.append(year)
                                                                                
        df_results = pd.DataFrame({  
            'date': dates, 'year': years, 'month': months,
            'RMSE prior': RMSE_prior, 'RMSE aao loc': RMSE_aao_loc, 'RMSE seq loc': RMSE_seq_loc,
            'ES prior': ES_prior, 'ES aao loc': ES_aao_loc, 'ES seq loc': ES_seq_loc,
            'RE aao loc': RE_aao_loc, 'RE seq loc': RE_seq_loc})
        df_results.to_pickle(os.path.join(results_folder, 'scores_n{}.pkl'.format(n_data)))        
