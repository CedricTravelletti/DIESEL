""" Run 20th century assimilation, but with station data from CRUTEM dataset this time.

"""
import os
import numpy as np
import dask
import pandas as pd
import dask.array as da
import xarray as xr
from climate.utils import load_dataset, match_vectors_indices
from climate.data_wrapper import StationDataset


from dask.distributed import Client, wait, progress                             
import diesel as ds                                                             
from diesel.scoring import compute_RE_score, compute_CRPS, compute_energy_score, compute_RMSE
from diesel.estimation import localize_covariance 
from diesel.utils import build_forward_mean_per_cell




base_folder = "/storage/homefs/ct19x463/Dev/Climate/Data/"
results_folder = "/storage/homefs/ct19x463/Dev/DIESEL/reporting/paleoclimate/results/twentieth_century/stations/"


# Build Cluster
cluster = ds.cluster.UbelixCluster(n_nodes=12, mem_per_node=64, cores_per_node=3,
            partition="gpu", qos="job_gpu")                                     
cluster.scale(18)                                                           
client = Client(cluster)                                                    
                                                                                
# Add to builtins so we have one global client.
# Note that this is necessary before importing the EnsembleKalmanFilter module, so that the module is aware of the cluster.
__builtins__.CLIENT = client                                                


from diesel.kalman_filtering import EnsembleKalmanFilter 
from dask.diagnostics import ProgressBar
ProgressBar().register()

TOT_ENSEMBLES_NUMBER = 30
(dataset_mean, dataset_members,
    dataset_instrumental, dataset_reference,
    dataset_members_zarr)= load_dataset(
    base_folder, TOT_ENSEMBLES_NUMBER, ignore_members=True)

stationDataset = StationDataset(base_folder)
print("Loading done.")

from climate.kalman_filter import EnsembleKalmanFilterScatter
helper_filter = EnsembleKalmanFilterScatter(dataset_mean, dataset_members_zarr, dataset_instrumental, client)

my_filter = EnsembleKalmanFilter()                                      
data_std = 0.1


# ## Run Assimilation.

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
for year in range(1990, 2000):
## Loop over months.
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        # Prepare vectors.
        assimilation_date = '{}-{}-16'.format(year, month)
        mean_ds = helper_filter.dataset_mean.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
        ensemble_ds = helper_filter.dataset_members.get_window_vector(assimilation_date, assimilation_date, variable='temperature')
    
        mean_ds, ensemble_ds = client.persist(mean_ds), client.persist(ensemble_ds)

        # Get anomaly.
        anomaly = helper_filter.dataset_mean.get_window_vector(assimilation_date, assimilation_date, variable='anomaly')
        climatology = mean_ds - anomaly

        ensemble_anomaly = ensemble_ds.data - climatology.data.reshape(-1)[None, :]
        
        # Load data.
        data = stationDataset.get_station_data(year, month, "16")
        data_df = pd.DataFrame(data, columns = ['temperature', 'climatology','latitude','longitude'])
        data_ds = xr.Dataset.from_dataframe(data_df)

        # Rename the date variable and make latitude/longitude into coordinates.
        data_ds = data_ds.set_coords(['latitude', 'longitude'])
    
       # data_month_ds = data_month_ds.where((data_month_ds > -100.0) & (data_month_ds < 100.0) & (da.abs(data_month_ds) > 0.0001), drop=True)
        data_ds['anomaly'] = (data_ds['temperature'] - data_ds['climatology'])
        
        # Build cell-averaged forward.
        G_mean, d_mean, d_lons, d_lats = build_forward_mean_per_cell(mean_ds, data_ds['anomaly'])
        G_mean = client.persist(da.from_array(G_mean))
        d_mean = client.persist(da.from_array(d_mean))
    
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
            anomaly.data, ensemble_anomaly, G_mean,
            d_mean, data_std, loc_estimated_cov)

        # Trigger computations and block. Otherwise will clutter the scheduler. 
        mean_updated_aao_loc = client.persist(mean_updated_aao_loc)                
        ensemble_updated_aao_loc = client.persist(ensemble_updated_aao_loc)
        progress(ensemble_updated_aao_loc) # Block till end of computations.        
    
        # Save data.
        np.save(os.path.join(results_folder, "mean_updated_aao_loc_{}.npy".format(assimilation_date)),
            mean_updated_aao_loc.compute())
        np.save(os.path.join(results_folder, "ensemble_updated_aao_loc_{}.npy".format(assimilation_date)),
            ensemble_updated_aao_loc.compute())
        
        # Assimilate sequential.
        # ----------------------
        mean_updated_seq_loc, ensemble_updated_seq_loc = my_filter.update_ensemble_sequential_nondask(
                mean_ds.data, ensemble_ds.data, G_mean,
                d_mean, data_std, localization_matrix)
    
        # Save data.
        np.save(os.path.join(results_folder, "mean_updated_seq_loc_{}.npy".format(assimilation_date)),
            mean_updated_seq_loc)
        np.save(os.path.join(results_folder, "ensemble_updated_seq_loc_{}.npy".format(assimilation_date)),
                ensemble_updated_seq_loc)
        
        # Compute scores. 
        # Before computing, have to put into unstacked form.
        unstacked_updated_mean_aao_loc = helper_filter.dataset_mean.unstack_window_vector(mean_updated_aao_loc.compute(), time=assimilation_date, variable_name='temperature')
        unstacked_updated_mean_seq_loc = helper_filter.dataset_mean.unstack_window_vector(mean_updated_seq_loc, time=assimilation_date, variable_name='temperature')
        unstacked_updated_ensemble_aao_loc = helper_filter.dataset_members.unstack_window_vector(ensemble_updated_aao_loc.compute(), time=assimilation_date, variable_name='temperature')
        unstacked_updated_ensemble_seq_loc = helper_filter.dataset_members.unstack_window_vector(ensemble_updated_seq_loc, time=assimilation_date, variable_name='temperature')
        unstacked_prior = helper_filter.dataset_mean.unstack_window_vector(anomaly.values, time=assimilation_date, variable_name='temperature')
        unstacked_prior_ens = helper_filter.dataset_members.unstack_window_vector(ensemble_anomaly.compute(), time=assimilation_date, variable_name='temperature')

        # Load HadCRUT reference
        ref_ds = xr.open_dataset(os.path.join(base_folder, "Reference/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc"))
        if month == '02':
            ref_date = '{}-{}-15'.format(year, month)
        else: ref_date = assimilation_date
        ref = ref_ds['tas_mean'].sel(time=ref_date)

        # Regrid to common extent.
        # Note that it was found out (see cornell_Nov_8_diagnose_stations.py) that regridding to a coarser grid (that of the reference), 
        # for comparison, lead to poor performances. The postulated reason for the discrepancy is that a coarse grid cell would contain 
        # too many highly different datapoints during assimilation.
        #
        # Hence, we instead regrid the reference to the finer (assimilation) grid.
        regridded_ref = ref.isel(time=0).interp(
            latitude=unstacked_updated_mean_aao_loc.latitude).interp(
            longitude=unstacked_updated_mean_aao_loc.longitude)
        stacked_ref = regridded_ref.stack(stacked_dim=('latitude', 'longitude')).compute()

        """
        regridded_prior = unstacked_prior.interp(latitude=ref.latitude).interp(longitude=ref.longitude)
        regridded_prior_ens = unstacked_prior_ens.interp(latitude=ref.latitude).interp(longitude=ref.longitude)
        regridded_mean_updated_aao_loc = unstacked_updated_mean_aao_loc.interp(latitude=ref.latitude).interp(longitude=ref.longitude)
        regridded_mean_updated_seq_loc = unstacked_updated_mean_seq_loc.interp(latitude=ref.latitude).interp(longitude=ref.longitude)
        regridded_ensemble_updated_aao_loc = unstacked_updated_ensemble_aao_loc.interp(latitude=ref.latitude).interp(longitude=ref.longitude)
        regridded_ensemble_updated_seq_loc = unstacked_updated_ensemble_seq_loc.interp(latitude=ref.latitude).interp(longitude=ref.longitude)

        # Now restack.
        stacked_ref = ref.stack(stacked_dim=('latitude', 'longitude')).isel(time=0).compute()
        stacked_prior = regridded_prior.stack(stacked_dim=('latitude', 'longitude')).compute()
        stacked_prior_ens = regridded_prior_ens.stack(stacked_dim=('latitude', 'longitude')).compute()
        stacked_mean_updated_aao_loc = regridded_mean_updated_aao_loc.stack(stacked_dim=('latitude', 'longitude')).compute()
        stacked_mean_updated_seq_loc = regridded_mean_updated_seq_loc.stack(stacked_dim=('latitude', 'longitude')).compute()
        stacked_ensemble_updated_aao_loc = regridded_ensemble_updated_aao_loc.stack(stacked_dim=('latitude', 'longitude')).compute()
        stacked_ensemble_updated_seq_loc = regridded_ensemble_updated_seq_loc.stack(stacked_dim=('latitude', 'longitude')).compute()
        """
        stacked_prior = anomaly.values
        stacked_prior_ens = ensemble_anomaly.compute()
        stacked_mean_updated_aao_loc = mean_updated_aao_loc.compute()
        stacked_mean_updated_seq_loc = mean_updated_seq_loc
        stacked_ensemble_updated_aao_loc = ensemble_updated_aao_loc.compute()
        stacked_ensemble_updated_seq_loc = ensemble_updated_seq_loc

        ES, _, _ = compute_energy_score(stacked_prior_ens, stacked_ref, min_lat=-70.0, max_lat=70.0)
        ES_prior.append(ES)                                                     
                                                                                
        ES, _, _ = compute_energy_score(stacked_ensemble_updated_aao_loc, stacked_ref, min_lat=-70.0, max_lat=70.0)
        ES_aao_loc.append(ES)                                                   
                                                                                
        ES, _, _ = compute_energy_score(stacked_ensemble_updated_seq_loc, stacked_ref, min_lat=-70.0, max_lat=70.0)
        ES_seq_loc.append(ES)                                                   
           
        RE_score_map = compute_RE_score(stacked_prior, stacked_mean_updated_aao_loc, stacked_ref,
                min_lat=-70, max_lat=70)
        RE = np.median(RE_score_map)
        RE_aao_loc.append(RE)                                                   
                                                                                
        RE_score_map = compute_RE_score(stacked_prior, stacked_mean_updated_seq_loc, stacked_ref,
            min_lat=-70, max_lat=70)
        RE = np.median(RE_score_map)
        RE_seq_loc.append(RE)                                                                                       

        RMSE_prior.append(compute_RMSE(stacked_prior, stacked_ref, min_lat=-70.0, max_lat=70.0))
        RMSE_aao_loc.append(compute_RMSE(stacked_mean_updated_aao_loc, stacked_ref, min_lat=-70.0, max_lat=70.0))
        RMSE_seq_loc.append(compute_RMSE(stacked_mean_updated_seq_loc, stacked_ref, min_lat=-70.0, max_lat=70.0))
        
        dates.append(assimilation_date), months.append(month), years.append(year)
                                                                                
        df_results = pd.DataFrame({  
                'date': dates, 'year': years, 'month': months,
                'RMSE prior': RMSE_prior, 'RMSE aao loc': RMSE_aao_loc, 'RMSE seq loc': RMSE_seq_loc,
                'ES prior': ES_prior, 'ES aao loc': ES_aao_loc, 'ES seq loc': ES_seq_loc,
                'RE aao loc': RE_aao_loc, 'RE seq loc': RE_seq_loc})
        df_results.to_pickle(os.path.join(results_folder, 'scores.pkl'))
