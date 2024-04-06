"""
Created on Tue Mar 26 10:07:32 2024

@author: zitong
"""

import sys,os,glob
sys.path.append('/projects/bbkc/zitong/Oversampling_matlab')
from popy import Level3_List
sys.path.append('/projects/bbkc/zitong/IDS.py')
from IDS import AgriRegion
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%% download nei, only need once
from IDS import Inventory
# nei = Inventory()
# nei.download_NEI_ag('/home/kangsun/data/NEI')
# your_path = '/projects/bbkc/zitong/Data/NEI_gridded'
# nei_filenames = [os.path.join(your_path,f) for f in os.listdir(your_path) if os.path.isfile(os.path.join(your_path, f))]
# nei.read_NEI_ag(nei_filenames)

#%%
# l3 data downloaded from https://uofi.app.box.com/s/wqmsg4wd605g0vk467czrcjvtw4m88ts
# l3_path_pattern = '/home/kangsun/data/IASINH3/L3/IASINH3L3_flux01/CONUS_%Y_%m_%d.nc'
l3_path_pattern= '/projects/bbkc/zitong/Data/IASINH3L3_flux01/CONUS_%Y_%m_%d.nc'  # flux004：flux_grid_size=0.04
nei_dir = '/projects/bbkc/zitong/Data/NEI_gridded'
landmask_dir = '/projects/bbkc/zitong/Data/NLDAS_masks-veg-soil.nc4'
start_dt = dt.datetime(2020,1,1)
end_dt = dt.datetime(2021,12,31)
# west,east,south,north = -125,-115,37,50
# west,east,south,north = -125,-65,24,50 # CONUS
# west,east,south,north = -109,-101,36,42 # CO
# west,east,south,north = -125,-116,35,44 # CA
west,east,south,north = -105,-83,37,45 # midwest
# create a region with lat/lon bounds
mdwst = AgriRegion(start_dt=start_dt,end_dt=end_dt,
                    west=west,east=east,south=south,north=north)
# load emissions in the region
mdwst.get_region_emission(l3_path_pattern=l3_path_pattern,
                          masking_kw=dict(
                              nei_dir=nei_dir,
                              landmask_dir=landmask_dir,
                              max_topo_emission=1e-9,
                              max_chem_emission=1e-9,
                            #   max_topo_wind_column=2e-7, # zitong try parameters
                            #   max_topo_windtopo=1e-7,
                            #   min_topo_windtopo=5e-9,
                            #   max_chem_wind_column=5e-10,
                            #   max_chem_windtopo=5e-5,
                            #   min_chem_column_amount=2.5e-5
                              ),
                          fit_topo_kw=dict(
                              resample_rule='1M',
                              max_iter=1,
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf
                              ),
                          fit_chem_kw=dict(
                              resample_rule='1M',
                              max_iter=1,
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf,
                              max_wind_column=np.inf))

# %%
plt.close('all')
fig,axs = plt.subplots(4,2,sharex=True,sharey=True,constrained_layout=True,
                       subplot_kw=dict(projection=ccrs.PlateCarree()),
                       figsize=(15,10))
flds = ['column_amount','wind_column','wind_column_topo','wind_column_topo_chem',
        'surface_altitude','NEI','topo_mask','chem_mask']
vlims = [(0,.5e-3),(-1e-8,1e-8),(-1e-8,1e-8),(-1e-8,1e-8),
         (0,1500),(-1e-8,1e-8),(0,1),(0,1)]

for ax,fld,vlim in zip(axs.ravel(order='F'),flds,vlims):
    figout = mdwst.l3all.plot(fld,vmin=vlim[0],vmax=vlim[1],existing_ax=ax) #.block_reduce(new_grid_size=0.2)
plt.savefig('a_flux004_2020-2021_midwest2_moore.png')


# %%
plt.close('all')
months = pd.period_range(start=start_dt, end=end_dt, freq='D')
timestamps = months.to_timestamp()
fig, axes = plt.subplots(2,2,figsize=(15, 5))
axes[0,0].plot(range(0,len(mdwst.l3s.df['topo_scale_height']),1),mdwst.l3s.df['topo_scale_height'],marker='.',markersize=10) 
axes[0,0].set_ylabel('Scale height (m)')
axes[0,1].plot(range(0,len(mdwst.l3s.df['chem_lifetime']),1),mdwst.l3s.df['chem_lifetime'],marker='.',markersize=10)
axes[0,1].set_ylabel('Lifetime (h)')
axes[1,0].plot(range(0,len(mdwst.l3s.df['topo_r2']),1),mdwst.l3s.df['topo_r2'],marker='.',markersize=10)
axes[1,0].set_ylabel('topo_r2')
axes[1,0].set_ylim(0,0.8)
axes[1,1].plot(range(0,len(mdwst.l3s.df['chem_r2']),1),mdwst.l3s.df['chem_r2'],marker='.',markersize=10)
axes[1,1].set_ylabel('chem_r2')
axes[1,1].set_ylim(0,0.8)
plt.tight_layout()
plt.savefig('a_flux004_2020-2021_midwest2_moore_X_T_r2.png')