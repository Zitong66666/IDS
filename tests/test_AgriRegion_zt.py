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
# l3_path_pattern= '/projects/bbkc/zitong/Data/S5PNO2/L3/CONUS_%Y_%m.nc' 
l3_path_pattern= '/projects/bbkc/zitong/Data/IASINH3L3_flux02/CONUS_%Y_%m_%d.nc'
nei_dir = '/projects/bbkc/zitong/Data/NEI_gridded'
landmask_filename = '/projects/bbkc/zitong/Data/NLDAS_masks-veg-soil.nc4'

start_dt = dt.datetime(2021,1,1)
end_dt = dt.datetime(2022,10,31)
dt_array = pd.period_range(start_dt,end_dt,freq='1D') # freq align with l3 files
west,east,south,north = -105,-83,37,45
mdwst = AgriRegion(start_dt=start_dt,end_dt=end_dt,
                    west=west,east=east,south=south,north=north)
# load emissions in the region
mdwst.get_region_emission(l3_path_pattern=l3_path_pattern,
                          field='NH3',l3_freq='1D', # field='NO2',l3_freq='1M',
                          masking_kw=dict(
                              nei_dir=nei_dir,
                              landmask_filename=landmask_filename,
                              max_topo_emission=1e-9,
                              max_chem_emission=1e-9,
                            #   max_topo_wind_column=1e-10, # zitong try parameters
                            # # # #   max_topo_windtopo=1e-7,
                            #   min_topo_windtopo=0.003,
                            #   max_chem_wind_column=1e-10,
                            #   max_chem_windtopo=0.003,
                            #   min_chem_column_amount=2e-5
                              ),
                          fit_topo_kw=dict(
                              resample_rule='6M',
                              max_iter=2,
                              outlier_std=2, # zitong add
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf,
                            #   fit_chem=True # zitong add
                              ),
                          fit_chem_kw=dict(
                              resample_rule='6M',#month_of_year
                              max_iter=2,
                              outlier_std=2, # zitong add
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
vmax = 1e-9 # 1e-9 for NO2, 1e-8 for NH3
vlims = [(0,.5e-3),(-vmax,vmax),(-vmax,vmax),(-vmax,vmax),
         (0,1500),(-vmax,vmax),(0,1),(0,1)]

for ax,fld,vlim in zip(axs.ravel(order='F'),flds,vlims):
    figout = mdwst.l3all.plot(fld,vmin=vlim[0],vmax=vlim[1],existing_ax=ax) #.block_reduce(new_grid_size=0.2)
plt.show()
