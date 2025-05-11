"""
Get ammonia emissions from Level3 data for a region
"""

import pandas as pd
import os,sys,glob
import logging
from netCDF4 import Dataset
sys.path.append('/projects/bbkc/zitong/Oversampling_matlab')
try:
    from popy import (popy,
                      datetime2datenum,
                      F_center2edge,
                      datedev_py,
                      Level3_Data, 
                      Level3_List)
except:
    logging.error('clone https://github.com/Kang-Sun-CfA/Oversampling_matlab.git and add to your path!')
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import kendalltau
from astropy.convolution import convolve_fft
import statsmodels.formula.api as smf
from shapely.ops import unary_union
import shapely
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes

class Inventory(dict):
    '''class based on dict, representing a gridded emission inventory'''
    def __init__(self,name='inventory',west=-180,east=180,south=-90,north=90):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.west = west
        self.east = east
        self.south = south
        self.north = north
    
    def download_NEI_ag(self,nei_dir):
        '''download nei ag from http://geoschemdata.wustl.edu/ExtData/HEMCO/NEI2016/v2021-06/
        to nei_dir
        '''
        cwd = os.getcwd()
        if not os.path.exists(nei_dir):
            self.logger.warning(f'making dir {nei_dir}')
            os.makedirs(nei_dir)
        os.chdir(nei_dir)
        for imonth in range(1,13):
            url = 'http://geoschemdata.wustl.edu/ExtData/HEMCO/NEI2016/v2021-06/'
            murl = '{}/2016fh_16j_merge_0pt1degree_month_{:02d}.ncf'.format(url,imonth)
            self.logger.info(f'downloading {murl}')
            os.system('wget -N {}'.format(murl))
        os.chdir(cwd)


    
    def read_NEI_ag(self,monthly_filenames=None,nei_dir=None,field='NH3',unit='mol/m2/s'):
        '''read a list of monthly NEI inventory files
        monthly_filenames:
            a list of file paths
        nei_dir:
            if provided, supersedes monthly_filenames
        field:
            data field to read from the nc file
        unit:
            emission will be converted from kg/m2/s (double check) to this unit
        '''
        monthly_filenames = monthly_filenames or\
            glob.glob(os.path.join(nei_dir,'2016fh_16j_merge_0pt1degree_month_*.ncf'))
        
        # sort by month
        # Function to extract the month number from the filename

        def extract_month(filename):
            # Assuming the filename contains 'month_MM' where MM is the month (e.g., 01 for January, 12 for December)
            basename = os.path.basename(filename)
            month_part = basename.split('_')[5]  # Adjust index based on filename structure
            month_part = month_part[0:2]
            return int(month_part)  # Convert month to integer for sorting    

        # Sort the filenames based on extracted month
        monthly_filenames_sorted = sorted(monthly_filenames, key=extract_month)


        mons = []
        for i,filename in enumerate(monthly_filenames_sorted):
            print(filename)
            nc = Dataset(filename)
            if i == 0:
                monthly_fields = np.zeros((nc.dimensions['lat'].size,
                                           nc.dimensions['lon'].size,
                                           len(monthly_filenames_sorted)))
                xgrid = nc['lon'][:].data
                ygrid = nc['lat'][:].data
                xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
                ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
                self.xgrid_size = xgrid_size
                self.ygrid_size = ygrid_size
                if not np.isclose(xgrid_size,ygrid_size,rtol=1e-03):
                    self.logger.warning(f'x grid size {xgrid_size} does not equal to y grid size {ygrid_size}')
                self.grid_size = (xgrid_size+ygrid_size)/2
                xmask = (xgrid >= self.west) & (xgrid <= self.east)
                ymask = (ygrid >= self.south) & (ygrid <= self.north)
                self['xgrid'] = xgrid[xmask]
                self['ygrid'] = ygrid[ymask]
                self.west = self['xgrid'].min()-self.grid_size
                self.east = self['xgrid'].max()+self.grid_size
                self.south = self['ygrid'].min()-self.grid_size
                self.north = self['ygrid'].max()+self.grid_size
                xmesh,ymesh = np.meshgrid(self['xgrid'],self['ygrid'])
                self['grid_size_in_m2'] = np.cos(np.deg2rad(ymesh/180*np.pi))*np.square(self.grid_size*111e3)
                nc_unit = nc[field].units
                if nc_unit == 'kg/m2/s' and unit=='nmol/m2/s' and field in ['NH3','NH3_FERT','NO2']:
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1e9/0.017
                elif nc_unit == 'kg/m2/s' and unit=='mol/m2/s' and field in ['NH3','NH3_FERT','NO2']:
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1/0.017
                else:
                    self.logger.info('no unit conversion is done')
                    self[f'{field} unit'] = nc_unit
                    unit_factor = 1.
            monthly_fields[:,:,i] = unit_factor*nc[field][:].filled(np.nan)[0,0,:,:]# time and lev are singular dimensions
            mons.append(dt.datetime(int(str(nc.SDATE)[0:4]),1,1)+dt.timedelta(days=-1+int(str(nc.SDATE)[-3:])))
            nc.close()
        self[field] = monthly_fields
        self['mons'] = pd.to_datetime(mons).to_period('1M')
        self['data'] = np.mean(monthly_fields, axis=2) # mean emission over months, named "data" to match basin_emissions.py

        return self
    
    def read_regrid_landmask(self,l3,landmask_filename=None,field='CONUS_mask'): # zitong add
        '''
        remove the water body
        read and regrid land mask to match the mesh of a l3 data object
        landmask -- 1:land, 0:water/background
        '''
        landmask_filename = landmask_filename or '/projects/bbkc/zitong/Data/NLDAS_masks-veg-soil.nc4'        
        nc = Dataset(landmask_filename)
        xgrid = nc['lon'][:].data
        ygrid = nc['lat'][:].data
        xgrid_size = np.abs(np.nanmedian(np.diff(xgrid)))
        ygrid_size = np.abs(np.nanmedian(np.diff(ygrid)))
        grid_size = (xgrid_size+ygrid_size)/2
        ymesh,xmesh = np.meshgrid(l3['ygrid'],l3['xgrid'])

        if grid_size < l3.grid_size/2:
            method = 'drop_in_the_box'
        else:
            method = 'interpolate'

        if method in ['interpolate']:
            f = RegularGridInterpolator((nc['lat'][:].data,nc['lon'][:].data),np.squeeze(nc[field][:]),method='nearest',bounds_error=False)
            landmask = f((ymesh,xmesh)).T
        elif method in ['drop_in_the_box']:
            data = np.full((len(l3['ygrid']),len(l3['xgrid'])),np.nan)
            for iy,y in enumerate(l3['ygrid']):
                ymask = (self['ygrid']>=y-l3.grid_size/2) & (self['ygrid']<y+l3.grid_size/2)
                for ix,x in enumerate(l3['xgrid']):
                    xmask = (self['xgrid']>=x-l3.grid_size/2) & (self['xgrid']<x+l3.grid_size/2)
                    if np.sum(ymask) == 0 and np.sum(xmask) == 0:
                        continue
                    data[iy,ix] = np.nanmean(nc['data'][np.ix_(ymask,xmask)])
            landmask = data

        self['landmask'] = landmask  
        return self


    
    def regrid(self,l3,fields_to_copy=None,method=None):
        '''regrid inventory to match the mesh of a l3 data object
        fields_to_copy:
            list of fields to copy from l3 to the output Inventory object
        method:
            if none, choose from drop_in_the_box and interpolate based on the relative grid size of inventory and l3
        '''
        
        if fields_to_copy is None:
            fields_to_copy = ['column_amount','wind_topo','surface_altitude']
        if method is None:
            if self.grid_size < l3.grid_size/2:
                method = 'drop_in_the_box'
#             elif (self.grid_size >= l3.grid_size/2) and (self.grid_size < l3.grid_size*2):
#                 method = 'tessellate'
            else:
                method = 'interpolate'
            self.logger.warning(f'regridding from {self.grid_size} to {l3.grid_size} using {method}')
        
        inv = Inventory(name=self.name)
        inv['xgrid'] = l3['xgrid']
        inv['ygrid'] = l3['ygrid']
        inv['NH3'] = np.full((len(inv['ygrid']),len(inv['xgrid']),12),np.nan)
        ymesh,xmesh = np.meshgrid(inv['ygrid'],inv['xgrid'])
        inv.grid_size = l3.grid_size
        inv.west = inv['xgrid'].min()-inv.grid_size
        inv.east = inv['xgrid'].max()+inv.grid_size
        inv.south = inv['ygrid'].min()-inv.grid_size
        inv.north = inv['ygrid'].max()+inv.grid_size
        if method in ['interpolate']:            
            f = RegularGridInterpolator((self['ygrid'],self['xgrid']),self['data'],bounds_error=False)
            inv['data'] = f((ymesh,xmesh)).T
            for i in range(12):
                datam = self['NH3'][:,:,i]
                f = RegularGridInterpolator((self['ygrid'],self['xgrid']),datam,bounds_error=False)
                inv['NH3'][:,:,i] = f((ymesh,xmesh)).T

        elif method in ['drop_in_the_box']:
            data = np.full((len(inv['ygrid']),len(inv['xgrid'])),np.nan)            
            for iy,y in enumerate(inv['ygrid']):
                ymask = (self['ygrid']>=y-inv.grid_size/2) & (self['ygrid']<y+inv.grid_size/2)
                for ix,x in enumerate(inv['xgrid']):
                    xmask = (self['xgrid']>=x-inv.grid_size/2) & (self['xgrid']<x+inv.grid_size/2)
                    if np.sum(ymask) == 0 and np.sum(xmask) == 0:
                        continue
                    data[iy,ix] = np.nanmean(self['data'][np.ix_(ymask,xmask)])
            inv['data'] = data

            for i in range(12):
                datam = np.full((len(inv['ygrid']),len(inv['xgrid'])),np.nan)
                for iy,y in enumerate(inv['ygrid']):
                    ymask = (self['ygrid']>=y-inv.grid_size/2) & (self['ygrid']<y+inv.grid_size/2)
                    for ix,x in enumerate(inv['xgrid']):
                        xmask = (self['xgrid']>=x-inv.grid_size/2) & (self['xgrid']<x+inv.grid_size/2)
                        if np.sum(ymask) == 0 and np.sum(xmask) == 0:
                            continue
                        datam[iy,ix] = np.nanmean(self['NH3'][:,:,i][np.ix_(ymask,xmask)])
                inv['NH3'][:,:,i] = datam

        for field in fields_to_copy:
            if field in l3.keys():
                inv[field] = l3[field].copy()
            else:
                self.logger.warning(f'{field} does not exist in l3!')
        inv['landmask'] = self['landmask']
        # inv['NH3'] = self['NH3']

        return inv
    
    def get_mask(self,max_emission=1e-9,include_nan=True,
                 min_windtopo=None,max_windtopo=None,
                 min_surface_altitude=None,max_surface_altitude=None,
                 min_column_amount=None,max_column_amount=None,
                 min_wind_column=None,max_wind_column=None):# zitong add threshold for wind_column
        '''get a mask based on self['data']. pixels lower than max_emission will be True.
        nan will alse be True if include_nan
        '''
        mask = self['data'] <= max_emission
        mask = mask & (self['landmask'] == 1)
        if include_nan:
            mask = mask | np.isnan(self['data'])
        if min_windtopo is not None:
            wt = np.abs(self['wind_topo']/self['column_amount'])
            mask = mask & (wt >= min_windtopo)
        if max_windtopo is not None:
            wt = np.abs(self['wind_topo']/self['column_amount'])
            mask = mask & (wt <= max_windtopo)
        if min_surface_altitude is not None:
            mask = mask & (self['surface_altitude'] > min_surface_altitude)
        if max_surface_altitude is not None:
            mask = mask & (self['surface_altitude'] <= max_surface_altitude)
        if min_column_amount is not None:
            mask = mask & (self['column_amount'] > min_column_amount)
        if max_column_amount is not None:
            mask = mask & (self['column_amount'] <= max_column_amount)
        if min_wind_column is not None:
            mask = mask & (self['wind_column'] > min_wind_column)
        if max_wind_column is not None:
            mask = mask & (self['wind_column'] <= max_wind_column)  

        return mask
    
    def plot(self,ax=None,scale='log',**kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5),subplot_kw={"projection": ccrs.PlateCarree()})
        else:
            fig = plt.gcf()
        if scale == 'log':
            from matplotlib.colors import LogNorm
            if 'vmin' in kwargs:
                inputNorm = LogNorm(vmin=kwargs['vmin'],vmax=kwargs['vmax'])
                kwargs.pop('vmin');
                kwargs.pop('vmax');
            else:
                inputNorm = LogNorm()
            pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self['data'],norm=inputNorm,
                                         **kwargs)
        else:
            pc = ax.pcolormesh(*F_center2edge(self['xgrid'],self['ygrid']),self['data'],**kwargs)
        ax.set_extent([self.west,self.east,self.south,self.north])
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='None',
                       edgecolor='black', linewidth=1)
        cb = fig.colorbar(pc,ax=ax)
        figout = {'fig':fig,'pc':pc,'ax':ax,'cb':cb}
        return figout

# class naming convention: https://pep8.org/#class-names
class AgriRegion():
    '''class for Level4 data (NH3 emission) calculated from Level3 data'''
    def __init__(self,geometry=None,xys=None,start_dt=None,end_dt=None,
                 west=None,east=None,south=None,north=None):
        '''
        geometry:
            a list of tuples for the polygon, e.g., [(xarray,yarray)], or geometry in a gpd row
        start/end_dt:
            datetime objects accurate to the day
        west, east, south, north:
            boundary of the region
        '''
        self.logger = logging.getLogger(__name__)
        self.start_dt = start_dt or dt.datetime(2008,1,1)
        self.end_dt = end_dt or dt.datetime.now()
        if geometry is None and xys is not None:
            geometry = xys
        if isinstance(geometry,list):
            xys = geometry
            self.west = west or np.min([np.min(xy[0]) for xy in xys])
            self.east = east or np.max([np.max(xy[0]) for xy in xys])
            self.south = south or np.min([np.min(xy[1]) for xy in xys])
            self.north = north or np.max([np.max(xy[1]) for xy in xys])
            self.xys = xys
        elif isinstance(geometry,shapely.geometry.multipolygon.MultiPolygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [g.exterior.xy for g in geometry.geoms] #zitong edit
        elif isinstance(geometry,shapely.geometry.polygon.Polygon):
            bounds = geometry.bounds
            self.west = west or bounds[0]
            self.east = east or bounds[2]
            self.south = south or bounds[1]
            self.north = north or bounds[3]
            self.xys = [geometry.exterior.xy]
        elif geometry is None:
            self.west = west
            self.east = east
            self.south = south
            self.north = north
            self.xys = [([west,west,east,east],[south,north,north,south])]     
    
    def get_region_emission(self,dt_array=None,l3_path_pattern=None,l3s=None,
                            masking_kw=None,
                            fit_topo_kw=None,fit_chem_kw=None,
                            field='NH3',l3_freq='1D'):# zitong add
        '''handle ammonia emission for the region
        dt_array:
            if provided, loads l3s from files. if none, assume daily within start/end_dt
        l3_path_pattern:
            if provided, loads l3s from files
        l3s:
            if l3_path_pattern not available, have to use provided l3s
        masking_kw:
            kw args to handle masking. may include
                nei_dir: directory of nei ag data, or
                monthly_filenames: a list of monthly nei ag files
                max_topo_emission: inventory emission threshold for topo fit,in mol/m2/s
                max_topo_emission: inventory emission threshold for chem fit
                {min,max}_{topo,chem}_{windtopo,column_amount}: min/max bounds for
                windtopo and column_amount in the aggregated l3 for topo or chem fits
                landmask_filename: file path of landmask (netcdf file)
        fit_topo/chem_kw:
            kw args for topo/chem fitting
        field:
            gas field of NEI2016: 'NH3' or 'NO2'
        freq:
            time freq of L3 files: '1D' for NH3, '1M' for NO2
        '''
        if l3_path_pattern is not None:
            if dt_array is None:
                dt_array = pd.period_range(self.start_dt,self.end_dt,freq=l3_freq)
            if_exist = np.array([os.path.exists(d.strftime(l3_path_pattern)) for d in dt_array])
            if np.sum(if_exist) < len(dt_array):
                self.logger.warning(
                    '{} l3 files exist for the {} periods'.format(
                        np.sum(if_exist),len(dt_array)))
                dt_array = dt_array.delete(np.arange(len(dt_array))[~if_exist])
            l3s = Level3_List(dt_array,west=self.west,east=self.east,
                              south=self.south,north=self.north)
            l3s.read_nc_pattern(l3_path_pattern=l3_path_pattern,
                                fields_name=['column_amount','wind_column',
                                             'num_samples','wind_topo','surface_altitude','wind_column_xy','wind_column_rs']) #,'skt'
        else:
            l3s = l3s.trim(west=self.west,east=self.east,south=self.south,north=self.north)
        self.l3all = l3s.aggregate()
        print('l3all:',self.l3all.keys())
        # make sure default kws are all empty dict
        masking_kw = masking_kw or {}
        fit_topo_kw = fit_topo_kw or {}
        fit_chem_kw = fit_chem_kw or {}
        # handle masks
        nei_dir = masking_kw.pop('nei_dir',None)
        landmask_filename = masking_kw.pop('landmask_filename',None)
        monthly_filenames = masking_kw.pop('monthly_filenames',None)
        max_topo_emission = masking_kw.pop('max_topo_emission',1e-9)
        max_chem_emission = masking_kw.pop('max_chem_emission',max_topo_emission)
        max_topo_windtopo = masking_kw.pop('max_topo_windtopo',np.inf)
        min_topo_windtopo = masking_kw.pop('min_topo_windtopo',-np.inf)
        max_chem_windtopo = masking_kw.pop('max_chem_windtopo',np.inf)
        min_chem_windtopo = masking_kw.pop('min_chem_windtopo',-np.inf)
        max_topo_column_amount = masking_kw.pop('max_topo_column_amount',np.inf)
        min_topo_column_amount = masking_kw.pop('min_topo_column_amount',-np.inf)
        max_chem_column_amount = masking_kw.pop('max_chem_column_amount',np.inf)
        min_chem_column_amount = masking_kw.pop('min_chem_column_amount',-np.inf)
        max_topo_wind_column = masking_kw.pop('max_topo_wind_column',np.inf)
        min_topo_wind_column = masking_kw.pop('min_topo_wind_column',-np.inf)
        max_chem_wind_column = masking_kw.pop('max_chem_wind_column',np.inf)
        min_chem_wind_column = masking_kw.pop('min_chem_wind_column',-np.inf)
        if nei_dir is None and monthly_filenames is None:
            self.logger.warning('no info provided about inventory, skipping')
            topo_mask = np.ones(self.l3all['num_samples'].shape,dtype=bool)
            chem_mask = np.ones(self.l3all['num_samples'].shape,dtype=bool)
            inventory_data = np.zeros(self.l3all['num_samples'].shape,dtype=bool)
        else:
            inv = Inventory(name='NEI',
                ).read_NEI_ag(
                monthly_filenames=monthly_filenames,
                nei_dir=nei_dir,unit='mol/m2/s',
                field=field # zitong add
                ).read_regrid_landmask(
                    self.l3all,landmask_filename=landmask_filename
                    ).regrid(
                        self.l3all,
                        fields_to_copy=['column_amount',
                                        'wind_topo','surface_altitude',
                                        'wind_column'] # zitong add 'skt'
                        )
            inventory_data = inv['data']
            landmask_data = inv['landmask']
            topo_mask = inv.get_mask(max_emission=max_topo_emission,
                                     min_windtopo=min_topo_windtopo,
                                     max_windtopo=max_topo_windtopo,
                                     min_column_amount=min_topo_column_amount,
                                     max_column_amount=max_topo_column_amount,
                                     min_wind_column=min_topo_wind_column,
                                     max_wind_column=max_topo_wind_column)

            chem_mask = inv.get_mask(max_emission=max_chem_emission,
                                     min_windtopo=min_chem_windtopo,
                                     max_windtopo=max_chem_windtopo,
                                     min_column_amount=min_chem_column_amount,
                                     max_column_amount=max_chem_column_amount,
                                     min_wind_column=min_chem_wind_column,
                                     max_wind_column=max_chem_wind_column)
        if 'mask' in fit_topo_kw.keys():
            topo_mask = topo_mask & fit_topo_kw['mask']
        if 'mask' in fit_chem_kw.keys():
            chem_mask = chem_mask & fit_chem_kw['mask']
        fit_topo_kw.update(dict(mask=topo_mask))
        fit_chem_kw.update(dict(mask=chem_mask))
        l3s.fit_topography(**fit_topo_kw)
        l3s.fit_chemistry(**fit_chem_kw)
        self.l3s = l3s
        self.l3all = l3s.aggregate()
        # store masks and inventory data to l3all for easy plotting
        self.l3all['topo_mask'] = topo_mask
        self.l3all['chem_mask'] = chem_mask
        self.l3all[inv.name] = inventory_data
        self.l3all['landmask'] = landmask_data   
        self.l3all['nei_month'] = inv['NH3']    


#### example usage
import sys,os,glob
sys.path.append('/projects/bbkc/zitong/Oversampling_matlab')
from popy import Level3_List
sys.path.append('/projects/bbkc/zitong/IDS/IDS.py')
from IDS import AgriRegion
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
import cartopy.feature as cfeature
from IDS import Inventory

start_dt = dt.datetime(2008,1,1)
end_dt = dt.datetime(2022,12,31)
dt_array = pd.period_range(start_dt,end_dt,freq='1D')

west,east,south,north = -125,-67,25.1,49 # CONUS
l3_path_pattern= '/projects/bbkc/zitong/Data/IASINH3L3_flux01/CONUS_%Y_%m_%d.nc' 
nei_dir = '/projects/bbkc/zitong/Data/NEI_gridded'
landmask_filename = '/projects/bbkc/zitong/Data/NLDAS_masks-veg-soil.nc4'
mdwst = AgriRegion(start_dt=start_dt,end_dt=end_dt,
                    west=west,east=east,south=south,north=north)
# load emissions in the region
mdwst.get_region_emission(l3_path_pattern=l3_path_pattern,
                          field='NH3',l3_freq='1D',
                          masking_kw=dict(            
                              nei_dir=nei_dir,
                              landmask_filename=landmask_filename,
                              max_topo_emission=1e-9,
                              max_chem_emission=1e-9,
                              ),
                          fit_topo_kw=dict(
                              resample_rule='6M',
                              max_iter=2,
                              outlier_std=2, # zitong add
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf,
                              ),
                          fit_chem_kw=dict(
                              resample_rule='3M',
                              max_iter=2,
                              outlier_std=2, # zitong add
                              min_windtopo=-np.inf,
                              max_windtopo=np.inf,
                              max_wind_column=np.inf))