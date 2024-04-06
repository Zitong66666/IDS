import pandas as pd
import os,sys,glob
import logging
from netCDF4 import Dataset
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
            murl = '{}/2016fh_16j_ag_0pt1degree_month_{:02d}.ncf'.format(url,imonth)
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
            glob.glob(os.path.join(nei_dir,'2016fh_16j_ag_0pt1degree_month_*.ncf'))
        mons = []
        for i,filename in enumerate(monthly_filenames):
            nc = Dataset(filename)
            if i == 0:
                monthly_fields = np.zeros((nc.dimensions['lat'].size,
                                           nc.dimensions['lon'].size,
                                           len(monthly_filenames)))
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
                if nc_unit == 'kg/m2/s' and unit=='nmol/m2/s' and field in ['NH3','NH3_FERT']:
                    self.logger.warning(f'unit of {field} will be converted from {nc_unit} to {unit}')
                    self[f'{field} unit'] = unit
                    unit_factor = 1e9/0.017
                elif nc_unit == 'kg/m2/s' and unit=='mol/m2/s' and field in ['NH3','NH3_FERT']:
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
    
    def read_regrid_landmask(self,l3,landmask_dir=None,field='CONUS_mask'): # zitong add
        '''
        read and regrid land mask to match the mesh of a l3 data object
        landmask -- 1:land, 0:water/background
        '''
        nc = Dataset(landmask_dir)
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
        ymesh,xmesh = np.meshgrid(inv['ygrid'],inv['xgrid'])
        inv.grid_size = l3.grid_size
        inv.west = inv['xgrid'].min()-inv.grid_size
        inv.east = inv['xgrid'].max()+inv.grid_size
        inv.south = inv['ygrid'].min()-inv.grid_size
        inv.north = inv['ygrid'].max()+inv.grid_size
        if method in ['interpolate']:
            f = RegularGridInterpolator((self['ygrid'],self['xgrid']),self['data'],bounds_error=False)
            inv['data'] = f((ymesh,xmesh)).T
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
        for field in fields_to_copy:
            if field in l3.keys():
                inv[field] = l3[field].copy()
            else:
                self.logger.warning(f'{field} does not exist in l3!')
        inv['landmask'] = self['landmask']
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
    
    def  get_region_emission(self,dt_array=None,l3_path_pattern=None,l3s=None,
                            masking_kw=None,
                            fit_topo_kw=None,fit_chem_kw=None):
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
        fit_topo/chem_kw:
            kw args for topo/chem fitting
        '''
        if l3_path_pattern is not None:
            if dt_array is None:
                dt_array = pd.period_range(self.start_dt,self.end_dt,freq='1D')
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
                                             'num_samples','wind_topo','surface_altitude'])
        else:
            l3s = l3s.trim(west=self.west,east=self.east,south=self.south,north=self.north)
        self.l3all = l3s.aggregate()
        # make sure default kws are all empty dict
        masking_kw = masking_kw or {}
        fit_topo_kw = fit_topo_kw or {}
        fit_chem_kw = fit_chem_kw or {}
        # handle masks
        nei_dir = masking_kw.pop('nei_dir',None)
        landmask_dir = masking_kw.pop('landmask_dir',None)
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
                nei_dir=nei_dir,unit='mol/m2/s'
                ).read_regrid_landmask(
                    self.l3all,landmask_dir=landmask_dir
                    ).regrid(
                        self.l3all,
                        fields_to_copy=['column_amount',
                                        'wind_topo','surface_altitude',
                                        'wind_column'] # zitong add
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
    
    def process_by_region(self,l3_path_pattern,topo_kw=None,chem_kw=None,
                 region_shpfilename=None, region_num=None):
        
        '''
        calculate scale heights and lifetimes by regions

        region_shpfilename: multipolygon shp filename of regions

        region_num: if not None, calculate for one specific given region
        '''

        topo_scale_heights = {}
        chem_lifetimes = {}
        wind_column_topo = []     

        # for one region
        if region_shpfilename is not None and region_num is not None: 
            gdf = gpd.read_file(region_shpfilename)
            gdf_region = gdf[gdf['Region'] == region_num]
            region_polygon = unary_union(gdf_region.geometry)
            if isinstance(region_polygon,shapely.geometry.polygon.Polygon):
                self.region_xys = [region_polygon.exterior.xy]
            elif isinstance(region_polygon,shapely.geometry.multipolygon.MultiPolygon):
                self.region_xys = [g.exterior.xy for g in region_polygon.geoms]

            self.get_l3(l3_path_pattern=l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw, region_xys=self.region_xys)
            topo_scale_heights = self.l3ms.df['topo_scale_height']
            chem_lifetimes = self.l3ms.df['chem_lifetime']
            # wind_column_topo = self.l3ms.df['wind_column_topo']

        # for all regions
        region_names = ['Pacific','Mountain','Northern Plains','Southern Plains','Lake States',
                        'Corn Belt','Delta States','Southeast','Appalachia','Northeast']   
        if region_shpfilename is not None and region_num is None: 
            gdf = gpd.read_file(region_shpfilename)
            region_polygons = unary_union(gdf.geometry)
            region_xys = []
            for i in range(0,11):
                if isinstance(region_polygons,shapely.geometry.polygon.Polygon):
                    region_xys.append([region_polygons.exterior.xy])
                elif isinstance(region_polygons,shapely.geometry.multipolygon.MultiPolygon):
                    region_xys.append([g.exterior.xy for g in region_polygons.geoms])
            self.region_xys = region_xys

            for i in range(0,11):
                region_xy = self.region_xys[i]
                self.get_l3(l3_path_pattern=l3_path_pattern,topo_kw=topo_kw,chem_kw=chem_kw, region_xy=region_xy)
                topo_scale_heights[region_names[i]] = self.l3ms.df['topo_scale_height']
                chem_lifetimes[region_names[i]] = self.l3ms.df['chem_lifetime']
                wind_column_topo = wind_column_topo + self.l3ms.df['wind_column_topo']

        self.topo_scale_heights = topo_scale_heights
        self.chem_lifetimes = chem_lifetimes     

        
        for l3 in self.l3ds:
            month = l3.start_python_datetime.strftime('%Y-%m')
            sh = self.topo_scale_heights[month]
            cl = self.chem_lifetimes[month]
            wc_topo = l3['wind_column']-sh*l3['wind_topo']
            l3['wind_column_topo'] = wc_topo*self.region_mask
            wc_chem = l3['wind_column_topo']-cl*l3['column_amount']
            l3['wind_column_topo_chem'] = wc_chem*self.region_mask
            l3['month'] = month
            l3['date'] = l3.start_python_datetime
        
        l3ds_df = pd.DataFrame(self.l3ds)
        self.l3ds_df = l3ds_df
          
    
    def get_l3(self,l3_path_pattern,file_freq='1D',lonlat_margin=0.5,xgb_pblh_path_pattern=None,
               topo_kw=None,if_smooth_X=True,X_smooth_kw=None,cp_kw=None,chem_kw=None,
               region_xys = None):
        '''interface popy level 3 objects. get daily clean/polluted vcd and sfc conc.
        l3_path_pattern:
            time pattern of level3 files, e.g., '/projects/bbkc/zitong/Data/IASINH3L3_flux02/CONUS_%Y_%m_%d.nc'
        file_freq:
            frequency code by which l3 files are saved, e.g., 1D
        lonlat_margin:
            Level3_List will be trimmed this amount broader than Ammonia boundaries
        xgb_pblh_path_pattern:
            add sfc ppb estimated using xgb/amdar-based pblh instead of era5_blh,
            see https://doi.org/10.5194/amt-16-563-2023
        topo_kw:
            key word arguments for fit_topo related functions, may include min/max to mask l3 and args for fit_topography
        if_smooth_X:
            when the l3 record is short, it is better not to smooth scale height
        X_smooth_kw:
            key word arguments to smooth inverse scale height (X), default using savgol, window_length=5;polyorder=3
        cp_kw:
            key word arguments to separate clean/polluted and covert from vcd to sfc conc seperately
        '''
        topo_kw = topo_kw or {}
        chem_kw = chem_kw or {}

        # load level 3 files
        ewsn_dict = dict(west=self.west-lonlat_margin,east=self.east+lonlat_margin,
                            south=self.south-lonlat_margin,north=self.north+lonlat_margin)
        l3ds = Level3_List(pd.period_range(self.start_dt,self.end_dt,freq=file_freq),**ewsn_dict)
        # dt_array = pd.period_range(start=self.start_dt,end=self.end_dt,freq=file_freq)
        # l3ds = Level3_List(dt_array=dt_array,west=-128,east=-65,south=24,north=50)
        fields_name = ['column_amount','wind_topo','wind_column','surface_altitude']
        l3ds.read_nc_pattern(l3_path_pattern,
                            fields_name=fields_name.copy())
        l3 = l3ds.aggregate()
        l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit

        # create mask
        lonmesh,latmesh = np.meshgrid(l3['xgrid'],l3['ygrid'])
        region_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        # create region mask (by polygon)
        for xy in region_xys:
            # boundary_polygon = path.Path(xy)
            boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            region_mask = region_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)
        print(np.sum(region_mask==True))

        E0_mask = np.zeros(l3['num_samples'].shape,dtype=bool)
        # create E=0 mask (by raster) based on region mask
        for xy in self.xys:
            boundary_polygon = path.Path(xy)
            # boundary_polygon = path.Path([(x,y) for x,y in zip(*xy)])
            all_points = np.column_stack((lonmesh.ravel(),latmesh.ravel()))
            E0_mask = E0_mask | boundary_polygon.contains_points(all_points).reshape(lonmesh.shape)
        E0_mask = region_mask & E0_mask
        print(np.sum(E0_mask==True))
        if topo_kw is not None:
            if 'max_iter' not in topo_kw.keys():
                topo_kw['max_iter'] = 1

            # create topo mask
            min_windtopo = topo_kw.pop('min_windtopo',0.001)
            max_windtopo = topo_kw.pop('max_windtopo',0.1)
            min_H = topo_kw.pop('min_H',0.1)
            max_H = topo_kw.pop('max_H',2000)
            max_wind_column = topo_kw.pop('max_wind_column',1e-9) #???????
            topo_mask = E0_mask &\
            (np.abs(l3['wind_topo']/l3['column_amount'])>=min_windtopo) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['surface_altitude']>=min_H) &\
            (l3['surface_altitude']<=max_H) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in topo_kw.keys():
                topo_kw['mask'] = topo_kw['mask'] & topo_mask
            else:
                topo_kw['mask'] = topo_mask

            topo_kw['region_mask'] = region_mask
            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_topography(**topo_kw)
            self.topo_mask = topo_mask
            fields_name.append('wind_column_topo')
            

        if chem_kw is not None:
            max_windtopo = chem_kw.pop('max_windtopo',0.001)
            # print(np.nanmax(l3['column_amount'].reshape(-1, 1)))
            # print(np.nanmin(l3['column_amount'].reshape(-1, 1)))
            min_column_amount = chem_kw.pop('min_column_amount',2.5e-5) # not sure
            max_wind_column = chem_kw.pop('max_wind_column',1e-9) # not sure

            chem_mask = E0_mask &\
            (l3['column_amount']>=min_column_amount) &\
            (np.abs(l3['wind_topo']/l3['column_amount'])<=max_windtopo) &\
            (l3['wind_column']<=max_wind_column)
            if 'mask' in chem_kw.keys():
                chem_kw['mask'] = chem_kw['mask'] & chem_mask
            else:
                chem_kw['mask'] = chem_mask

            chem_kw['region_mask'] = region_mask
            # l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule']) # zitong edit
            l3ms.fit_chemistry(**chem_kw)
            self.chem_mask = chem_mask
            fields_name.append('wind_column_topo_chem')            


        # attach results to self
        self.l3ds = l3ds
        tmpdf = l3ms.df.copy()
        l3ms,_ = l3ds.resample(rule=topo_kw['resample_rule'])
        l3ms.df = tmpdf
        self.l3ms = l3ms # L3 list
        self.l3 = l3ds.aggregate() # L3 Data
        self.topo_mask = topo_mask
        self.region_mask = region_mask

