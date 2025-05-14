import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import sys, os, glob
import dill as pickle
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

# Load the land cover data
land_cover_file = "/projects/bbkc/zitong/Data/NLCD/nlcd_2023/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
land_cover = rasterio.open(land_cover_file)

# Load the IASI L3 data
with open('/projects/bbkc/zitong/IDS/l3all_iasi.pkl', 'rb') as file:
    l3all = pickle.load(file)

IASI = l3all['wind_column_topo']
ammonia_lons = l3all['xgrid']
ammonia_lats = l3all['ygrid']
lons, lats = np.meshgrid(ammonia_lons, ammonia_lats)

# Mapping of land cover codes to classes
land_cover_to_class = {
    11: 'Water', 12: 'Water', 
    21: 'Developed', 22: 'Developed', 23: 'Developed', 24: 'Developed',
    31: 'Barren',
    41: 'Forest', 42: 'Forest', 43: 'Forest',
    51: 'Shrubland', 52: 'Shrubland',
    71: 'Herbaceous', 72: 'Herbaceous', 73: 'Herbaceous', 74: 'Herbaceous',
    81: 'Pasture/Hay',
    82: 'Crop',
    90: 'Wetlands', 95: 'Wetlands'
}

# Convert dictionary to array (fill unmapped with 'Unknown')
max_code = max(land_cover_to_class.keys()) + 1
land_cover_to_class_np = np.full(max_code + 1, 'Unknown', dtype='<U20')
for k, v in land_cover_to_class.items():
    land_cover_to_class_np[k] = v

# Function to compute the bounding box of an IASI pixel
def get_bounding_box(lon, lat, resolution):
    half_res = resolution / 2
    return lon - half_res, lat - half_res, lon + half_res, lat + half_res

# Determine the resolution (degrees)
iasi_resolution_lon = np.abs(lons[0, 1] - lons[0, 0])
iasi_resolution_lat = np.abs(lats[1, 0] - lats[0, 0])
iasi_resolution = max(iasi_resolution_lon, iasi_resolution_lat)

# Initialize output arrays
dominant_land_cover_classes = np.full_like(IASI, '', dtype='<U20')
dominant_land_cover_fractions = np.full_like(IASI, np.nan)

# Loop through IASI grid
for i in range(lons.shape[0]):
    for j in range(lons.shape[1]):
        lon = lons[i, j]
        lat = lats[i, j]

        # Get bounding box in WGS84 and transform to raster CRS
        west, south, east, north = get_bounding_box(lon, lat, iasi_resolution)
        minx, miny, maxx, maxy = transform_bounds('EPSG:4326', land_cover.crs.to_string(), west, south, east, north)

        # Define window and read data
        try:
            window = from_bounds(minx, miny, maxx, maxy, transform=land_cover.transform)
            img_data = land_cover.read(1, window=window, boundless=True, fill_value=0)
        except Exception as e:
            dominant_land_cover_classes[i, j] = ''
            dominant_land_cover_fractions[i, j] = np.nan
            continue

        # Filter out invalid values
        img_data_flat = img_data.flatten()
        img_data_flat = img_data_flat[(img_data_flat >= 0) & (img_data_flat < len(land_cover_to_class_np))]
        if img_data_flat.size == 0:
            continue

        # Map to broader classes
        img_classes = land_cover_to_class_np[img_data_flat]

        # Count and find dominant
        unique_classes, counts = np.unique(img_classes, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_class = unique_classes[dominant_idx]
        dominant_fraction = counts[dominant_idx] / counts.sum()

        dominant_land_cover_classes[i, j] = dominant_class
        dominant_land_cover_fractions[i, j] = dominant_fraction

# Save outputs
np.savetxt('/projects/bbkc/zitong/dominant_land_cover_classes_2023.csv', dominant_land_cover_classes, delimiter=',', fmt='%s', header='dominant_land_cover_classes', comments='')
np.savetxt('/projects/bbkc/zitong/dominant_land_cover_fractions_2023.csv', dominant_land_cover_fractions, delimiter=',', fmt='%.4f', header='dominant_land_cover_fractions', comments='')
