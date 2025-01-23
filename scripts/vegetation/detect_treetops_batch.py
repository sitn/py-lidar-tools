# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan
Last revision: January 23, 2024
Licence: GNU General Public Licence (GPL), see https://www.gnu.org/licenses/gpl.html
"""

# Import libraries
import time
import rasterio
from rasterio.windows import from_bounds
from canopy_peaks import canopy_peaks
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


#%% Input parameters

dhm_in = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022/5_Grid/mnc/20cm/mnc2022_20cm_cog.tif'
# dsm_in = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022/5_Grid/mnc/20cm/mnc2022_20cm_cog.tif'

fpath_tile_index = 'D:/Data/pointclouds/2022/tiles/tile_index_lidar_2022_local.shp'
fpath_coords_out = 'D:/Data/treetops/20241112/ne_2002_treetops_20241115_test.shp'


overlap = 40
radius_function = lambda h: 0.28 * h**0.59
verbose = True
fig = False


#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)

#%% Apply buffer around each tile

# tiles_buffered = tiles.copy()
# tiles_buffered['geometry'] = tiles_buffered['geometry'].buffer(overlap, resolution=16, cap_style='square', join_style='mitre')


#%% Create empty geodataframe to store output

# Define the schema for the empty GeoDataFrame, including columns and geometry type
columns = {'geometry': 'geometry', 'H': 'float', 'RADIUS': 'float', 'LOCATION': 'int', 'TILEID': 'str'}  # Adjust attribute names and types as needed
gdf = gpd.GeoDataFrame(columns=columns)
gdf.set_crs(tiles.crs, inplace=True)  # Set the CRS, adjust as necessary

# Save the empty GeoDataFrame to create an empty shapefile
# gdf.to_file(filename=fpath_coords_out, driver='ESRI Shapefile')


#%% Process files

for index, tile in tiles.iterrows():
# for index, tile in tiles_buffered.iterrows():
    
    '''
    if index==100:
        break    
    '''
    
    print('--------------------------------------------------------------------------------')
    print(f'{index+1} / {n_tiles} - ID: {tile.tileid}')
    print('--------------------------------------------------------------------------------')
    
    # tile = tiles.iloc[1]
    
    try:
        
        tic = time.perf_counter()

        
        '''
        #%% Create Region of Interest polygon
        roi = gpd.GeoDataFrame(
            {'geometry':  [tile.geometry]},
             crs=tiles_buffered.crs
        )
        
        #%% read region of interest (ROI) from raster CHM
     
        bbox = roi.iloc[0].geometry.bounds
        '''
        
        tile_bbox = np.asarray(tile.geometry.bounds)
        tile_bbox_buffered = tile_bbox + np.array([-overlap, -overlap, overlap, overlap])
        
        with rasterio.open(dhm_in) as src:
            
            raster_bounds = src.bounds
    
            # Adjust the bbox if it goes beyond the raster's extent
            adjusted_bbox = (
                max(tile_bbox_buffered[0], raster_bounds.left),   # x_min
                max(tile_bbox_buffered[1], raster_bounds.bottom),  # y_min
                min(tile_bbox_buffered[2], raster_bounds.right),   # x_max
                min(tile_bbox_buffered[3], raster_bounds.top)      # y_max
            )
        
            # Compute ROI window form the bounding box coordinates
            window = from_bounds(*adjusted_bbox, transform=src.transform)
            crs = src.crs
            
            # Read the region of interest (ROI)
            subset = src.read(1, window=window)
            
            # Affine transform
            win_transform  = src.window_transform(window)
        
        '''
        # Export subset to geotiff 
        fout = rasterio.open(
                'D:/Data/treetops/20241112/subset_test.tif',
                'w',
                driver = 'GTiff',
                height = subset.shape[0],
                width = subset.shape[1],
                count = 1,
                dtype = subset.dtype,
                crs = crs,
                transform = win_transform ,
            )
        fout.write(subset, 1)
        fout.close()
        '''
        
        # plot
        if fig:
            plt.figure(figsize=(6,8.5))
            plt.imshow(subset)
            plt.colorbar(shrink=0.5)
            plt.xlabel('Column #')
            plt.ylabel('Row #')
        
        #%% Detect local maxima points in raster CHM
        
        crh, xyh = canopy_peaks(
            subset,
            win_transform,
            smoothing_filter=1,
            method='default',      
            min_tree_height=2,   
            search_radius=radius_function, 
            fig=fig,           
            verbose=verbose   
        )
        
        idxl_inside_bbox = (xyh[:,0] > tile_bbox[0]) & (xyh[:,0] < tile_bbox[2]) & (xyh[:,1] > tile_bbox[1]) & (xyh[:,1] < tile_bbox[3])
        idxl_on_bbox = np.isin(xyh[:,0], tile_bbox[[0,2]]) | np.isin(xyh[:,1], tile_bbox[[1,3]])
        idxl_outside_bbox = np.invert(idxl_inside_bbox) & np.invert(idxl_on_bbox)
    
        location = np.zeros(len(xyh), dtype=np.uint8)   
        location[idxl_outside_bbox] = 0
        location[idxl_inside_bbox] = 1
        location[idxl_on_bbox] = 2

        idxl_filter = idxl_inside_bbox | idxl_on_bbox
    
        #%% Append points to geodataframe
        d = {}
        d['H'] = xyh[idxl_filter,2]
        d['RADIUS'] = radius_function(xyh[idxl_filter,2])
        d['LOCATION'] = location[idxl_filter]
        d['TILEID'] = tile.tileid
        
        # create dataframe from dictionnary
        df = pd.DataFrame(data=d)
        new_rows = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=xyh[idxl_filter,0], y=xyh[idxl_filter,1], crs='EPSG:2056'))
        
        gdf = pd.concat([gdf, new_rows], ignore_index=True)
    
        toc = time.perf_counter()
        print(f"Elapsed time {toc - tic:0.4f} seconds")
    
    
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue  # Skip to the next tile if an error occurs


#%% Write tile to single shapefiles

fpath_dir_out = 'D:/Data/treetops/20241112/'


for index, tile in tiles.iterrows():
# for index, tile in tiles_buffered.iterrows():
    
    '''
    if index==100:
        break    
    '''
    
    print('--------------------------------------------------------------------------------')
    print(f'{index+1} / {n_tiles} - ID: {tile.tileid}')
    print('--------------------------------------------------------------------------------')
    
    # tile = tiles.iloc[1]
    
    try:
        
        output_path = fpath_dir_out + tile.tileid + '.shp'

        # Filter the GeoDataFrame for the current TILEID
        tile_gdf = gdf[gdf["TILEID"] == tile.tileid]
        
        # Write the GeoDataFrame to a shapefile
        tile_gdf.to_file(output_path, driver='ESRI Shapefile')
        
        print(f"Saved shapefile for TILEID {tile.tileid} at {output_path}")

        
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue  # Skip to the next tile if an error occurs


#%% Write all points to single shapefile

# write geodataframe to ESRI shapefile
gdf.to_file(filename=fpath_coords_out, driver='ESRI Shapefile')
