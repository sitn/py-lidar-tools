# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: 
Last revision: January 29, 2024
Licence: BSD 3-Clause License
"""

import time
import rasterio
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import glob

from utilities.utilities import get_filepath
from vegetation.canopy_peaks import canopy_peaks


#%% Input parameters

dir_in_chm = 'D:/Data/images/2022/CHM_BUFFERED/' # Path to input directory (containing geotiff raster files with .tif extension)
dir_out = 'D:/Data/treetops/20250129/' # Path to output directory
suffix_out = '_tree_tops' # Suffix added to output files

fpath_tile_index = 'D:/Data/pointclouds/2022/tiles/tile_index_lidar_2022_local.shp' # Path to the source extent file with a geometry column containing the extent of each tile and a column with an identifier for each raster file
tile_identifier = 'tileid' # Column name used to uniquely identify a tile in the tile index

export_shapefile = True # Export detected tree tops to shapefile
export_feather = True # Export detected tree tops to feather file

radius_function = lambda h: 0.28 * h**0.59
min_height = 2

verbose = True
fig = False


#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)


#%% Add filepath column to tile index

files_in = glob.glob(dir_in_chm + '*.tif')
tiles['filepath'] = tiles[tile_identifier].apply(lambda x: get_filepath(files_in, x))


#%% Process files

for index, tile in tiles.iterrows():
    
    print('--------------------------------------------------------------------------------')
    print(f'{index+1} / {n_tiles} - ID: {tile[tile_identifier]}')
    print('--------------------------------------------------------------------------------')
    
    try:
        
        tic = time.perf_counter()

        # Read canopy height model
        # fpath_chm = get_filepath(files_chm_in, tile.tileid)
        src = rasterio.open(tile.filepath)
        src_transform = src.transform
        src_crs = src.crs
        chm = src.read(1)
        src.close()
        
        # plot
        if fig:
            plt.figure(figsize=(6,8.5))
            plt.imshow(chm)
            plt.colorbar(shrink=0.5)
            plt.xlabel('Column #')
            plt.ylabel('Row #')
            plt.title('Canopy height model')
        
        # Detect local maxima points in raster CHM
        crh, xyh = canopy_peaks(
            chm,
            src_transform,
            smoothing_filter=1,
            method='default',      
            min_tree_height=min_height,   
            search_radius=radius_function, 
            fig=fig,           
            verbose=verbose   
        )
        
        # Filter points located outside the bounding box
        x_min, y_min, x_max, y_max = tile.geometry.bounds
        tile_bbox = np.asarray(tile.geometry.bounds)
        idxl_inside_bbox = (xyh[:,0] > x_min) & (xyh[:,0] < x_max) & (xyh[:,1] > y_min) & (xyh[:,1] < y_max)
        idxl_on_bbox = np.isin(xyh[:,0], tile_bbox[[0,2]]) | np.isin(xyh[:,1], tile_bbox[[1,3]])
        idxl_outside_bbox = np.invert(idxl_inside_bbox) & np.invert(idxl_on_bbox)
        location = np.zeros(len(xyh), dtype=np.uint8)   
        location[idxl_outside_bbox] = 0
        location[idxl_inside_bbox] = 1
        location[idxl_on_bbox] = 2
        idxl_filter = idxl_inside_bbox | idxl_on_bbox
        
        # Create dataframe
        d = {}
        d['H'] = xyh[idxl_filter,2]
        d['RADIUS'] = radius_function(xyh[idxl_filter,2])
        d['LOCATION'] = location[idxl_filter]
        d['SOURCE'] = tile[tile_identifier]
        df = pd.DataFrame(data=d)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=xyh[idxl_filter,0], y=xyh[idxl_filter,1], crs=src_crs))
        
     
        # Write geodataframe to ESRI shapefile
        if export_shapefile:
        
            fpath_out = dir_out + tile[tile_identifier] + suffix_out + '.shp'
            gdf.to_file(filename=fpath_out, driver="ESRI Shapefile")

        # Write geodataframe to Feather file
        if export_feather:
            fpath_out = dir_out + tile[tile_identifier] + suffix_out + '.feather'
            gdf.to_feather(path=fpath_out, index=True)

        toc = time.perf_counter()
        print(f"Elapsed time {toc - tic:0.4f} seconds")
    
    
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue  # Skip to the next tile if an error occurs

