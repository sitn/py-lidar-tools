# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: Detects tree top locations in a raster canopy height model
Last revision: January 31, 2024
Licence: BSD 3-Clause License
"""

import os
import glob
import yaml
import time
import geopandas as gpd
import rasterio
# Set PROJ_LIB and GDAL_DATA dynamically (to use GDAL install linked to rasterio  package instead of GDAL in system path)
RASTERIO_PATH = os.path.dirname(rasterio.__file__)
os.environ["PROJ_LIB"] = os.path.join(RASTERIO_PATH, "proj_data")
os.environ["GDAL_DATA"] = os.path.join(RASTERIO_PATH, "gdal_data")
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.utilities import get_filepath
from vegetation.canopy_peaks import canopy_peaks


#%% Parameters

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)


dir_in_chm = params["tree_top_detection"]["dir_in_chm"] # Path to input directory (containing geotiff raster files with .tif extension)
dir_out = params["tree_top_detection"]["dir_out"] # Path to output directory
suffix_out = params["tree_top_detection"]["suffix_out"] # Suffix added to output files

fpath_tile_index = params["tile_index"]["fpath"] # Path to the source extent file with a geometry column containing the extent of each tile and a column with an identifier for each raster file
tile_identifier = params["tile_index"]["identifier"] # Column name used to uniquely identify a tile in the tile index

export_shapefile = params["tree_top_detection"]["export_shapefile"] # Export detected tree tops to shapefile
export_feather = params["tree_top_detection"]["export_feather"] # Export detected tree tops to feather file

radius_function = eval(params["tree_top_detection"]["radius_function"])
min_height = params["tree_top_detection"]["min_height"] 
gaussian_filter_sigma = params["tree_top_detection"]["gaussian_filter_sigma"]

verbose = params["tree_top_detection"]["verbose"]
fig = params["tree_top_detection"]["fig"]


#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)


#%% Add filepath column to tile index

files_in = glob.glob(dir_in_chm + "*.tif")
tiles["filepath"] = tiles[tile_identifier].apply(lambda x: get_filepath(files_in, x))


#%% Process files

for index, tile in tiles.iterrows():
    
    print("--------------------------------------------------------------------------------")
    print(f"{index+1} / {n_tiles} - ID: {tile[tile_identifier]}")
    
    try:
        
        tic = time.perf_counter()

        # Read canopy height model
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
            plt.xlabel("Column #")
            plt.ylabel("Row #")
            plt.title("Canopy height model")
        
        # Detect local maxima points in raster CHM
        crh, xyh = canopy_peaks(
            chm,
            src_transform,
            gaussian_filter_sigma=gaussian_filter_sigma,
            method="default",      
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
        #idxl_filter = idxl_inside_bbox | idxl_on_bbox
        
        # Create dataframe
        d = {}
        d["H"] = xyh[:,2]
        d["RADIUS"] = radius_function(xyh[:,2])
        d["LOCATION"] = location[:]
        d["SOURCE"] = tile[tile_identifier]
        df = pd.DataFrame(data=d)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=xyh[:,0], y=xyh[:,1], crs=src_crs))
        
        # Write geodataframe to ESRI shapefile
        if export_shapefile:
            current_time = datetime.now().strftime("%H:%M:%S")
            fpath_out = dir_out + tile[tile_identifier] + suffix_out + ".shp"
            gdf.to_file(filename=fpath_out, driver="ESRI Shapefile")
            print(f"{current_time} - File written to: {fpath_out}")

        # Write geodataframe to Feather file
        if export_feather:
            current_time = datetime.now().strftime("%H:%M:%S")
            fpath_out = dir_out + tile[tile_identifier] + suffix_out + ".feather"
            gdf.to_feather(path=fpath_out, index=True)
            print(f"{current_time} - File written to: {fpath_out}")

        toc = time.perf_counter()
        print(f"Elapsed time {toc - tic:0.4f} seconds")
    
    
    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue  # Skip to the next tile if an error occurs

