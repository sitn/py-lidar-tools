# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: Merge vector files
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
import numpy as np
from pathlib import Path

from utilities.utilities import get_filepath


#%% Parameters

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)

fpath_in = params["vector_file_merging"]["fpath_in"] # Path to input directory
fpath_out = params["vector_file_merging"]["fpath_out"] # Path to output file

fpath_tile_index = params["tile_index"]["fpath"] # Path to the source extent file with a geometry column containing the extent of each tile and a column with an identifier for each raster file
tile_identifier = params["tile_index"]["identifier"] # Column name used to uniquely identify a tile in the tile index


#%% Output file

extension = Path(fpath_out).suffix.lower()

if not np.isin(extension, [".shp", ".feather"]):
    raise RuntimeError("The output filepath mush have extension .shp or .feather")


#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)


#%% Add filepath column to tile index

files_in = glob.glob(fpath_in)
tiles["filepath"] = tiles[tile_identifier].apply(lambda x: get_filepath(files_in, x))


#%% Merge all tiles in a single geodataframe

tic = time.perf_counter()

for index, tile in tiles.iterrows():
    
    print(f"Reading {index+1} / {n_tiles} - {tile.filepath}")
    
    try:
        
        match extension:
            
            case ".feather":
                if index == 0:
                    gdf_all = gpd.read_feather(tile.filepath)
                else:
                    gdf = gpd.read_feather(tile.filepath)
                
            case ".shp":
                if index == 0:
                    gdf_all = gpd.read_file(tile.filepath)
                else: 
                    gdf = gpd.read_file(tile.filepath)
        
        coords = gdf.get_coordinates(include_z=True, ignore_index=False, index_parts=False)
        
        # Remove points located outside the tile bounding box
        tile_bbox = np.asarray(tile.geometry.bounds)
        idxl_inside_bbox = (coords.x > tile_bbox[0]) & (coords.x < tile_bbox[2]) & (coords.y > tile_bbox[1]) & (coords.y < tile_bbox[3])
        idxl_on_bbox = np.isin(coords.x, tile_bbox[[0,2]]) | np.isin(coords.y, tile_bbox[[1,3]])
        idxl_filter = idxl_inside_bbox | idxl_on_bbox
        
        # Append points to geodataframe
        gdf_all = pd.concat([gdf_all, gdf.loc[idxl_filter,:]], ignore_index=True)

    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue  # Skip to the next tile if an error occurs
                
toc = time.perf_counter()
current_time = datetime.now().strftime("%H:%M:%S")
print(f"{current_time} Finished merging - Elapsed time {toc - tic:0.4f} seconds")
   

#%% Write all points to file

tic = time.perf_counter()

print(f"Writing data to {fpath_out}")
match extension:
    case ".feather":
        gdf_all.to_feather(path=fpath_out, index=True)
    case ".shp":
        gdf_all.to_file(filename=fpath_out, driver="ESRI Shapefile")

toc = time.perf_counter()
current_time = datetime.now().strftime("%H:%M:%S")
print(f"{current_time} Finished merging - Elapsed time {toc - tic:0.4f} seconds")
   

