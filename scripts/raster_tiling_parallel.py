# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: Creates a raster tile set from a vector tile index and raster source directory (parallelized version)
Last revision: February 3, 2025
Licence: BSD 3-Clause License 
"""

import os
import glob
import yaml
import time
import geopandas as gpd
import rasterio

# Set PROJ_LIB and GDAL_DATA dynamically (to use GDAL install linked to rasterio  package instead of GDAL in system path)
def init_worker():
    import os, rasterio
    RASTERIO_PATH = os.path.dirname(rasterio.__file__)
    os.environ["PROJ_LIB"] = os.path.join(RASTERIO_PATH, "proj_data")
    os.environ["GDAL_DATA"] = os.path.join(RASTERIO_PATH, "gdal_data")
from concurrent.futures import ProcessPoolExecutor

from datetime import datetime
from general.raster_clip import raster_clip
from utilities.utilities import get_filepath

print(f'Rasterio GDAL version: {rasterio.__gdal_version__}')
print(f'PROJ_LIB: {os.environ.get("PROJ_LIB")}')
print(f'GDAL_DATA: {os.environ.get("GDAL_DATA")}')

#%% Parameters

with open("config.yaml", "r") as file:
    params = yaml.safe_load(file)


dir_in = params["raster_tiling"]["dir_in"] # Path to input directory (containing geotiff raster files with .tif extension)
dir_out =  params["raster_tiling"]["dir_out"]  # Path to output directory
suffix_out = params["raster_tiling"]["suffix_out"] # Suffix added to output files

fpath_tile_index = params["tile_index"]["fpath"] # Path to the source extent file with a geometry column containing the extent of each tile and a column with an identifier for each raster file
tile_identifier = params['tile_index']["identifier"] # Column name used to uniquely identify a tile in the tile index

crs = params["raster_tiling"]["crs"] # Coordinate reference system
buffer_width = params["raster_tiling"]["buffer_width"] # Width of the buffer to apply around each tile (set to 0, if you do not want a buffer)

max_workers = params["parallel_processing"]["max_workers"]

#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)

#%% Add filepath column to tile index

files_in = glob.glob(dir_in + '*.tif')
tiles['filepath'] = tiles[tile_identifier].apply(lambda x: get_filepath(files_in, x))

#%% Apply buffer around each tile

tiles_buffered = tiles.copy()

if buffer_width > 0:
    tiles_buffered['geometry'] = tiles_buffered['geometry'].buffer(buffer_width, cap_style='square', join_style='mitre')

#%% Process tile function

def process_tile(index):
    clipper = tiles_buffered.iloc[[index]]
    print('--------------------------------------------------------------------------------')
    print(f'{index+1} / {n_tiles} - ID: {clipper[tile_identifier].values}')
    
    try:
        tic = time.perf_counter()
            
        # Clip raster
        windowed_subset, windowed_transform = raster_clip(clipper, tiles, False)
 
        # Write clipped raster to file
        fpath_out = dir_out + clipper.tileid.iloc[0] + suffix_out + '.tif'
        fout = rasterio.open(
            fpath_out,
            mode = 'w',
            driver = 'GTiff',
            height = windowed_subset.shape[1],
            width = windowed_subset.shape[2],
            count = windowed_subset.shape[0],
            dtype = windowed_subset.dtype,
            crs = crs,
            compress="lzw",
            transform = windowed_transform,
            )
        
        fout.write(windowed_subset)
        fout.close()
    
        current_time = datetime.now().strftime("%H:%M:%S")
        toc = time.perf_counter()
        
        print(f"{current_time} - File written to: {fpath_out}")
        print(f"Elapsed time {toc - tic:0.4f} seconds")

    except Exception as e:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time} - ERROR PROCESSING TILE {index}: {e}")

if __name__ == '__main__':

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        # Return results in order of task submission 
        list(executor.map(process_tile, tiles_buffered.index))
