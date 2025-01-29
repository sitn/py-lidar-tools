# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan
Description: Creates a raster tile set from a vector tile grid and raster source
Last revision: January 23, 2025
Licence: BSD 3-Clause License 
"""

import time
import geopandas as gpd
import rasterio
import glob

from general.raster_clip import raster_clip
from utilities.utilities import get_filepath

        
#%% Parameters

dir_in = 'D:/Data/images/2022/RGBI/' # Path to input directory (containing geotiff raster files with .tif extension)
dir_out = 'D:/Data/images/2022/RGBI_BUFFERED/' # Path to output directory

suffix_out = '_20cm_rgbi_buffered' # Suffix added to output files

# Path to the the source extent file. The source extent index must be a shapefile 
# with a geometry column containing the extent of each file and a tileid column with the name of each raster file
fpath_tile_index = 'D:/Data/pointclouds/2022/tiles/tile_index_lidar_2022_local.shp'
tile_identifier = 'tileid' # column name sed to uniquely identify a tile in the tile index

crs = 'EPSG:2056' # Coordinate reference system
buffer_width = 20 # Width of the buffer to apply around each tile (set to 0, if you do not want a buffer)


#%% Read tile index

tiles = gpd.read_file(fpath_tile_index)
n_tiles = len(tiles)


#%% Add filepath column to tile index

files_in = glob.glob(dir_in + '*.tif')
tiles['filepath'] = tiles[tile_identifier].apply(lambda x: get_filepath(files_in, x))

'''
tiles['filepath'] = ''
for index, tile in tiles.iterrows(): 
    tiles.at[index,'filepath'] = get_filepath(files_in, tile[tile_identifier])
'''

#%% Apply buffer around each tile

tiles_buffered = tiles.copy()

if buffer_width > 0:
    tiles_buffered['geometry'] = tiles_buffered['geometry'].buffer(buffer_width, cap_style='square', join_style='mitre')


#%% Process tiles

for index in tiles_buffered.index:
    
    clipper = tiles_buffered.iloc[[index]]

    print('--------------------------------------------------------------------------------')
    print(f'{index+1} / {n_tiles} - ID: {clipper[tile_identifier].iloc[0]}')
    print('--------------------------------------------------------------------------------')
    
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
    
        toc = time.perf_counter()
        print(f"File written to {fpath_out}")
        print(f"Elapsed time {toc - tic:0.4f} seconds")

    except Exception as e:
        print(f"Error processing tile {index}: {e}")
        continue
