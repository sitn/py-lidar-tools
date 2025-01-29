# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Description: clips and merges subsets of rasters
Last revision: January 23, 2025
Licence: BSD 3-Clause License 
"""

import numpy as np
import geopandas as gpd
import rasterio

#%% Functions

def raster_clip(clipper_extent, 
               source_extent,
               verbose: bool = False):
    
        # Get clipper bounds
        bounds = clipper_extent.total_bounds
        xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
        
        # Calculate raster dimensions
        width = xmax - xmin
        height = ymax - ymin

        # Compute intrsection between source and clipper extents
        tiles_s = gpd.overlay(source_extent, clipper_extent, how='intersection', keep_geom_type=True)
            
        # Clipper and merge source rasters
        for index, tile in tiles_s.iterrows(): 
            
            if verbose:
                print('--------------------------------------------------------------------------------')
                print(f'Clipping source {index+1} / {len(tiles_s)}')
                print('--------------------------------------------------------------------------------')
            
            src_raster = rasterio.open(tile.filepath_1)
            src_transform = src_raster.transform
            
            window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=src_raster.transform)
            subset = src_raster.read(window=window, masked=False, boundless=True, fill_value=0)
            src_raster.close()
            
            if index==0:

                pixel_resolution = src_transform.a
                cols = int(np.ceil(width / pixel_resolution))
                rows = int(np.ceil(height / pixel_resolution))

                # Affine transform for the output raster
                out_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)
                out_array = subset.copy()
                
            out_array = np.maximum(out_array, subset)
                
        return out_array, out_transform   