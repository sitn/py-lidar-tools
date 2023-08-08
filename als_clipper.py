# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:55:55 2023

@author: ParkanM
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:46:54 2023

@author: ParkanM
"""

# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan
Last revision: May 31, 2023
Licence: GNU General Public Licence (GPL), see https://www.gnu.org/licenses/gpl.html
"""

# https://github.com/laspy/laspy/issues/156
# https://pypi.org/project/laspy/

# Install with LAZ support via both lazrs & laszip
# pip install laspy[lazrs,laszip]


# Import libraries
import os
from pyproj import CRS
from pyproj.enums import WktVersion
import geopandas as gpd
import matplotlib
import laspy
import numpy as np


#%% input files

fpath_las_in = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022/2_las/'

# clipper vector
# clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry19_ch1903p_survey_chull.shp")
clipper = gpd.read_file("D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp")

# LAS tiles
tiles = gpd.read_file("D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/tuiles_lidar_2022.shp")

#%% output folder

fpath_dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/'


#%% clip function

def clip(candidates, geometry):

    #  boundary polygon
    x_poly, y_poly = geometry.exterior.coords.xy
    boundary = np.dstack([x_poly, y_poly])[0].tolist()
    
    # create boundary polygon
    path_p = matplotlib.path.Path(boundary)
    
    # create empty LAS file
    
    # configure LAS header
    header = laspy.LasHeader(point_format=7, version="1.4")
    # header = laspy.LasHeader(point_format=3, version="1.2")
    crs = CRS.from_epsg(2056)
    header.add_crs(crs)
    header.offsets = np.append(np.min(boundary, axis=0), 0.0)
    header.scales = np.array([0.01, 0.01, 0.01])
    
    # process candidates
    all_points = laspy.PackedPointRecord.empty(header.point_format)
    
    for fpath in candidates:
        
        # print("Processing file: %s" % fpath)
        
        # read LAS file
        pc = laspy.read(fpath)
        
        # point in polygon filter
        idxl_in = path_p.contains_points(np.stack([pc.x, pc.y], axis=0).transpose((1, 0)))
        n_in = np.count_nonzero(idxl_in)
        
        print("Points filtered: %u" % n_in)
        pc.points = pc.points[idxl_in]
    
        # adjust scalings and offsets
        pc.change_scaling(scales = header.scales, offsets = header.offsets)
        
        # append clipped points to array
        all_points.array = np.append(all_points.array, pc.points.array)
        
        
    las = laspy.LasData(header=header, points=all_points)
        
    return las


#%% add filepath to clipper

clipper.insert(1, "fpath", fpath_las_in + clipper.tileid + ".las", True)


#%% spatial join of tiles and bounding polygon

tiles_s = gpd.overlay(tiles, clipper, how='intersection', keep_geom_type=None, make_valid=True)

tiles_s.insert(1, "fpath", fpath_las_in + tiles_s.tileid_1 + ".copc.laz", True)


#%%  clip each feature individually

n = len(clipper)

for index, feature in clipper.iterrows():
    
    print("Processing file %u / %u: %s" % (index+1, n, feature.tileid))
    
    idxl = (tiles_s["tileid_2"] == feature.tileid)

    # clip
    las = clip(tiles_s[idxl].fpath, feature.geometry)

    # write LAS file
    fpath_las_out = fpath_dir_out + feature.tileid + '.las'
    las.write(fpath_las_out)   
    print("LAS file written to: %s" % fpath_las_out)




    
    