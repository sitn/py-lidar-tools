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



# fpath_las_in = 'C:/Projects/lidar_tree_detection/input/ne_2022/'

# fpath_in = []
# fpath_in.append('C:/Projects/lidar_tree_detection/input/ne_2022/2552000_1201000.copc.laz')
# fpath_in.append('C:/Projects/lidar_tree_detection/input/ne_2022/2552000_1200500.copc.laz')

# clipper vector
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry19_ch1903p_survey_chull.shp")
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry01_ch1903p_survey_chull.shp")
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry20e_ch1903p_survey_chull.shp")
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_chambrelien_ch1903p_survey_chull.shp")
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_rochefort_ch1903p_survey_v220418_chull.shp")
clipper = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ge_2017_versoix_ch1903p_survey_chull.shp")


# LAS tiles
tiles = gpd.read_file("C:/Projects/lidar_tree_detection/input/ne_2022/tuiles_lidar_2022.shp")


#%% output files

fpath_las_out = 'C:/Projects/lidar_tree_detection/input/ne_2022/ne_2022_rochefort_ch1903p_survey_v220418.las'


#%% spatial join of tiles and bounding polygon

tiles_s = gpd.overlay(tiles, clipper, how='intersection', keep_geom_type=None, make_valid=True)

tiles_s.insert(1, "fpath", fpath_las_in + tiles_s.tileid + ".copc.laz", True)


#%% extract boundary polygon from convex hull

# plot convex hull
clipper.geometry[0]

x_chull, y_chull = clipper.geometry[0].exterior.coords.xy
# boundary = xy_chull.tolist()
boundary = np.dstack([x_chull, y_chull])[0].tolist()

# create boundary polygon
path_p = matplotlib.path.Path(boundary)


#%% create empty LAS file

# configure LAS header
header = laspy.LasHeader(point_format=7, version="1.4")
# header = laspy.LasHeader(point_format=3, version="1.2")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
header.offsets = np.append(np.min(boundary, axis=0), 0.0)
header.scales = np.array([0.01, 0.01, 0.01])


#%% process tiles

all_points = laspy.PackedPointRecord.empty(header.point_format)

for fpath in tiles_s.fpath:
    
    print("Processing file: %s" % fpath)
    
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
    
    
#%% write LAS file

las = laspy.LasData(header=header, points=all_points)
las.write(fpath_las_out)

print("LAS file written to: %s" % fpath_las_out)























#%% TEST

    
# Create a Las
header = laspy.LasHeader(point_format=7, version="1.4")
# header = laspy.LasHeader(point_format=3, version="1.2")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
header.offsets = np.append(np.min(boundary, axis=0), 0.0)
header.scales = np.array([0.01, 0.01, 0.01])
las = laspy.LasData(header)

# Read tiles
pc1 = laspy.read(fpath_in[0])
pc2 = laspy.read(fpath_in[1])

# point in polygon overlay
idxl_1 = path_p.contains_points(np.stack([pc1.x, pc1.y], axis=0).transpose((1, 0)))
idxl_2 = path_p.contains_points(np.stack([pc2.x, pc2.y], axis=0).transpose((1, 0)))

pc1.points = pc1.points[idxl_1]
pc2.points = pc2.points[idxl_2]
print("Points 1: %u" % len(pc1.points))
print("Points 2: %u" % len(pc2.points))

pc1.change_scaling(scales = header.scales, offsets = header.offsets)
pc2.change_scaling(scales = header.scales, offsets = header.offsets)


# las.x = np.append(pc1.x, pc2.x)
# las.y = np.append(pc1.y, pc2.y)
# las.z = np.append(pc1.z, pc2.z)
# las.intensity = np.append(pc1.intensity, pc2.intensity)
# las.return_number = np.append(pc1.return_number, pc2.return_number)
# las.number_of_returns = np.append(pc1.number_of_returns, pc2.number_of_returns)
# las.synthetic = np.append(pc1.synthetic, pc2.synthetic)
# las.key_point = np.append(pc1.key_point, pc2.key_point)
# las.withheld = np.append(pc1.withheld, pc2.withheld)
# las.overlap = np.append(pc1.overlap, pc2.overlap)
# las.scanner_channel = np.append(pc1.scanner_channel, pc2.scanner_channel)
# las.scan_direction_flag = np.append(pc1.scan_direction_flag, pc2.scan_direction_flag)

len(pc1)
las['x'] = 1


las['x'] = np.append(las['x'], pc1['x'])
len(las.x)
len(las.intensity)



las.x[0:4] = 1


las['intensity'] = np.append(las['intensity'], pc1['intensity'])



for dimension in las.point_format.dimensions:
    las[dimension.name] = np.append(las[dimension.name], pc1[dimension.name])
    
    print(dimension.name)
    print("Current points: %u" % len(las[dimension.name]))
    print("Points appended: %u" % len(pc1[dimension.name]))
    print("Current points: %u" % len(las[dimension.name]))
    
    
    
    
    
    
for dimension in las.point_format.dimensions:
    las[dimension.name] = np.append(pc1[dimension.name], pc2[dimension.name])
    print(dimension.name)

# las.change_scaling(scales = header.scales, offsets = header.offsets)

las.write("C:/Projects/lidar_tree_detection/input/clip_temp_3.las")
