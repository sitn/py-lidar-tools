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
from pathlib import Path
from shapely.geometry import Polygon
from pyproj import CRS
from pyproj.enums import WktVersion
import geopandas as gpd
import fiona
import pandas as pd
import laspy
import numpy as np
import pathlib
import glob


#%% settings

# path_in = 'D:/Projects/lidar_jura_beech/data/als_jura_2022/*.las'
# path_out = 'D:/Projects/lidar_jura_beech/data/als_jura_2022/tile_index.shp'

# path_in = r'D:/LiDAR/lidar2023GE/LAS_classifie/*.las'
# path_out = D:/LiDAR/lidar2023GE/LAS_classifie/tile_index.shp'

# path_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/*.laz'
# path_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/LCDF_LV95_NF02_tile_index2.shp'

# path_in = 'D:/Projects/intemperie_cdf_20230724/data/pointclouds/livraison_20230810/*.laz'
# path_out = 'D:/Projects/intemperie_cdf_20230724/data/pointclouds/livraison_20230810/LCDF_LV95_NF02_tile_index.shp'

# path_in = 'D:/Data/LiDAR/2023 - CDF/flai_classification_v1/corrected/*.las'
# path_out = 'D:/Data/LiDAR/2023 - CDF/flai_classification_v1/corrected/LCDF_LV95_NF02_tile_index.shp'


# path_in = 'D:/Projects/intemperie_cdf_20230724/data/pointclouds/Flight_1_Geospatial_predict_all_classes_Flai_v2/*.laz'
# path_out = 'D:/Projects/intemperie_cdf_20230724/data/pointclouds/Flight_1_Geospatial_predict_all_classes_Flai_v2/tile_index.shp'


#path_in = '\\\\nesitn5/h$/geodata/pointclouds/Aeriallidar/Lidar2023_CHXFDS/2_las/2_laz/2_laz_helimap_v2/*.laz'
#path_out = '\\\\nesitn5/h$/geodata/pointclouds/Aeriallidar/Lidar2023_CHXFDS/2_las/2_laz/2_laz_helimap_v2/tile_index.shp'


#path_in = '\\\\nesitn5/h$/geodata/pointclouds/Aeriallidar/Lidar2023_CHXFDS/2_las/vol2/2_laz_helimap/*.laz'
#path_out = '\\\\nesitn5/h$/geodata/pointclouds/Aeriallidar/Lidar2023_CHXFDS/2_las/vol2/2_laz_helimap/tile_index.shp'

path_in = 'D:/Data/pointclouds/2023/all/flight_1/*.las'
path_out = 'D:/Data/pointclouds/2023/all/flight_1/tile_index.shp'

path_in = 'D:/Data/pointclouds/2023/all/flight_2/*.laz'
path_out = 'D:/Data/pointclouds/2023/all/flight_2/tile_index.shp'

path_in = 'D:/Data/pointclouds/2023/all/merged/*.las'
path_out = 'D:/Data/pointclouds/2023/all/merged/tile_index.shp'


#%% input files   

files_in = glob.glob(path_in)

gdf = gpd.GeoDataFrame() 

n = len(files_in)

for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    try:
      # pc = laspy.read(fpath)
      pc = laspy.open(fpath) # faster to read header only
    except:
      print("Error when reading file. Skipping to next.")
      continue
    
    # file name
    fname = Path(fpath).stem
    
    rows_list = []
    rows_list.append({"tileid": fname})
    df = pd.DataFrame(rows_list)
    
    # read bounding box
    x_bbox = np.array([pc.header.x_min, pc.header.x_max, pc.header.x_max, pc.header.x_min, pc.header.x_min])
    y_bbox = np.array([pc.header.y_max, pc.header.y_max, pc.header.y_min, pc.header.y_min, pc.header.y_max])
    
    # scale, round, unscale
    # x_bbox = np.round(x_bbox / 500.0) * 500.0
    # y_bbox = np.round(y_bbox / 500.0) * 500.0

    polygon_geom = Polygon(zip(x_bbox, y_bbox))
    polygon = gpd.GeoDataFrame(df, crs='epsg:2056', geometry=[polygon_geom]) 

    # append
    gdf = pd.concat([gdf, polygon])


gdf.to_file(filename=path_out, driver="ESRI Shapefile")





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST



df = pd.DataFrame()



new_row = {'Courses':'Hyperion', 'Fee':24000, 'Duration':'55days', 'Discount':1800}

new_row = {"tileid": "2496000_1116000"}
df2 = df.append(new_row, ignore_index=True)


df = pd.DataFrame(
    {
        "tileid": ["2496000_1116000", ],
    }
)

polygon_geom = Polygon(zip(x_bbox, y_bbox))
polygon = gpd.GeoDataFrame(df, crs='epsg:2056', geometry=[polygon_geom]) 

polygon.to_file(filename='D:/LiDAR/lidar2023GE/LAS_classifie/2496000_1116000.shp', driver="ESRI Shapefile")


gpd[]


# define schema
schema = {
    'geometry':'Polygon',
    'properties':[('Name','str')]
}

#open a fiona object
polyShp = fiona.open('../ShapeOut/cropPoly.shp', mode='w', driver='ESRI Shapefile',
          schema = schema, crs = "EPSG:2056")


#iterate over each row in the dataframe and save record
for index, row in pointDf.iterrows():
    rowDict = {
        'geometry' : {'type':'Point',
                     'coordinates': (row.X,row.Y)},
        'properties': {'Name' : row.Name},
    }
    pointShp.write(rowDict)
#close fiona object
pointShp.close()



inFile = File("./laspytest/data/simple.las", mode = "r")
 

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
