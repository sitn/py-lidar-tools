# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:46:54 2023

@author: ParkanM
"""

# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan
Last revision: May 24, 2023
Licence: GNU General Public Licence (GPL), see https://www.gnu.org/licenses/gpl.html
"""

# https://gis.stackexchange.com/questions/230574/inserting-lidar-points-from-laspy-in-geodataframe-without-using-a-numpy-array
# https://stackoverflow.com/questions/20474549/extract-points-coordinates-from-a-polygon-in-shapely
# https://stackoverflow.com/questions/31542843/inpolygon-examples-of-matplotlib-path-path-contains-points-method
# https://iotespresso.com/find-which-points-of-a-set-lie-within-a-polygon/

# pip install shapely


# Import libraries
import os
import shapefile
from pyproj import CRS
from pyproj.enums import WktVersion
import geopandas as gpd
from geopandas import GeoDataFrame

from pandas import DataFrame
import shapely
from shapely import Polygon
from shapely.geometry import Point
import matplotlib

import pathlib
from pathlib import Path

import laspy
import numpy as np
from scipy import spatial
import time


#%% parameters

fpath = 'C:/Projects/lidar_tree_detection/input/ne_2022/2552000_1201000.copc.las'


fpath = 'C:/Projects/lidar_tree_detection/input/ne_2016_boudry01_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/ne_2016_rochefort_ch1903p_survey_v220418.las'
fpath = 'C:/Projects/lidar_tree_detection/input/ne_2016_chambrelien_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/ge_2017_versoix_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/ne_2016_boudry19_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/ne_2016_boudry20e_ch1903p_survey.las'


# fname = os.path.basename(fpath)
fname = pathlib.Path(fpath).stem
# fext =
fpath_out = "C:/Projects/lidar_tree_detection/input/convex_hulls/%s_chull.shp" % (fname)


#%% read LAS file

pc = laspy.read(fpath)

print(pc.header.offset)
print(pc.header.scale)


#%% compute convex hull

xy = np.stack([pc.x, pc.y], axis=0).transpose((1, 0))
hull = spatial.ConvexHull(xy, qhull_options="Qt")
xy_chull = hull.points[hull.vertices,]


#%% write convex hull to Shapefile

w = shapefile.Writer(fpath_out)
w.field('id', 'N')
w.poly([xy_chull.tolist()])
w.record(1)
w.close()

# write the .prj file
crs = CRS.from_epsg(2056)
prj = open("C:/Projects/lidar_tree_detection/input/convex_hulls/%s_chull.prj" % (fname), "w")
# call the function and supply the epsg code

#print(crs.to_wkt(WktVersion.WKT1_GDAL, pretty=False))

#epsg = crs.to_wkt(pretty=True)
epsg = crs.to_wkt(WktVersion.WKT1_GDAL, pretty=False)
prj.write(epsg)
prj.close()


#%% check intersection

# tiles = geopandas.read_file("C:/Projects/lidar_tree_detection/input/ne_2022/tuiles_lidar_2022.shp", layer='tuiles_lidar_2022')
tiles = gpd.read_file("C:/Projects/lidar_tree_detection/input/ne_2022/tuiles_lidar_2022.shp")
chull = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry19_ch1903p_survey_chull.shp")


tiles_s = gpd.overlay(tiles, chull, how='intersection', keep_geom_type=None, make_valid=True)
# tiles_s = geopandas.sjoin(tiles, chull, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

points_gpd = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pc.x, pc.x), crs='epsg:2056') 

# point in polygon
pointInPolys  = gpd.sjoin(points_gpd, chull, how='left', predicate='within')

# clip

pnt_LA = points_gpd[pointInPolys.id==1]


tiles_s.tileid[0]
tiles_s.id[0]
tiles_s.geometry[0]
tiles_s.geometry[1]


# Import LAS into numpy array (X=raw integer value x=scaled float value)
lidar_points = np.array((pc.x,pc.y)).transpose()

# Transform to pandas DataFrame
lidar_df = DataFrame(lidar_points)

#Transform to geopandas GeoDataFrame
crs = None
geometry = [Point(xy) for xy in zip(pc.x, pc.x)]

points_geodf = GeoDataFrame(lidar_df, crs='epsg:2056', geometry=geometry)



pointInPolys  = gpd.sjoin(points_gpd, tiles, how='left', predicate='within')


# Example use: get points in Los Angeles, CA.
pnt_LA = points_gpd[pointInPolys.id==1]


xy_gp = geopandas.points_from_xy(pc.x, pc.x, z=None, crs='epsg:2056')

s = geopandas.GeoSeries([lidar_geodf])
chull2 = s.convex_hull


# gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['x'], df['y']))

# lidar_geodf.crs = 'epsg:2056' # set  spatial reference




#%% Point in polygon (Matplotlib)


fpath = []
fpath.append('C:/Projects/lidar_tree_detection/input/ne_2022/2552000_1201000.copc.las')
fpath.append('C:/Projects/lidar_tree_detection/input/ne_2022/2552000_1200500.copc.las')

# polygon
chull = gpd.read_file("C:/Projects/lidar_tree_detection/input/convex_hulls/ne_2016_boudry19_ch1903p_survey_chull.shp")

# plot
chull.geometry[0]


x_chull, y_chull = chull.geometry[0].exterior.coords.xy
# boundary = xy_chull.tolist()
boundary = np.dstack([x_chull, y_chull])[0].tolist()


# initialize empty LAS file
header = laspy.LasHeader(point_format=7, version="1.4")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
# header.scales = pc.header.scales
# header.offsets = pc.header.offsets
header.offsets = np.append(np.min(boundary, axis=0), 0.0)
header.scales = np.array([0.01, 0.01, 0.01])

las = laspy.LasData(header)
las.write("C:/Projects/lidar_tree_detection/input/final.las")


#######

path_p = matplotlib.path.Path(boundary)
    
pc1 = laspy.read(fpath[0])
pc2 = laspy.read(fpath[1])

idxl_1 = path_p.contains_points(np.stack([pc1.x, pc1.y], axis=0).transpose((1, 0)))
idxl_2 = path_p.contains_points(np.stack([pc2.x, pc2.y], axis=0).transpose((1, 0)))

pc1.points = pc1.points[idxl_1]
pc2.points = pc2.points[idxl_2]

header = laspy.LasHeader(point_format=7, version="1.4")
# header = laspy.LasHeader(point_format=3, version="1.2")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
header.offsets = np.append(np.min(boundary, axis=0), 0.0)
header.scales = np.array([0.01, 0.01, 0.01])

pc1.change_scaling(scales = header.scales, offsets = header.offsets)
pc2.change_scaling(scales = header.scales, offsets = header.offsets)


# 2. Create a Las
las = laspy.LasData(header)

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


for dimension in las.point_format.dimensions:
    las[dimension.name] = np.append(pc1[dimension.name], pc2[dimension.name])
    print(dimension.name)

# las.change_scaling(scales = header.scales, offsets = header.offsets)

las.write("C:/Projects/lidar_tree_detection/input/clip_temp.las")

#######





pc.X = np.append(pc.X, 0)

las.points = pc.points[idx_in].copy()


# append points

las.append_points(pc.points[idx_in].copy())






#######

pnts_array = pc.points.change_scaling(scales, offsets)

las = laspy.LasData(header)

las.points = pc.points[idx_in].copy()

offsets = np.append(np.min(boundary, axis=0), 0.0)
scales = np.array([0.01, 0.01, 0.01])
las.change_scaling(scales = scales, offsets = offsets)

las.write("C:/Projects/lidar_tree_detection/input/final.las")



header = laspy.LasHeader()
header.scales = np.array([0.1, 0.1, 0.1])
header.offsets = np.array([0, 0 ,0])
las = laspy.LasData(header=header)
las.x = [10.0]
las.y = [20.0]
las.z = [30.0]
las.change_scaling(scales=[0.01, 0.1, 0.1])

#######


# # read LAS file
pc = laspy.read(fpath[1])

pc.X[0] = 0


pc.X = np.append(pc.X, 0)


             
pc.X.append(0)

points.append_points()

# pc.header.offsets
# pc.header.scales

# # points
# xy = np.stack([pc.x, pc.y], axis=0).transpose((1, 0))
# points = list(xy)

# path_p = matplotlib.path.Path(boundary)
# idx_in = path_p.contains_points(points)

# pc.points[idx_in].copy()


# las = laspy.LasData(header)
# las.points = pc.points[idx_in].copy()
# las.write("C:/Projects/lidar_tree_detection/input/clip_test.las")


#######

def scaled_x_dimension(las_file):
    x_dimension = las_file.X
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (x_dimension * scale) + offset


def append_to_las(in_las, out_las):
    with laspy.open(out_las, mode='a') as outlas:
        with laspy.open(in_las) as inlas:
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points) 
                
                
def clip_las(in_las, clipper, las_header):
    # read LAS file
    pc = laspy.read(in_las)
    
    # points
    xy = np.stack([pc.x, pc.y], axis=0).transpose((1, 0))
    points = list(xy)
    
    path_p = matplotlib.path.Path(clipper)
    idx_in = path_p.contains_points(points)

    # write to temporary LAS file
    # las = laspy.LasData(las_header)
    header = laspy.LasHeader(point_format=7, version="1.4")
    # crs = CRS.from_epsg(2056)
    # header.add_crs(crs)
    # header.offsets = pc.header.offsets;
    # header.scales = pc.header.scales;
    
    las = laspy.LasData(las_header)
    
    las.points.change_scaling(las_header.scales, las_header.offsets)
    las.points = pc.points[idx_in].copy()
    
    las.write("C:/Projects/lidar_tree_detection/input/clip_temp.las")
    
    append_to_las("C:/Projects/lidar_tree_detection/input/clip_temp.las", "C:/Projects/lidar_tree_detection/input/final.las")
    

    
clip_las(fpath[0], boundary, header)    
clip_las(fpath[1], boundary, header)        


# read LAS file
pc = laspy.read(fpath)

# points
xy = np.stack([pc.x, pc.y], axis=0).transpose((1, 0))
points = list(xy)

# polygon
x_chull, y_chull = chull.geometry[0].exterior.coords.xy
# boundary = xy_chull.tolist()
boundary = np.dstack([x_chull, y_chull])[0].tolist()
# boundary = np.stack([x_chull, y_chull], axis=0).transpose((1, 0))


path_p = matplotlib.path.Path(boundary)

t1 = time.time()
idx_in = path_p.contains_points(points)
t2 = time.time()
print(t2-t1)


# 1. Create a new header
header = laspy.LasHeader(point_format=7, version="1.4")

crs = CRS.from_epsg(2056)
header.add_crs(crs)


# header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
# header.offsets = np.min(my_data, axis=0)
# header.scales = np.array([0.1, 0.1, 0.1])


# 2. Create a Las
las = laspy.LasData(header)

las.points = pc.points[idx_in].copy()


# append points

las.append_points(pc.points[idx_in].copy())

las.write("C:/Projects/lidar_tree_detection/input/clip_temp.las")




#%% Using LasWriter

# https://gis.stackexchange.com/questions/410809/append-las-files-using-laspy

           
            
            
            
            
# 1. Create a new header
header = laspy.LasHeader(point_format=7, version="1.4")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
# header.offsets = np.min(my_data, axis=0)
# header.scales = np.array([0.1, 0.1, 0.1])

# 3. Create a LasWriter and a point record, then write it
with laspy.open("C:/Projects/lidar_tree_detection/input/clip.las", mode="w", header=header) as outlas:
    outlas.append_points(pc.points[idx_in].copy())
    






in_laz = 
out_las = "C:/Projects/lidar_tree_detection/input/clip.las"   
  
append_to_las("C:/Projects/lidar_tree_detection/input/clip_temp.las", "C:/Projects/lidar_tree_detection/input/clip.las" )







    point_record.x = my_data[:, 0]
    point_record.y = my_data[:, 1]
    point_record.z = my_data[:, 2]

    writer.write_points(point_record)







new_las = laspy.LasData(pc.header)
new_las.points[idx_in].copy()
new_las.write("C:/Projects/lidar_tree_detection/input/clip.las")


with pylas.open('big.laz') as input_las:
    with pylas.open('ground.laz', mode="a") as ground_las:
        for points in input_las.chunk_iterator(2_000_000):
            ground_las.append_points(points[points.classification == 2])




# 0. Creating some dummy data
my_data_xx, my_data_yy = np.meshgrid(np.linspace(-20, 20, 15), np.linspace(-20, 20, 15))
my_data_zz = my_data_xx ** 2 + 0.25 * my_data_yy ** 2
my_data = np.hstack((my_data_xx.reshape((-1, 1)), my_data_yy.reshape((-1, 1)), my_data_zz.reshape((-1, 1))))



pc.points

las.x = pc.x
las.y = my_data[:, 1]
las.z = my_data[:, 2]
las.random = np.random.randint(-1503, 6546, len(las.points), np.int32)

las.write("new_file.las")




#%% Point in polygon (Shapely)

# p = Polygon(xy_chull.tolist())
# boundary = list(p.exterior.coords)




shapely.intersects(p, p)

path_p = matplotlib.path.Path([(0,0), (0, 1), (1, 1), (1, 0)])

boundary = p.boundary






path_p = matplotlib.path.Path(p.boundary)


points_within_p_3 = np.array(points)[inside_points]



# point in polygon 


