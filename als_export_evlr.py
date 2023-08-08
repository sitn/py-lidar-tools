# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan
Last revision: July 19, 2023
Licence: GNU General Public Licence (GPL), see https://www.gnu.org/licenses/gpl.html
"""

# pip install GDAL
# pip install laspy
# pip install open3d

#%% Import libraries

# from osgeo import gdal
import shapefile
import os
from pathlib import Path
import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from sklearn import metrics
from scipy import spatial


#%% parameters

fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_rochefort_ch1903p_survey_v220418.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_chambrelien_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ge_2017_versoix_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry01_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry19_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry20e_ch1903p_survey.las'


# fname = os.path.basename(fpath)
fname = Path(fpath).stem
# fext =
fpath_out_pc_subset = "D:/Projects/lidar_tree_detection/input/pointclouds_subsets/%s_sub.las" % (fname)
fpath_out_tree_trunks = "D:/Projects/lidar_tree_detection/input/tree_trunks/%s_tree_trunks.shp" % (fname)



#%% read LAS file

pc = laspy.read(fpath)

print('Point format:', pc.header.point_format)
print('Points from Header:', pc.header.point_count)
print('File source ID:', pc.header.file_source_id)
print('UUID:', pc.header.uuid)
print('Generating software:', pc.header.generating_software)
print('Extra VLR bytes:', pc.header.extra_vlr_bytes)
print('Number of EVLR:', pc.header.number_of_evlrs)

# apply scale and offset
x_s = pc.x * pc.header.scales[0] + pc.header.offsets[0]
y_s = pc.y * pc.header.scales[1] + pc.header.offsets[1]
z_s = pc.z * pc.header.scales[2] + pc.header.offsets[2]
intensity = pc.intensity
luid = pc.luid


# list(pc.point_format.dimension_names)
# set(list(pc.classification))


#%% write only labelled points to LAS file

new_file = laspy.create(point_format=pc.header.point_format, file_version=pc.header.version)

idxl_1 = pc.luid != 0
idxl_2 = np.isin(pc.classification, [2])
idxl = np.any([idxl_1, idxl_2], axis=0)

new_file.points = pc.points[idxl]

new_file.vlrs
new_file.write(fpath_out_pc_subset)


#%% visualization

"""
When this code runs,
you will see Hello World! 
in the console.
"""

n = len(pc.x)
xyz = np.stack([pc.x, pc.y, pc.z], axis=0).transpose((1, 0))
xyz_s = np.stack([x_s, y_s, z_s], axis=0).transpose((1, 0))

           
#%% visualize by acquisition RGB

rgb = np.stack([pc.red, pc.green, pc.blue], axis=0).transpose((1, 0)) / 65535

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])


#%% visualize by labelled/unlabelled

# set colors
rgb = np.zeros((n, 3))
idx = luid != 0
rgb[np.invert(idx)] = [0,0,1]
rgb[idx] = [1,0,0]

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])


#%% visualize by segmentation

# set colors
ncolors = 12
cmap = np.asarray(plt.get_cmap("tab20").colors)
idxn_col = np.array(luid % ncolors, dtype=np.uint8)
rgb = cmap[idxn_col,]
rgb[luid == 0] = [0,0,0]

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])


#%% visualize a single instance

k = 22

# create sample points index
idxl_sample = luid == k

xyz_s = xyz[idxl_sample,]
rgb_s = rgb[idxl_sample,]


pcd_s  = o3d.geometry.PointCloud()
pcd_s.points = o3d.utility.Vector3dVector(xyz_s)
pcd_s.colors = o3d.utility.Vector3dVector(rgb_s/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)

o3d.visualization.draw_geometries([pcd_s])


#%% compute convex hull of labelled subset

xy = xyz_s[:, [0,1]]

xy = xyz[:, [0,1]]
hull = spatial.ConvexHull(xy, qhull_options="Qt")
hull_indices = hull.vertices

xy_chull = hull.points[hull.vertices,]


w = shapefile.Writer('C:/Projects/lidar_tree_detection/chull.shp')
w.field('id', 'N')

# bb = [[113,24], [112,32], [117,36], [122,37], [118,20]]
# cc = hull.points.tolist()
w.poly([xy_chull.tolist()])

w.record(1)

w.close()


#%%  extract attributes from EVLR

# create dictionnary
d = {}
d['luid'] = getEVLR(pc.header.evlrs, 5000, [])
d['uuid'] = getEVLR(pc.header.evlrs, 5001, [])
d['x'] = getEVLR(pc.header.evlrs, 5010, [])
d['y'] = getEVLR(pc.header.evlrs, 5011, [])
d['z'] = getEVLR(pc.header.evlrs, 5012, [])
d['proxy'] = getEVLR(pc.header.evlrs, 5013, [])
d['diameter'] = getEVLR(pc.header.evlrs, 5014, [])
d['height'] = getEVLR(pc.header.evlrs, 5015, [])
d['area'] = getEVLR(pc.header.evlrs, 5019, [])
d['volume'] = getEVLR(pc.header.evlrs, 5020, [])
d['ipni'] = getEVLR(pc.header.evlrs, 5041, [])
d['ivy'] = getEVLR(pc.header.evlrs, 5043, [])
d['ivy'] = getEVLR(pc.header.evlrs, 5043, [])
d['dead'] = getEVLR(pc.header.evlrs, 5044, [])
d['ambiguous'] = getEVLR(pc.header.evlrs, 5061, [])
d['timeStamp'] = getEVLR(pc.header.evlrs, 5062, [])
                   
# create dataframe from dictionnary
df = pd.DataFrame(data=d)

# create geodataframe from dataframe
gdf = gpd.GeoDataFrame(df, crs='epsg:2056', geometry=gpd.points_from_xy(df.x, df.y, z=df.z, crs='epsg:2056'))

# write geodataframe to ESRI shapefile
gdf.to_file(filename=fpath_out_tree_trunks, driver="ESRI Shapefile")


#%% get data from extended variable length records

def getEVLR(evlr, record_id, luid):
    """
    Returns the specified EVLR record ID (second argument) 
    value(s) for the specified tree identifier(s) (third argument) 
    """

    format = dict([(5000, 'float32'), # LUID
                  (5001, 'S32'), # UUID
                  (5010, 'float64'), # X
                  (5011, 'float64'), # Y 
                  (5012, 'float64'), # Z
                  (5013, 'B'), # Location Proxy
                  (5014, 'uint16'), # Diameter
                  (5015, 'uint16'), # Total height
                  (5019, 'float32'), # Total projected area
                  (5020, 'float32'), # Total volume
                  (5041, 'S12'), # IPNI
                  (5043, 'B'), # Ivy carrier
                  (5044, 'B'), # Standing dead
                  (5061, 'B'), # Ambiguity
                  (5062, 'float64'), # TimeStamp
                  ])

    record_list = list(map(lambda x: x.record_id, evlr))
    
    
    if any(format[record_id] in s for s in {'S32', 'S12'}):
        
        b = list(evlr[record_list.index(record_id)].record_data)
        idxn_sort = np.squeeze(np.arange(len(b)).reshape(-1,len(b)//np.dtype(format[record_id]).itemsize).transpose().reshape(1,-1))
        b = [b[i] for i in idxn_sort]
        v = np.frombuffer(bytes(b), dtype=format[record_id], count=-1, offset=0).astype('U')
        
    else:
    
        v = np.frombuffer(evlr[record_list.index(record_id)].record_data, dtype=format[record_id], count=-1, offset=0)
        
    # filter output    
    if luid:
        
        luid_list = np.frombuffer(evlr[record_list.index(5000)].record_data, dtype=format[5000], count=-1, offset=0)
        idxl_luid = np.fromiter((i in luid for i in luid_list), 'bool')
        v = v[idxl_luid] 
    
    return (v)











#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DRAFT





#%% read  Variable Lenght Records (VLR)

from laspy.vlrs.known import ExtraBytesVlr, WktCoordinateSystemVlr

vlr = pc.vlrs.index(1)

pc.vlrs.vlr.description


pc.vlrs.get("WktCoordinateSystemVlr")
pc.vlrs.get("GeoKeyDirectoryVlr")

pc.vlrs.get("GeoDoubleParamsVlr")
pc.vlrs.get("GeoAsciiParamsVlr")
pc.vlrs.get("ExtraBytesVlr")


pc.vlrs.get_by_id(ExtraBytesVlr.official_user_id())[0]

pc.vlrs.get_by_id(ExtraBytesVlr.official_user_id())
pc.vlrs.get_by_id(ExtraBytesVlr.record_data_bytes())


new_evlr = laspy.header.EVLR(user_id = 10, record_id = 2, VLR_body = "Lots of data can go here.")



#%% 


k = 42

n_obj = len(getEVLR(pc.header.evlrs, 5000, []))

getEVLR(pc.header.evlrs, 5000, []) # LUID
getEVLR(pc.header.evlrs, 5001, []) # UUID
evrl_x = getEVLR(pc.header.evlrs, 5010, []) # X
evrl_y = getEVLR(pc.header.evlrs, 5011, []) # Y
getEVLR(pc.header.evlrs, 5012, []) # Z
getEVLR(pc.header.evlrs, 5013, []) # Location Proxy
getEVLR(pc.header.evlrs, 5014, []) # Diameter
getEVLR(pc.header.evlrs, 5015, []) # Total height
getEVLR(pc.header.evlrs, 5019, []) # Total projected area
getEVLR(pc.header.evlrs, 5020, []) # Total volume
getEVLR(pc.header.evlrs, 5041, []) # IPNI
getEVLR(pc.header.evlrs, 5043, []) # Ivy carrier
getEVLR(pc.header.evlrs, 5044, []) # Standing dead
getEVLR(pc.header.evlrs, 5061, []) # Ambiguity
getEVLR(pc.header.evlrs, 5062, []) # TimeStamp


rows_list = []
rows_list.append({
    "luid": getEVLR(pc.header.evlrs, 5000, []),
    "uuid": getEVLR(pc.header.evlrs, 5001, []),
    "ipni": getEVLR(pc.header.evlrs, 5041, [])
                  })

d = {}
d.con

d = {
     'luid': getEVLR(pc.header.evlrs, 5000, []), 
     'uuid': getEVLR(pc.header.evlrs, 5001, [])
     }








df = pd.DataFrame(rows_list)
    
    
# initialize empty geodataframe
gdf = gpd.GeoDataFrame()


geom = gpd.points_from_xy(evrl_x, evrl_y, z=None, crs='epsg:2056')




polygon_geom = Polygon(zip(x_bbox, y_bbox))


polygon = gpd.GeoDataFrame(df, crs='epsg:2056', geometry=[geom]) 

# append
gdf = pd.concat([gdf, polygon])




#%%  write segment data to shapefile


w = shapefile.Writer('C:/Projects/lidar_tree_detection/test.shp')
w.field('TEXT', 'C')
w.field('SHORT_TEXT', 'C', size=5)
w.field('LOWPREC', 'N', decimal=2)
w.field('LONG_TEXT', 'C', size=250)

# w.null() # no geometry

w.point(122, 37) 
w.record('Hello', 'World', 123.23, 'World'*50)

w.point(321, 17) 
w.record('Hello', 'World', 123.23, 'World'*50)

for i in range(0, 10):
  print(x)
  
  
w.close()



def scaled_x_dimension(las_file):
    x_dimension = las_file.X
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (x_dimension * scale) + offset

scaled_x = scaled_x_dimension(las)



#%% laspy.vlrs.known module

laspy.vlrs.known.WktCoordinateSystemVlr.official_user_id()


laspy.vlrs.known.WktCoordinateSystemVlr.parse_crs()

# GeoDoubleParamsVlr
laspy.vlrs.known.GeoDoubleParamsVlr.official_user_id()
laspy.vlrs.known.GeoDoubleParamsVlr.official_record_ids()

laspy.vlrs.known.GeoDoubleParamsVlr.parse_record_data()




laspy.vlrs.known.ExtraBytesVlr.official_user_id()




#%% read Extended Variable Lenght Records (EVLR)

len(set(pc.luid))

pc.header.number_of_evlrs

evlr = pc.evlrs

pc.header.evlrs[0]

format = dict([(5000, 'float32'), # LUID
              (5001, 'S32'), # UUID
              (5010, 'float64'), # X
              (5011, 'float64'), # Y 
              (5012, 'float64'), # Z
              (5013, 'B'), # Location Proxy
              (5014, 'uint16'), # Diameter
              (5015, 'uint16'), # Total height
              (5019, 'float32'), # Total projected area
              (5020, 'float32'), # Total volume
              (5041, 'S12'), # IPNI
              (5043, 'B'), # Ivy carrier
              (5044, 'B'), # Standing dead
              (5061, 'B'), # Ambiguity
              (5062, 'float64'), # TimeStamp
              ])

record_list = list(map(lambda x: x.record_id, pc.header.evlrs))

record_id = 5001

evlr[record_list.index(record_id)].record_id
evlr[record_list.index(record_id)].user_id
evlr[record_list.index(record_id)].description

evlr[record_list.index(record_id)].record_data_bytes()
evlr[record_list.index(record_id)].record_data


b = list(evlr[record_list.index(record_id)].record_data)
idxn_sort = np.squeeze(np.arange(len(b)).reshape(-1,len(b)//np.dtype(format[record_id]).itemsize).transpose().reshape(1,-1))
b = [b[i] for i in idxn_sort]
v = np.frombuffer(bytes(b), dtype=format[record_id], count=-1, offset=0).astype('U')
        
        
b = list(evlr[record_list.index(record_id)].record_data)
v = np.frombuffer(bytes(b), dtype=format[record_id], count=-1, offset=0)

.astype('U')



idxn_sort = np.squeeze(np.arange(len(b)).reshape(-1,len(b)//np.dtype(format[record_id]).itemsize).transpose().reshape(1,-1))
b = [b[i] for i in idxn_sort]

v = np.frombuffer(bytes(b), dtype=format[record_id], count=-1, offset=0)

v = np.frombuffer(bytes(b), dtype=format[record_id], count=-1, offset=0).astype('U')
        
        
        
        
        
        
b = list(evlr[record_list.index(record_id)]. VLR_body)
b = list(evlr[record_list.index(record_id)].record_id)

pc.header.evlrs[0]


######

c1 = np.logical_and(aa, bb)
c2 = aa * bb
c3 = aa & bb
c4 = np.and([aa, bb], axis=0)

c1 = np.logical_or(aa, bb)
c3 = aa | bb


np.array_equiv(c1,c4) 



labels 
colors = plt.get_cmap("tab20")(labels.numpy() / (max_label if max_label > 0 else 1))


pcd  = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)

pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))


# downsample pointcloud
voxel_size = 0.25
pcd_down = pcd.voxel_down_sample(voxel_size)

radius_normal = voxel_size * 2
print(":: Estimate normal with search radius %.3f." % radius_normal)
pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))


o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))


KD = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30)
o3d.geometry.estimate_normals(pcd, search_param = KD)

normals = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30), fast_normal_computation=True)

pcd.has_normals()

pcd.normals[0]

pcd.points = o3d.utility.Vector3dVector(xyz)



pcd.paint_uniform_color([1, 0.706, 0])

pcd.colors = o3d.utility.Vector3dVector(rgb/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)


o3d.visualization.draw_geometries([pcd])


#%% clustering test

labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
labels = np.array(pcd.cluster_dbscan(eps=1, min_points=100, print_progress=True))
    