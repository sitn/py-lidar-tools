# -*- coding: utf-8 -*-
"""
Author: Matthew Parkan, SITN
Last revision: July 19, 2023
Licence: GNU General Public Licence (GPL), see https://www.gnu.org/licenses/gpl.html
Instructions:
1. Set the working directory
2. Set the path to the input labelled LAS file
3. Run the sections of the script you need
"""


#%% Import libraries

# from osgeo import gdal
import os
from pathlib import Path
import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


#%% parameters

# set working directory
os.chdir('D:/Projects/lidar_tree_detection/scripts/')

# set path to input LAS file
fpath_in = './data/input/ne_2016_boudry20e_ch1903p_survey.las'

# file name
fname = Path(fpath_in).stem

# path to output LAS file (labelled points only)
fpath_out_pc_subset = "./data/output/%s_sub.las" % (fname)

# path to output SHP file (with segmentation instances attributes)
fpath_out_tree_trunks = "./data/output/%s_trees.shp" % (fname)


#%% read LAS file

pc = laspy.read(fpath_in)

# print a few header attributes
print('Point format:', pc.header.point_format)
print('Points from Header:', pc.header.point_count)
print('File source ID:', pc.header.file_source_id)
print('UUID:', pc.header.uuid)
print('Generating software:', pc.header.generating_software)
print('Number of EVLR:', pc.header.number_of_evlrs)

# apply scale and offset
x_s = pc.x * pc.header.scales[0] + pc.header.offsets[0]
y_s = pc.y * pc.header.scales[1] + pc.header.offsets[1]
z_s = pc.z * pc.header.scales[2] + pc.header.offsets[2]

xyz = np.stack([pc.x, pc.y, pc.z], axis=0).transpose((1, 0))

n = len(pc.x)

intensity = pc.intensity
luid = pc.luid


#%% visualize by acquisition RGB

rgb = np.stack([pc.red, pc.green, pc.blue], axis=0).transpose((1, 0)) / 65535

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.visualization.draw_geometries([pcd])


#%% visualize by labelled (red) vs unlabelled (blue)

# set colors
rgb_lab = np.zeros((n, 3))
idx = luid != 0
rgb_lab[np.invert(idx)] = [0,0,1]
rgb_lab[idx] = [1,0,0]

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb_lab)
o3d.visualization.draw_geometries([pcd])


#%% visualize by segmentation

# set colors
ncolors = 12
cmap = np.asarray(plt.get_cmap("tab20").colors)
idxn_col = np.array(luid % ncolors, dtype=np.uint8)
rgb_seg = cmap[idxn_col,]
rgb_seg[luid == 0] = [0,0,0]

# plot
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb_seg)
o3d.visualization.draw_geometries([pcd])


#%% visualize a single instance

k = 2 # instance index

# points subset
idxl_sample = luid == k
xyz_s = xyz[idxl_sample,]
rgb_s = rgb[idxl_sample,]

pcd_s  = o3d.geometry.PointCloud()
pcd_s.points = o3d.utility.Vector3dVector(xyz_s)
pcd_s.colors = o3d.utility.Vector3dVector(rgb_s)
# pcd.normals = o3d.utility.Vector3dVector(normals)

o3d.visualization.draw_geometries([pcd_s])


#%% export labelled points to LAS file

new_file = laspy.create(point_format=pc.header.point_format, file_version=pc.header.version)

idxl_1 = pc.luid != 0
idxl_2 = np.isin(pc.classification, [2])
idxl = np.any([idxl_1, idxl_2], axis=0)

new_file.points = pc.points[idxl]

new_file.vlrs
new_file.write(fpath_out_pc_subset)


#%% get data from extended variable length records (EVLR)

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


#%%  export segmentation instances attributes to ESRI shapefile 

# create dictionnary
d = {}
d['luid'] = getEVLR(pc.header.evlrs, 5000, [])  # LUID
d['uuid'] = getEVLR(pc.header.evlrs, 5001, []) # UUID
d['x'] = getEVLR(pc.header.evlrs, 5010, []) # X
d['y'] = getEVLR(pc.header.evlrs, 5011, []) # Y 
d['z'] = getEVLR(pc.header.evlrs, 5012, []) # Z
d['proxy'] = getEVLR(pc.header.evlrs, 5013, []) # Location Proxy
d['diameter'] = getEVLR(pc.header.evlrs, 5014, []) # Diameter
d['height'] = getEVLR(pc.header.evlrs, 5015, []) # Total height
d['area'] = getEVLR(pc.header.evlrs, 5019, []) # Total projected area
d['volume'] = getEVLR(pc.header.evlrs, 5020, []) # Total volume
d['ipni'] = getEVLR(pc.header.evlrs, 5041, []) # IPNI
d['ivy'] = getEVLR(pc.header.evlrs, 5043, []) # Ivy carrier
d['dead'] = getEVLR(pc.header.evlrs, 5044, []) # Standing dead
d['ambiguous'] = getEVLR(pc.header.evlrs, 5061, []) # Ambiguity
d['timeStamp'] = getEVLR(pc.header.evlrs, 5062, []) # TimeStamp
                   
# create dataframe from dictionnary
df = pd.DataFrame(data=d)

# create geodataframe from dataframe
gdf = gpd.GeoDataFrame(df, crs='epsg:2056', geometry=gpd.points_from_xy(df.x, df.y, z=df.z, crs='epsg:2056'))

# write geodataframe to ESRI shapefile
gdf.to_file(filename=fpath_out_tree_trunks, driver="ESRI Shapefile")
