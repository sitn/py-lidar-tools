# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:37:40 2023

@author: Matthew Parkan, SITN
"""

#%%  todo

# check growth effect on tile 005

#%% libraries 

import time
import os
import laspy
import numpy as np
import glob
import pathlib
import geopandas as gpd
from scipy import stats
from scipy import ndimage
from scipy import spatial
import open3d as o3d
import faiss
import rasterio
from rasterio.transform import Affine
import pykdtree.kdtree as pykdt
# from sklearn.neighbors import KDTree
# import cv2 as cv
# import matplotlib.pyplot as plt
# from skimage.morphology import reconstruction
# import skimage.measure

#%% parameters

# directories/files
tile_index = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp'
dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/*.las'
dir_ref_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/'
dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/'

# processing
target_classes = [2,3,4,5,31]
d_max = 1.5


#%% change detection in point cloud

files_in = glob.glob(dir_in)

# del files_in[0] 

# files_in = ['D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/LCDF_LV95_NF02_000011.las']
# files_ref_in = glob.glob(dir_ref_in)
# files_out = glob.glob(dir_out + '/*.las')

n = len(files_in)
    
for index, fpath in enumerate(files_in):

    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    if not os.path.isfile(fpath):
        print("File not found, skipping to next iteration")
        continue
    
    #%% read LAS files
    print("Reading %s" % (fpath))
    
    pc = laspy.read(fpath)
    xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()
    
    # fname = os.path.basename(fpath)
    fname = pathlib.Path(fpath).stem
    fpath_ref = dir_ref_in + fname + '.las'
    
    if not os.path.isfile(fpath_ref):
        print("File not found, skipping to next iteration")
        continue
    
    print("Reading reference %s" % (fpath_ref))
    pc_ref = laspy.read(fpath_ref)
    xyz_ref = np.vstack((pc_ref.x, pc_ref.y, pc_ref.z)).transpose()
    
    
    #%% change detection
    print("Building KDTree")
    
    # build KDTree
    # tree = spatial.KDTree(xyz, leafsize=24)
    tree = pykdt.KDTree(xyz, leafsize=24)
    
    # find neighbours
    print("Searching nearest neighbours")
    knn_dist, knn_ind = tree.query(xyz_ref, k=1)
    
    # assign change flag=1 to points beyond max range
    idxl_change = (knn_dist > d_max) & np.isin(pc_ref.classification, [4,5])
    idxl_target = np.isin(pc_ref.classification, target_classes)
    
    pc_ref.points.classification[np.invert(idxl_change)] = 0
    pc_ref.points.classification[idxl_change] = 5
    
    pc_ref.points.user_data = idxl_change.astype('uint8')
    
    #%% write result to LAS file
    fpath_las_out = dir_out + fname + '_change.las'
    print("Writing result to %s" % (fpath_las_out))
    las = laspy.LasData(header=pc_ref.header, points=pc_ref.points[idxl_target])
    las.write(fpath_las_out)


#%% rasterize point clouds

tiles = gpd.read_file('D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp', crs='epsg:2056')
files_in = glob.glob('D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/*.las')

del files_in[0] 

# fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/LCDF_LV95_NF02_000011_change.las'

n = len(files_in)

for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    # read LAS files
    pc = laspy.read(fpath)
    
    # tile center
    x_c = (pc.header.x_min + pc.header.x_max) / 2
    y_c = (pc.header.y_min + pc.header.y_max) / 2

    # match tile
    points_gpd = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x_c], [y_c]), crs='epsg:2056') 
    
    feature = gpd.sjoin(tiles, points_gpd, predicate='contains')

    if len(feature) == 0:
        print("File not found, skipping to next iteration")
        continue
    
    #%% get tile bounding box
    bbox = feature.geometry.bounds
    x_min = bbox.minx.values[0]
    x_max = bbox.maxx.values[0]
    y_min = bbox.miny.values[0]
    y_max = bbox.maxy.values[0]
    
    #%% create raster grid
    res = 1
    xe = np.arange(x_min, x_max + res, res)
    ye = np.arange(y_min, y_max + res, res)
    transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)

    #%% rasterize
    
    # idxl_change = pc.classification == 5
    idxl_change = pc.user_data == 1
    
    ret = stats.binned_statistic_2d(pc.x[idxl_change], pc.y[idxl_change], None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
    npoints = np.uint32(np.rot90(ret.statistic)) > 20
    # mask = npoints.astype(np.uint8)

    #%% morphological opening

    kernel = np.ones((3,3), bool)
    
    # mask = ndimage.binary_opening(npoints, structure=kernel, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False)
    # imgplot = plt.imshow(mask)
    
    # morphological erosion
    seeds = ndimage.binary_erosion(npoints, structure=kernel, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False)
    
    # find connected components 
    labels, n_objects = ndimage.label(npoints)
    mask = np.isin(labels, np.unique(labels[seeds]))
    
    #np.count_nonzero(mask)
    
    # plot
    #imgplot = plt.imshow(seeds)
    #imgplot = plt.imshow(mask)
    
    #%% export to geotiff
        
    fout = rasterio.open(
            dir_out + feature.tileid.values[0] + '_change.tif',
            'w',
            driver = 'GTiff',
            height = mask.shape[0],
            width = mask.shape[1],
            count = 1,
            dtype = np.uint8,
            nbits=1,
            crs = 'EPSG:2056',
            transform = transform_edges,
        )
        
    fout.write(mask, 1)
    fout.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% DRAFT


# leafsize=3, build=19s, query= 115s
# leafsize=6, build=18s, query= 83s
# leafsize=12, build=16s, query= 67s
# leafsize=24, build=15, query= 62s

# pykdtree
tic = time.perf_counter()
kd_tree = pykdt.KDTree(xyz, 24)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

tic = time.perf_counter()
knn_dist4, knn_ind4 = kd_tree.query(xyz_ref, k=1)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")


# scipy
tic = time.perf_counter()
tree1 = spatial.KDTree(xyz, leafsize=24)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

tic = time.perf_counter()
knn_dist1, knn_ind1 = tree1.query(xyz_ref, k=1)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")


# sklearn
tic = time.perf_counter()
tree2 = KDTree(xyz, leaf_size=24)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

tic = time.perf_counter()
knn_dist2, knn_ind2 = tree2.query(xyz_ref, k=1)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

# faiss

N = 1000000
d = 3
k = 1

xb = xyz.astype('float32')
xq = xyz_ref.astype('float32')


# create an array of N d-dimensional vectors (our search space)
#xb = np.random.random((N, d)).astype('float32')
# create a random d-dimensional query vector
#xq = np.random.random(d)

tic = time.perf_counter()
index = faiss.IndexFlatL2(xb.shape[1])
# index = faiss.index_factory(d, "Flat") 
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

index.add(xb)



index.train(xb)
index.add(xb)
distances, neighbors = index.search(xq.reshape(1,-1).astype(np.float32), k)

tic = time.perf_counter()
distances, neighbors = index.search(xq, k)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")


# open3d
pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

tic = time.perf_counter()
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

tic = time.perf_counter()
[knn_dist3, knn_ind3, _] = pcd_tree.search_knn_vector_3d(np.asarray(pcd.points), 1)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")
