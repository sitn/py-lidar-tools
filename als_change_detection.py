# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:37:40 2023

@author: Matthew Parkan, SITN, matthew.parkan@ne.ch
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
# tile_index = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp'
# dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/*.las'
# dir_ref_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/'
# dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/'


# directories/files
tile_index = 'D:/Projects/intemperie_cdf_20230724/grid/tuiles_lidar_2022.shp'
dir_in = 'D:/Projects/intemperie_cdf_20230724/data/pointclouds/Flight_1_Geospatial_predict_all_classes_Flai_v2/*.laz'
dir_ref_in = 'D:/Data/LiDAR/2022/'
dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v2/'

# processing
target_classes = [2,3,4,5,31]
d_max = 1.5



# pc1 = laspy.read('D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v2/2545000_1211500_change.las')
# pc1.distance              

# pc2 = laspy.read('D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v3/2544500_1211500_change.laz')
# pc2.distance         
   


#%% change detection in point cloud


files_in = glob.glob(dir_in)


# fpath = files_in[42]

# del files_in[0] 
# files_in = ['D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/LCDF_LV95_NF02_000011.las']
# files_ref_in = glob.glob(dir_ref_in)
# files_out = glob.glob(dir_out + '/*.las')

n = len(files_in)

# tiles_s = gpd.overlay(tiles, clipper, how='intersection', keep_geom_type=None, make_valid=True)
tiles = gpd.read_file(tile_index, crs='epsg:2056')
    
for index, fpath in enumerate(files_in):

    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    if not os.path.isfile(fpath):
        print("File not found, skipping to next iteration")
        continue
    
    # read LAS files
    print("Reading %s" % (fpath))
    
    pc = laspy.read(fpath)
    
    # tile center
    x_c = (pc.header.x_min + pc.header.x_max) / 2
    y_c = (pc.header.y_min + pc.header.y_max) / 2

    # match tile
    points_gpd = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x_c], [y_c]), crs='epsg:2056') 
    
    feature = gpd.sjoin(tiles, points_gpd, predicate='contains')

    if len(feature) == 0:
        print("Reference file not found, skipping to next iteration")
        continue
    
    xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()
    
    # fname = os.path.basename(fpath)
    # fname = pathlib.Path(fpath).stem
    fname = feature.tileid.values[0]
    fpath_ref = dir_ref_in + fname + '.copc.laz'
    
    if not os.path.isfile(fpath_ref):
        print("File not found, skipping to next iteration")
        continue
    
    print("Reading reference %s" % (fpath_ref))
    pc_ref = laspy.read(fpath_ref)
    xyz_ref = np.vstack((pc_ref.x, pc_ref.y, pc_ref.z)).transpose()
    
    # build KDTree
    print("Building KDTree")
    tree = pykdt.KDTree(xyz, leafsize=24)
    
    # find neighbours
    print("Searching nearest neighbours")
    knn_dist, knn_ind = tree.query(xyz_ref, k=1)
    
    # assign change flag=1 to points beyond max range
    idxl_change = (knn_dist > d_max) & np.isin(pc_ref.classification, [4,5])
    idxl_target = np.isin(pc_ref.classification, target_classes)
    
    # pc_ref.points.classification[np.invert(idxl_change)] = 0
    # pc_ref.points.classification[idxl_change] = 5
    # pc_ref.points.user_data = idxl_change.astype('uint8')
    
    # write result to LAS file
    fpath_las_out = dir_out + fname + '_change.las'
    print("Writing result to %s" % (fpath_las_out))
    
    # add extra dimension
    pc_ref.add_extra_dim(laspy.ExtraBytesParams(
        name = "distance",
        type = np.float32, # np.int32, # "float32",
        description = "Distance to reference points",
        offsets = None, # np.array([0.0]),
        scales = None  # np.array([100.0])
    ))
    
    
    # las.random = np.random.randint(-1503, 6546, len(las.points), np.int32)
    pc_ref.distance = np.float32(knn_dist)
    
    np.max(pc_ref.distance)
    
    # remove COPC VLRS/EVLRS
    pc_ref.header.vlrs.pop(0)
    # pc_ref.header.evlrs.pop(0)
    pc_ref.header.evlrs = []
    
    las = laspy.LasData(header=pc_ref.header, points=pc_ref.points[idxl_target])
    las.write(fpath_las_out)
    






pc3 = laspy.read('D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/LCDF_LV95_NF02_000043.las')
pc3.write('D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/test_laspy_LCDF_LV95_NF02_000043.las')


pc2 = laspy.read(fpath_las_out)

np.min(pc_ref.x)
np.max(pc_ref.x)


np.max(knn_dist)
np.max(pc_ref.distance)
np.max(pc2.distance)

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
    
    # get tile bounding box
    bbox = feature.geometry.bounds
    x_min = bbox.minx.values[0]
    x_max = bbox.maxx.values[0]
    y_min = bbox.miny.values[0]
    y_max = bbox.maxy.values[0]
    
    # create raster grid
    res = 1
    xe = np.arange(x_min, x_max + res, res)
    ye = np.arange(y_min, y_max + res, res)
    transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)

    # rasterize
    idxl_change = pc.user_data == 1
    ret = stats.binned_statistic_2d(pc.x[idxl_change], pc.y[idxl_change], None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
    npoints = np.uint32(np.rot90(ret.statistic)) > 20
    # mask = npoints.astype(np.uint8)

    # morphological filtering
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
    
    # export to geotiff 
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

# 1. Create a new header
header = laspy.LasHeader(pc_ref.header.point_format, pc_ref.header.version)
header.point_format = pc_ref.header.point_format
header.version = pc_ref.header.version


header = laspy.LasHeader(version=Version(1, 4), point_format=laspy.PointFormat(7))

header = laspy.LasHeader(point_format=7, version="1.4")



header = laspy.LasHeader(point_format=3, version="1.2")
header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
header.offsets = np.min(my_data, axis=0)
header.scales = np.array([0.1, 0.1, 0.1])

# 2. Create a Las
las = laspy.LasData(header)

las.x = my_data[:, 0]
las.y = my_data[:, 1]
las.z = my_data[:, 2]
las.random = np.random.randint(-1503, 6546, len(las.points), np.int32)

las.write("new_file.las")



###########


simple_las = "C:/Users/parkanm/Downloads/simple.las"

simple_las = "D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/LCDF_LV95_NF02_000043.las"

las = laspy.read(simple_las)

las.add_extra_dim(
    laspy.ExtraBytesParams(
        name="lol", 
        type="uint64",
        scales=np.array([2.0]), 
        offsets=np.array([0.0])
    )
)


las.add_extra_dim(
    laspy.ExtraBytesParams(
        name = "mydistance",
        type = "float32",
        description = "Distance to nearest neighbour",
        offsets = None,
        scales = None
    )
)


new_values = np.ones(len(las.points)) * 4
las.lol = new_values   








    
    
    

pc_ref = laspy.read(fpath_ref)

pc_out = laspy.LasData(pc_ref.header)
pc_out.header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))

pc_out.points = pc_ref.points

pc_out.random = np.random.randint(-1503, 6546, len(pc_out.points), np.int32)




fmt = pc_ref.header.point_format


pc_ref = laspy.read(fpath_ref)
    
pc_ref.header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.float32))

dd = np.float32(np.random.uniform(low=0.0, high=50, size=len(pc_ref.points)))

pc_ref.random = np.float32(np.random.uniform(low=0.0, high=50, size=len(pc_ref.points)))




pc_ref = laspy.read(fpath_ref)

pc_ref.header.add_extra_dim(laspy.ExtraBytesParams(name="bibi", type=np.int32))



pc_ref.bibi = np.random.randint(0, 6546, len(pc_ref.points), np.int32)

pc_ref.points.extra_dims
pc_ref.points




# fmt = laspy.PointFormat(6)
fmt = pc_ref.header.point_format
standard_dims = list(fmt.standard_dimensions)
extra_dims = list(fmt.extra_dimension_names)

dim = fmt.dimension_by_name("classification")
fmt.add_extra_dimension(laspy.ExtraBytesParams("distance", "float32"))

dim = fmt.dimension_by_name("distance")
fmt.num_standard_bytes
fmt.num_extra_bytes





pc.header


# 0. Creating some dummy data
my_data_xx, my_data_yy = np.meshgrid(np.linspace(-20, 20, 15), np.linspace(-20, 20, 15))
my_data_zz = my_data_xx ** 2 + 0.25 * my_data_yy ** 2
my_data = np.hstack((my_data_xx.reshape((-1, 1)), my_data_yy.reshape((-1, 1)), my_data_zz.reshape((-1, 1))))

# 1. Create a new header
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
header.offsets = np.min(my_data, axis=0)
header.scales = np.array([0.1, 0.1, 0.1])

# 2. Create a Las
las = laspy.LasData(header)

las.x = my_data[:, 0]
las.y = my_data[:, 1]
las.z = my_data[:, 2]
las.random = np.random.randint(-1503, 6546, len(las.points), np.int32)

las.write("new_file.las")











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
