# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:21:00 2023

@author: Matthew Parkan, SITN, matthew.parkan@ne.ch
"""

# INSTALLATION
# pip install numpy
# pip install laspy[lazrs,laszip]
# pip install cloth-simulation-filter

# REFERENCES: 
# https://github.com/jianboqi/CSF
# http://ramm.bnu.edu.cn/projects/CSF/
# https://search.r-project.org/CRAN/refmans/lidR/html/gnd_csf.html
# https://www.cloudcompare.org/doc/wiki/index.php/CSF_(plugin)

# coding: utf-8
import sys
import laspy
import CSF
import numpy as np
import glob
import pathlib
import re
import rasterio
from rasterio.transform import Affine
from rasterio.fill import fillnodata
from scipy import stats
from numpy_groupies import aggregate
import matplotlib.pyplot as plt
import pykdtree.kdtree as pykdt

#%% references

# https://github.com/ml31415/numpy-groupies/


#%% debugging

# stopped at file 2546500_1215500.laz


#%% parameters

# file paths
# \\nesitn5\geodata\pointclouds\Aeriallidar\Lidar2022_IGN\1_las\Ch1903plus_laz

# fpath_in = 'C:/Users/parkanm/Desktop/terrain_classification/tile1/tile1.laz'
# fpath_out = 'C:/Users/parkanm/Desktop/terrain_classification/tile1/tile1_classified_6.las'


# path = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022_IGN/1_las/Ch1903plus_laz/*.laz'
# files_in = glob.glob(path)

# dir_out = 'D:/LiDAR/lidar2022IGN/'
# files_out = glob.glob(dir_out + '/*.las')
# files_skip = ['2546500_1215500']

# path = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/LCDF_LV95_NF02_000052.las'
path = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/*.las' 
files_in = glob.glob(path)
skip_existing = False

dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/classified/'
files_out = glob.glob(dir_out + '/*.las')
files_skip = []

# CSF parameters
# more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

csf = CSF.CSF()
csf.params.bSloopSmooth = False # logical. When steep slopes exist, set this parameter to TRUE to reduce errors during post-processing.
csf.params.cloth_resolution = 0.5 # 0.2 scalar. The distance between particles in the cloth. This is usually set to the average distance of the points in the point cloud. The default value is 0.5.
csf.params.rigidness = 1; # 2 integer. The rigidness of the cloth. 1 stands for very soft (to fit rugged terrain), 2 stands for medium, and 3 stands for hard cloth (for flat terrain). The default is 1
csf.params.time_step = 0.65; # scalar. Time step when simulating the cloth under gravity. The default value is 0.65. Usually, there is no need to change this value. It is suitable for most cases.
csf.params.class_threshold = 0.5; # scalar. The distance to the simulated cloth to classify a point cloud into ground and non-ground. The default is 0.5.
csf.params.interations = 1000; # integer. Maximum iterations for simulating cloth. The default value is 500. Usually, there is no need to change this value.

#skip = any(list(map(lambda x: bool(re.search('2528000_1203500', x)), files_out)))
#skip = any(file == '2528000_1203500_classified.las' for file in files_out)
#bool(re.search('2528000_1203500', files_out))
#bool(re.search('2528000_1203500', '2528000_1203500_classified'))
#pathlib.Path(files_in[0]).stem
# any(list(map(lambda x: (fname + '_classified.las') in x, files_out)))
 
# fpath = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022_IGN/1_las/Ch1903plus_laz/2546500_1215500.laz'

#%% classify ground points

files_in = [files_in[42]]
n = len(files_in)

for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    # check if output file already exists
    fname = pathlib.Path(fpath).stem
    skip = any(list(map(lambda x: bool(re.search(fname, x)), files_out + files_skip)))  
    
    if skip & skip_existing:
        print("Already processed -> Skipping")
        continue
    
    # read LAS file
    pc = laspy.read(fpath)
    # points = pc.points
    # xyz2 = np.vstack((pc.x, pc.y, pc.z)).transpose()
    xyz = np.column_stack((pc.x, pc.y, pc.z))
    # (xyz == xyz2).all()
    
    n_pts = len(pc.points)

    # reset classification
    pc.classification = np.full(n_pts, 0, dtype=np.int8)
    
    # extract last returns
    idxl_last = pc.return_number == pc.number_of_returns
    idxn_last = np.where(idxl_last)[0]
    
    # xyz_last2 = np.vstack((pc.x[idxl_last], pc.y[idxl_last], pc.z[idxl_last])).transpose()
    xyz_last = xyz[idxl_last,:]
    # (xyz_last == xyz_last2).all()
    
    # terrain classification
    csf.setPointCloud(xyz_last)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.
    
    if len(ground) > 0:
    
        idxn_ground = idxn_last[np.array(ground)]
        pc.classification[idxn_ground] = 2
        # pc.classification[np.array(ground)] = 2
    
    # write classified point cloud to LAS file
    fpath_out = dir_out + fname + '_classified.las'
    outFile = laspy.LasData(pc.header)
    outFile.points = pc.points
    outFile.write(fpath_out)


#%% classify points below ground (low noise)

fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/classified/LCDF_LV95_NF02_000043_classified.las'
# parameters
class_terrain = 2
class_noise = 7
res = 1

# read LAS file
pc = laspy.read(fpath)

x = np.array(pc.x)
y = np.array(pc.y)
z = np.array(pc.z)

n_pts = len(pc.points)

# comput extent
x_min = np.min(pc.x)
x_max = np.max(pc.x)
y_min = np.min(pc.y)
y_max = np.max(pc.y)

# create raster grid
xe = np.arange(x_min, x_max + res, res)
ye = np.arange(y_min, y_max + res, res)

if xe[-1] == x_max:
    xe = np.append(xe, xe[-1]+res)

if ye[-1] == y_max:
    ye = np.append(ye, ye[-1]+res)

idxl_terr = np.isin(pc.classification, class_terrain)
ret_1 = stats.binned_statistic_2d(pc.x[idxl_terr], pc.y[idxl_terr], pc.z[idxl_terr], 'median', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
dtm = fillnodata(ret_1.statistic, mask=~np.isnan(ret_1.statistic), max_search_distance=250.0, smoothing_iterations=0)

##########################

# bin all points
ret_2 = stats.binned_statistic_2d(x, y, z, 'count', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
idxn_bin = ret_2.binnumber - 1
z_dtm = dtm[idxn_bin[0], idxn_bin[1]]
h = (z - z_dtm)
idxl_noise = h < -1
np.count_nonzero(idxl_noise)

##########################

# stat = np.float32(np.rot90(ret_1.statistic))
# dtm = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=250.0, smoothing_iterations=0)

# # map coordinates (x,y) to image (col,row) coordinates
# transform = Affine.translation(x_min, y_max) * Affine.scale(res, -res)
# transformer = rasterio.transform.AffineTransformer(transform)
# rc = transformer.rowcol(pc.x, pc.y)

# z_dtm = dtm[rc[0], rc[1]]
# h = (pc.z - z_dtm)
# idxl_noise = h < -1
# np.count_nonzero(idxl_noise)

##########################

pc.classification[idxl_noise] = class_noise

# write classified point cloud to LAS file
fname = pathlib.Path(fpath).stem
fpath_out = dir_out + fname + '_classified_2.las'
outFile = laspy.LasData(pc.header)
outFile.points = pc.points
outFile.write(fpath_out)


#%% classify isolated points (noise)


fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/classified/LCDF_LV95_NF02_000043_classified.las'
# parameters
class_terrain = 2
class_noise = 7
res = 1

# read LAS file
pc = laspy.read(fpath)

xyz = np.column_stack((pc.x, pc.y, pc.z))

# build KDTree
print("Building KDTree")
tree = pykdt.KDTree(xyz, leafsize=24)
    
# find neighbours
print("Searching nearest neighbours")
knn_dist, knn_ind = tree.query(xyz, k=40)

f = 10
d_mean = np.mean(knn_dist, axis=1)
# d_std = np.mean(knn_dist, axis=1) 
# d_max = d_mean + f * d_std


d_mean_global = np.mean(knn_dist)
d_std_global = np.std(knn_dist)
d_max_global = d_mean_global + f * d_std_global

idxl_outlier = d_mean > d_max_global


pc.classification[idxl_outlier] = class_noise

# write classified point cloud to LAS file
fname = pathlib.Path(fpath).stem
fpath_out = dir_out + fname + '_classified_3.las'
outFile = laspy.LasData(pc.header)
outFile.points = pc.points
outFile.write(fpath_out)


# The maximum distance will be: avg distance + m * std deviation.
# If quantile = TRUE, m becomes the quantile threshold


# simplify 
xyz_r = np.round(xyz)
u, idxn_u, count_u = np.unique(xyz_r, return_index=True, return_counts=True, axis=0)
# u = np.array([[1,2,3], [1,2,3], [3,4,3]])

a = np.array([1, 2, 3, 4])
np.add.at(a, [0, 1, 2, 2],1)



################################################################################################################################
#%% DRAFT


idxn_bin_1 = ret.binnumber - 1
z_terr_dtm = dtm[idxn_bin_1[0], idxn_bin_1[1]]
h_terr = z[idxl_terr] - z_terr_dtm
np.count_nonzero(h_terr < -0.5) 



np.min(h_terr)
np.max(h_terr)
np.mean(h_terr)
np.median(h_terr)

# bin all points
ret2 = stats.binned_statistic_2d(x, y, z, 'min', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
bibi = ret2.statistic 
idxn_bin = ret2.binnumber - 1
z_dtm = bibi[idxn_bin[0], idxn_bin[1]]
h = (z - z_dtm)

np.min(h)
np.max(h)
np.mean(h)
np.median(h)


z_dtm = dtm[idxn_bin[0], idxn_bin[1]]
h = (z - z_dtm)

idxl_noise = h < -0.5
np.count_nonzero(idxl_noise) 

bibi[0,0]


idxl_noise = (pc.z - z_dtm) < -1
pc.classification[idxl_noise] = class_noise

# write classified point cloud to LAS file
fpath_out = dir_out + fname + '_classified_2.las'
outFile = laspy.LasData(pc.header)
outFile.points = pc.points
outFile.write(fpath_out)


# 
nrows = len(ye)-1
ncols = len(xe)-1




# stat = np.float32(np.rot90(ret.statistic))
# filter points below ground



#%% draft

# https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html#affine
# https://github.com/rasterio/affine
# https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
# https://stackoverflow.com/questions/41914872/how-to-use-a-linear-index-to-access-a-2d-array-in-python
# https://stackoverflow.com/questions/14162026/how-to-get-the-values-from-a-numpy-array-using-multiple-indices

transformer = rasterio.transform.AffineTransformer(transform)

# [x, y] coordinates of center of upper left pixel 
transformer.xy(0, 0)

# [row, col] coordinates of center of upper left pixel 
transformer.rowcol(2546750.6, 1212200.0)

# image subscript [row, col] coordinates of points
rc = transformer.rowcol(pc.x, pc.y)
transformer.rowcol(x_min, y_max)

# image lindear [ind] coordinates of points
# np.ravel_multi_index(multi_index, dims, mode='raise', order='C')
ind = np.ravel_multi_index(rc, (nrows, ncols))

np.min(rc[0])
np.max(rc[0])
np.min(rc[1])
np.max(rc[1])

arr = np.array([[0,1,2],[0,1,1]])
np.ravel_multi_index(arr, (3,2), order='C')



#%% get image [row, col] coordinates of points directly from bin indices returned by binned_statistic_2d 

x = np.array([0.5, 0.5, 2.5])
y = np.array([0.5, 0.5, 1.5])
z = np.array([1, 1, 3])

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

res = 1
xe = np.arange(x_min, x_max+res, res)
ye = np.arange(y_min, y_max+res, res)

if xe[-1] == x_max:
    xe = np.append(xe, xe[-1]+res)

if ye[-1] == y_max:
    ye = np.append(ye, ye[-1]+res)

nrows = len(ye)-1
ncols = len(xe)-1

idxl_terr = [True, True, False]
ret = stats.binned_statistic_2d(x[idxl_terr], y[idxl_terr], z[idxl_terr], 'median', bins = [xe, ye], expand_binnumbers=True)
dtm = fillnodata(ret.statistic, mask=~np.isnan(ret.statistic), max_search_distance=250.0, smoothing_iterations=0)

ret2 = stats.binned_statistic_2d(x, y, z, 'count', bins = [xe, ye], expand_binnumbers=True)
bibi = ret2.statistic 
idxn_bin = ret2.binnumber - 1
z_dtm = dtm[idxn_bin[0], idxn_bin[1]]
h = z - z_dtm



transform = Affine.translation(x_min, y_max) * Affine.scale(res, -res)
transformer = rasterio.transform.AffineTransformer(transform)
transformer.rowcol(x, y)

transform = Affine.translation(-ncols*res/2, nrows*res/2) * Affine.scale(res, -res)
transformer = rasterio.transform.AffineTransformer(transform)
transformer.rowcol(x, y)


dx = res
dy = -res
# refmat = [0, dy; dx, 0; min(xy(:,1))-dx, max(xy(:,2))-dy];
refmat = np.matrix([[0, dy], [dx, 0], [x_min-dx, y_max-dy]])

# np.matmul(np.transpose(np.matrix([x - refmat[2,0], y - refmat[2,1]])), refmat[0:2,:])
P = np.dot( np.transpose(np.matrix([x - refmat[2,0], y - refmat[2,1]])), np.linalg.inv(refmat[0:2,:]) )

row = np.round(P[:,0])-1
col = np.round(P[:,1])-1



np.array((122,323),(212,323))
[[213,323],[1212,24]]

transformer.rowcol(x_min, y_max)
transformer.rowcol(x_max, y_max)
transformer.rowcol(x_max, y_min)
transformer.rowcol(x_min, y_min)

transformer.rowcol(x, y)

xe = np.arange(0, 4, 1)
ye = np.arange(0, 3, 1)

#%%

ret = stats.binned_statistic_2d(x, 
                                y, 
                                None, 
                                statistic='count', 
                                bins = [xe, ye], 
                                expand_binnumbers=True)
stat = ret.statistic
idxn = ret.binnumber # [cols],[rows]
ddd = stat[idxn[0]-1, idxn[1]-1]


ret = stats.binned_statistic_2d(pc.x, pc.y, None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
stat = ret.statistic
idxn = ret.binnumber
ddd = stat[idxn[0]-1, idxn[1]-1]


idxn[1][0]

stat[1][0] # [row][col]
stat[1][0] # [row][col]

ddd = stat[[0, 1, 1], [1, 0, 2]]



stat.iloc(1)

# np.ravel_multi_index(multi_index, dims, mode='raise', order='C')

npoints = np.uint32(np.rot90(ret.statistic))

arr[[0, 1, 1], [1, 0, 2]]


transform = Affine(300.0379266750948, 0.0, 101985.0, 0.0, -300.041782729805, 2826915.0)

a = np.array([[1, 0],
              [0, 1]])


b = np.array([1, 2])
np.matmul(a, b)


# ind2sub =
# sub2ind =
# [x,y] = pix2map(R,row,col)
# [row,col] = map2pix(R,x,y)

#%%

# Using rasterio and affine
a = ds.affine

# col, row to x, y
x, y = a * (col, row)

# x, y to col, row
col, row = ~a * (x, y)



