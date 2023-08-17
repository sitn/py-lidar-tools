# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:04:39 2023

@author: ParkanM
"""

# coding: utf-8
import laspy
import numpy as np
import glob

import pathlib
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from scipy import stats

import rasterio
from rasterio.transform import Affine
from rasterio.fill import fillnodata


#%% parameters

dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/las_rgb/*.las' 
# dir_out = 'D:/Data/ortho_lidar_cdf/'
dir_out = '\\\\nesitn5/h$/geodata/images/cdf_2023/ortho_lidar_10cm/'

tiles = gpd.read_file("D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/tuiles_lidar_2022.shp", crs='epsg:2056')

res = 0.1

#%% process

files_in = glob.glob(dir_in)

n = len(files_in)

# fpath = files_in[42]

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
    
    if len(feature) == 0:
        print("File not found, skipping to next iteration")
        continue
    
    # create raster grid
    xe = np.arange(x_min, x_max + res, res)
    ye = np.arange(y_min, y_max + res, res)
    transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)

    # compute highest point raster
    ret_1 = stats.binned_statistic_2d(pc.x, pc.y, pc.z, 'max', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
    idxn_bin = ret_1.binnumber - 1
    z_dsm = ret_1.statistic[idxn_bin[0], idxn_bin[1]]
    idxl_max = pc.z == z_dsm
    idxn_bin_s = idxn_bin[:,idxl_max]
    
    # create RGB arrays
    nrows, ncols = ret_1.statistic.shape
    ima_rgb = np.zeros((nrows, ncols, 3), dtype=np.uint16)
    ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 0] = pc.red[idxl_max]
    ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 1] = pc.green[idxl_max]
    ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 2] = pc.blue[idxl_max]
    
    # interpolate missing values
    ima_rgb[:, :, 0] = fillnodata(ima_rgb[:, :, 0], mask=ima_rgb[:, :, 0]!=0, max_search_distance=10.0, smoothing_iterations=0)
    ima_rgb[:, :, 1] = fillnodata(ima_rgb[:, :, 1], mask=ima_rgb[:, :, 1]!=0, max_search_distance=10.0, smoothing_iterations=0)
    ima_rgb[:, :, 2] = fillnodata(ima_rgb[:, :, 2], mask=ima_rgb[:, :, 2]!=0, max_search_distance=10.0, smoothing_iterations=0)
    
    ima_rgb = np.rot90(ima_rgb)
    
    # plot raster
    #fig = plt.figure()
    #plt.imshow(ima_rgb, cmap=None)
    #plt.show()
    
    # export to geotiff
    #transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
    #transform_edges * (0, 0)
    
    fname = pathlib.Path(fpath).stem
    fout = rasterio.open(
        dir_out + fname + '_ortho.tif',
        mode = 'w',
        driver = 'GTiff',
        height = ima_rgb.shape[0],
        width = ima_rgb.shape[1],
        count = 3,
        dtype = ima_rgb.dtype,
        crs = 'EPSG:2056',
        compress="lzw",
        transform = transform_edges,
        )
    
    ima_rgb_out = np.moveaxis(ima_rgb, [0, 1, 2], [1, 2, 0])
    # ima_rgb_out = np.rollaxis(ima_rgb, axis=2)
    
    #fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 0, 1]))
    #fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 1, 0]))
    fout.write(ima_rgb_out,[1,2,3])
    fout.close()





#%% tile index

tiles = gpd.read_file('D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp')
# las_extent = 

fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/las_rgb/2546000_1212000.las'
fname = pathlib.Path(fpath).stem

res = 0.5
class_subset = [2,3,4,5,6,7,8,9,10]

pc = laspy.read(fpath)

n_pts = len(pc.points)

x = np.array(pc.x)
y = np.array(pc.y)
z = np.array(pc.z)

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

# compute highest point raster
idxl_s = np.isin(pc.classification, class_subset)
ret_1 = stats.binned_statistic_2d(pc.x[idxl_s], pc.y[idxl_s], pc.z[idxl_s], 'max', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)

np.invert(np.isnan(ret_1.statistic)).sum()

dsm = fillnodata(ret_1.statistic, mask=~np.isnan(ret_1.statistic), max_search_distance=250.0, smoothing_iterations=0)

dsm_mat = np.matrix(dsm)


fig = plt.figure()
plt.imshow(dsm)
plt.show()

# filter highest points

# bin all points
ret_2 = stats.binned_statistic_2d(x, y, z, 'count', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
idxn_bin = ret_2.binnumber - 1

z_dsm = dsm[idxn_bin[0], idxn_bin[1]]

idxl_max = np.absolute(pc.z - z_dsm) < 0.05
idxl_max.sum()

idxl_max2 = pc.z == z_dsm
idxl_max2.sum()


idxn_max = np.where(z == z_dsm)

idxn_bin_s = idxn_bin[:,idxl_max]





nrows, ncols = dsm.shape

ima_r = np.zeros(dsm.shape, dtype=np.uint16)
ima_r[idxn_bin_s[1], idxn_bin_s[0]] = pc.red[idxl_max]



color_r = pc.red[idxl_max]
color_g = pc.green[idxl_max]
color_b = pc.blue[idxl_max]

ima_r = 
ima_g = 
ima_b =


fig = plt.figure()
plt.imshow(ima_r)
plt.show()



# DRAFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = np.array([0.21, 0.35, 0.12, 0.14, 1.34, 1.45, 1.22, 2.21, 2.34, 2.65])
y = np.array([0.32, 0.44, 0.63, 1.22, 0.56, 0.22, 1.23, 1.67, 1.34, 0.77])
z = np.array([22, 10, 5, 8, 23, 56, 34, 102, 76, 48])

# extent
x_min = 0
x_max = 3
y_min = 0
y_max = 2
res = 1

# create raster grid
xe = np.arange(x_min, x_max + res, res)
ye = np.arange(y_min, y_max + res, res)

# scatter plot
fig, axs = plt.subplots(1, 1)
axs.scatter(x, y)
for i, txt in enumerate(z):
    axs.annotate(txt, (x[i], y[i]))
axs.axis('equal')
fig.tight_layout()
plt.show()

# append left bin edge inclusion
if xe[-1] == x_max:
    xe = np.append(xe, xe[-1]+res)

if ye[-1] == y_max:
    ye = np.append(ye, ye[-1]+res)


# compute highest point raster
# ret_1 = stats.binned_statistic_2d(x, y, z, 'max', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)

# ret_1.statistic
# np.invert(np.isnan(ret_1.statistic)).sum()

#dsm = fillnodata(ret_1.statistic, mask=~np.isnan(ret_1.statistic), max_search_distance=250.0, smoothing_iterations=0)



# plot raster
fig = plt.figure()
plt.imshow(dsm)
plt.show()


# bin all points
ret = stats.binned_statistic_2d(x, y, z, 'max', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
idxn_bin = ret.binnumber - 1

z_dsm = ret.statistic[idxn_bin[0], idxn_bin[1]]
idxl_max = z == z_dsm
idxn_bin_s = idxn_bin[:,idxl_max]

# create colors arrays
nrows, ncols = ret.statistic.shape

# create colors arrays
ima = np.zeros((nrows, ncols), dtype=np.single)
ima[idxn_bin_s[0], idxn_bin_s[1]] = np.single(z[idxl_max])


ima = np.rot90(ima)

# plot raster
fig = plt.figure()
plt.imshow(ima, cmap=None)
plt.show()

transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
transform_edges * (0, 0)
 
 
# DRAFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fpath_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/las_rgb/2546000_1212000.las'

res = 0.1
pc = laspy.read(fpath_in)

n_pts = len(pc.points)

x = np.array(pc.x)
y = np.array(pc.y)
z = np.array(pc.z)

# comput extent
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

# create raster grid
xe = np.arange(x_min, x_max + res, res)
ye = np.arange(y_min, y_max + res, res)

if xe[-1] == x_max:
    xe = np.append(xe, xe[-1]+res)

if ye[-1] == y_max:
    ye = np.append(ye, ye[-1]+res)


# compute highest point raster
ret_1 = stats.binned_statistic_2d(x, y, z, 'max', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
idxn_bin = ret_1.binnumber - 1
z_dsm = ret_1.statistic[idxn_bin[0], idxn_bin[1]]

idxl_max = z == z_dsm
idxn_bin_s = idxn_bin[:,idxl_max]

# create colors arrays
#nrows, ncols = ret_1.statistic.shape
#ima_rgb = np.zeros((nrows, ncols, 3), dtype=np.single)
#ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 0] = np.single(pc.red[idxl_max]) / 65535
#ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 1] = np.single(pc.green[idxl_max]) / 65535
#ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 2] = np.single(pc.blue[idxl_max]) / 65535

# create RGB arrays
nrows, ncols = ret_1.statistic.shape
ima_rgb = np.zeros((nrows, ncols, 3), dtype=np.uint16)
ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 0] = pc.red[idxl_max]
ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 1] = pc.green[idxl_max]
ima_rgb[idxn_bin_s[0], idxn_bin_s[1], 2] = pc.blue[idxl_max]

# interpolate missing values
ima_rgb[:, :, 0] = fillnodata(ima_rgb[:, :, 0], mask=ima_rgb[:, :, 0]!=0, max_search_distance=10.0, smoothing_iterations=0)
ima_rgb[:, :, 1] = fillnodata(ima_rgb[:, :, 1], mask=ima_rgb[:, :, 1]!=0, max_search_distance=10.0, smoothing_iterations=0)
ima_rgb[:, :, 2] = fillnodata(ima_rgb[:, :, 2], mask=ima_rgb[:, :, 2]!=0, max_search_distance=10.0, smoothing_iterations=0)

# create grayscale array
#nrows, ncols = ret_1.statistic.shape
#ima_rgb = np.zeros((nrows, ncols), dtype=np.uint16)
#ima_rgb[idxn_bin_s[0], idxn_bin_s[1]] = pc.red[idxl_max]

ima_rgb = np.rot90(ima_rgb)

# plot raster
fig = plt.figure()
plt.imshow(ima_rgb, cmap=None)
plt.show()

# export to geotiff
transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
transform_edges * (0, 0)


dir_out = 'D:/Data/ortho_lidar_cdf/'
fout = rasterio.open(
    dir_out + fname + '_ortho.tif',
    mode = 'w',
    driver = 'GTiff',
    height = ima_rgb.shape[0],
    width = ima_rgb.shape[1],
    count = 3,
    dtype = ima_rgb.dtype,
    crs = 'EPSG:2056',
    compress="lzw",
    transform = transform_edges,
    )

ima_rgb_out = np.moveaxis(ima_rgb, [0, 1, 2], [1, 2, 0])
# ima_rgb_out = np.rollaxis(ima_rgb, axis=2)

#fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 0, 1]))
#fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 1, 0]))
fout.write(ima_rgb_out,[1,2,3])
fout.close()





np.max(ima_rgb[:, :, 0])
np.max(ima_rgb[:, :, 1])
np.max(ima_rgb[:, :, 2])

# plot raster
fig = plt.figure()
plt.imshow(ima_rgb[:,:,2], cmap=None)
plt.show()




#%% DRAFT


np.invert(np.isnan(ret_1.statistic)).sum()

# dsm = fillnodata(ret_1.statistic, mask=~np.isnan(ret_1.statistic), max_search_distance=250.0, smoothing_iterations=0)

# plot raster
fig = plt.figure()
plt.imshow(dsm)
plt.show()


# bin all points
ret_2 = stats.binned_statistic_2d(x, y, z, 'count', bins = [xe, ye], range = [[xe[0], xe[-1]], [ye[0], ye[-1]]], expand_binnumbers=True)
idxn_bin = ret_2.binnumber - 1
z_dsm = dsm[idxn_bin[0], idxn_bin[1]]

idxl_max = z == z_dsm

np.invert(np.isnan(ret_1.statistic)).sum()
idxl_max.sum()



#%% DRAFT

res = 1
xe = np.arange(x_min, x_max + res, res)
ye = np.arange(y_min, y_max + res, res)
transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
        
ret = stats.binned_statistic_2d(pc.x, pc.y, None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
npoints = np.uint32(np.rot90(ret.statistic))
        

# plot raster
fig = plt.figure()
plt.imshow(npoints, cmap=None)
plt.show()



#%% export to geotiff
        
fout = rasterio.open(
            dir_out + fname + '_density.tif',
            'w',
            driver = 'GTiff',
            height = npoints.shape[0],
            width = npoints.shape[1],
            count = 1,
            dtype = npoints.dtype,
            crs = 'EPSG:2056',
            transform = transform_edges,
        )
        
fout.write(npoints, 1)
fout.close()