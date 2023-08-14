# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:04:39 2023

@author: ParkanM
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:21:00 2023

@author: Matthew Parkan, SITN
"""

# coding: utf-8
import laspy
import numpy as np
import glob

import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from scipy import stats

import rasterio
from rasterio.transform import Affine
from rasterio.fill import fillnodata


#%% parameters

# dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/'
dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/' 


dir_out_density = 'D:/Projects/intemperie_cdf_20230724/LIDAR/RASTER/DENSITY/'
dir_out_dsm = 'D:/Projects/intemperie_cdf_20230724/LIDAR/RASTER/DSM/'
dir_out_dtm = 'D:/Projects/intemperie_cdf_20230724/LIDAR/RASTER/DTM/'
dir_out_dhm = 'D:/Projects/intemperie_cdf_20230724/LIDAR/RASTER/DHM/'


#%% tile index

tiles = gpd.read_file('D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/tileindex_500m.shp')


#%% grid

n_tiles = len(tiles)

# feature = tiles.iloc[90]

for index, feature in tiles.iterrows():

    print("Processing tile %u / %u: %s" % (index+1, n_tiles, feature.tileid))
    
    #%% get tile bounding box
    x_min = np.min(feature.geometry.exterior.coords.xy[0])
    x_max = np.max(feature.geometry.exterior.coords.xy[0])
    y_min = np.min(feature.geometry.exterior.coords.xy[1])
    y_max = np.max(feature.geometry.exterior.coords.xy[1])
    
    #%% read LAS file
    fpath_in = dir_in + feature.tileid + '.las'
    pc = laspy.read(fpath_in)
    
    #%% define raster grid cells

    # xe = np.arange(x_min - res/2, x_max + res/2, res)
    # ye = np.arange(y_min - res/2, y_max + res/2, res)
    # transform_edges = Affine.translation(xe[0] - res / 2, ye[0] - res / 2) * Affine.scale(res, -res)
    
    
    #%% compute terrain point density
    
    idxl = np.isin(pc.classification, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if any(idxl):
    
        res = 1
        xe = np.arange(x_min, x_max + res, res)
        ye = np.arange(y_min, y_max + res, res)
        transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
        
        ret = stats.binned_statistic_2d(pc.x[idxl], pc.y[idxl], None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
        npoints = np.uint32(np.rot90(ret.statistic))
        
        #%% export to geotiff
        
        fout = rasterio.open(
            dir_out_density + feature.tileid + '_density.tif',
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
        
    #%% compute terrain model (DTM)
    idxl_terr = np.isin(pc.classification, [2,8])
    if any(idxl_terr):
        res = 0.25
        xe = np.arange(x_min, x_max + res, res)
        ye = np.arange(y_min, y_max + res, res)
        transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
        
        ret = stats.binned_statistic_2d(pc.x[idxl_terr], pc.y[idxl_terr], pc.z[idxl_terr], 'mean', bins = [xe, ye], expand_binnumbers=True)
        stat = np.float32(np.rot90(ret.statistic))
        dtm = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=250.0, smoothing_iterations=0)
        
        #%% export to geotiff
        
        fout = rasterio.open(
            dir_out_dtm + feature.tileid + '_dtm.tif',
            'w',
            driver = 'GTiff',
            height = dtm.shape[0],
            width = dtm.shape[1],
            count = 1,
            dtype = dtm.dtype,
            crs = 'EPSG:2056',
            transform = transform_edges,
        )
        
        fout.write(dtm, 1)
        fout.close()
        
    
    #%% compute surface model
    
    idxl_surf = np.isin(pc.classification, [2,3,4,5,6,8])
    if any(idxl_surf):
        
        res = 0.25
        xe = np.arange(x_min, x_max + res, res)
        ye = np.arange(y_min, y_max + res, res)
        transform_edges = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)
        
        ret = stats.binned_statistic_2d(pc.x[idxl_surf], pc.y[idxl_surf], pc.z[idxl_surf], statistic='max', bins = [xe, ye], expand_binnumbers=True)
        stat = np.float32(np.rot90(ret.statistic))
        
        dsm = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=250.0, smoothing_iterations=0)
        
        #%% export to geotiff
        
        fout = rasterio.open(
            dir_out_dsm + feature.tileid + '_dsm.tif',
            'w',
            driver = 'GTiff',
            height = dsm.shape[0],
            width = dsm.shape[1],
            count = 1,
            dtype = dsm.dtype,
            crs = 'EPSG:2056',
            transform = transform_edges,
        )
        
        fout.write(dsm, 1)
        fout.close()
    
    #%% compute height model
    
    if any(idxl_surf) & any(idxl_terr):
        
        dhm = dsm - dtm
    
        #%% export to geotiff
        
        fout = rasterio.open(
            dir_out_dhm + feature.tileid + '_dhm.tif',
            'w',
            driver = 'GTiff',
            height = dhm.shape[0],
            width = dhm.shape[1],
            count = 1,
            dtype = dhm.dtype,
            crs = 'EPSG:2056',
            transform = transform_edges,
        )
        
        fout.write(dhm, 1)
        fout.close()

    
    
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    

# stopped at file 2546500_1215500.laz

# fpath = 'C:/Users/parkanm/Desktop/TOPO/LAS/2496000_1116000.las'

fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/LCDF_LV95_NF02_000004.laz'
path = 'D:/Projects/intemperie_cdf_20230724/LIDAR/*.laz'
files_in = glob.glob(path)


    
    points = pc.points
    xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()
    
    
    
for fpath in files_in:
    
    
    print("Processing file: %s" % fpath)
    
    
    #%% read LAS file
    pc = laspy.read(fpath)
    points = pc.points
    xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()

    
    #%% filter points
    
    # set to not classified
    n = len(pc.points)
    
    #%% bounding box
    
    x_min = np.min(pc.x)
    x_max = np.max(pc.x)
    y_min = np.min(pc.y)
    y_max = np.max(pc.y)

    xv = np.arange(x_min, x_max, x_res)
    yv = np.arange(y_min, y_max, y_res)
    transform = Affine.translation(xv[0] - res / 2, yv[0] - res / 2) * Affine.scale(res, res)

    #%% bin edges
    
    xe = np.arange(x_min - res/2, x_max + res/2, res)
    ye = np.arange(y_min - res/2, y_max + res/2, res)
    
    #%% compute terrain point density
    
    idxl = np.isin(pc.classification, [2, 8])
    ret = stats.binned_statistic_2d(pc.x[idxl], pc.y[idxl], None, statistic='count', bins = [xe, ye], expand_binnumbers=True)
    stat = np.flip(np.rot90(ret.statistic),0)
    
    #%% compute terrain model
    
    idxl = np.isin(pc.classification, [2,8])
    ret = stats.binned_statistic_2d(pc.x[idxl], pc.y[idxl], pc.z[idxl], 'mean', bins = [xe, ye], expand_binnumbers=True)
    stat = np.flip(np.rot90(ret.statistic), 0)
    dtm = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=100.0, smoothing_iterations=0)
    
    #%% compute surface model
    
    idxl = np.isin(pc.classification, [2,3,4,5,6,8])
    ret = stats.binned_statistic_2d(pc.x[idxl], pc.y[idxl], pc.z[idxl], statistic='max', bins = [xe, ye], expand_binnumbers=True)
    stat = np.rot90(ret.statistic)
    dsm = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=100.0, smoothing_iterations=0)
    
    #%% compute height model
    
    dhm = dsm - dtm
    
    #%% export to geotiff
    
    fout = rasterio.open(
        'C:/Users/parkanm/Desktop/TOPO/LAS/new.tif',
        'w',
        driver = 'GTiff',
        height = zi.shape[0],
        width = zi.shape[1],
        count = 1,
        dtype = zi.dtype,
        crs = 'EPSG:2056',
        transform=transform,
    )
    
    fout.write(zi, 1)
    fout.close()


#%% plot

fig = plt.figure()
plt.imshow(stat)
plt.show()

fig = plt.figure()
plt.imshow(dsm)
plt.show()

fig = plt.figure()
plt.imshow(dhm)
plt.show()


#%% export to geotiff

# transform_edges = Affine.translation(xe[0] - res / 2, ye[0] - res / 2) * Affine.scale(res, res)
transform_edges = Affine.translation(xe[0] - res / 2, ye[-1] + res / 2) * Affine.scale(res, -res)

fout = rasterio.open(
    'C:/Users/parkanm/Desktop/TOPO/LAS/dtm.tif',
    'w',
    driver = 'GTiff',
    height = ima.shape[0],
    width = ima.shape[1],
    count = 1,
    dtype = ima.dtype,
    crs = 'EPSG:2056',
    transform = transform_edges,
)

fout.write(dtm, 1)
fout.close()



fout = rasterio.open(
    'C:/Users/parkanm/Desktop/TOPO/LAS/dsm1233.tif',
    'w',
    driver = 'GTiff',
    height = dsm.shape[0],
    width = dsm.shape[1],
    count = 1,
    dtype = dsm.dtype,
    crs = 'EPSG:2056',
    transform = transform_edges,
)

fout.write(dsm, 1)
fout.close()

fout = rasterio.open(
    'C:/Users/parkanm/Desktop/TOPO/LAS/dhm.tif',
    'w',
    driver = 'GTiff',
    height = ima.shape[0],
    width = ima.shape[1],
    count = 1,
    dtype = ima.dtype,
    crs = 'EPSG:2056',
    transform = transform_edges,
)

fout.write(dhm, 1)
fout.close()



#%% DRAFT



xi, yi = np.meshgrid(xv, yv)
# xi, yi = np.meshgrid(xv, yv, indexing='ij')

#%% interpolate

idxl = np.isin(pc.classification, [2,8])
zi = griddata((pc.x[idxl], pc.y[idxl]), pc.z[idxl], (xi, yi), method='linear')

# ima = fillnodata(stat, mask=~np.isnan(stat) , max_search_distance=100.0, smoothing_iterations=0)


#%% rbf3 = RBFInterpolator((pc.x[idxl], pc.y[idxl]), pc.z[idxl], kernel="linear", smoothing=5)
rbf3 = RBFInterpolator((pc.x[idxl], pc.y[idxl]), pc.z[idxl], kernel="linear", smoothing=5)
znew = rbf3(xnew, ynew)

#%% plot

fig = plt.figure()
plt.imshow(zi)
plt.show()




#%% binning



xt = [0.1, 0.1, 0.1, 0.6]
yt = [2.1, 2.6, 2.1, 2.1]
zt = [2.,3.,5.,7.]

binx = np.arange(0, 2, 1)
biny = np.arange(0, 2, 1)


xt = [0.5, 0.4, 0.3, 0.6]
yt = [1.0, 2.6, 2.1, 2.1]

binx = [0, 1, 2]
biny = [0, 2, 4]


ret = stats.binned_statistic_2d(xt, yt, None, 'count', bins=[binx, biny], expand_binnumbers=True)


ret.binnumber
ret.statistic


