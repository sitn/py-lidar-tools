# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:41:48 2023

@author: ParkanM
"""

import numpy as np
import tifftools
import rasterio
from rasterio.transform import Affine
from rasterio.plot import show

from matplotlib import pyplot
import pathlib
# https://rasterio.readthedocs.io/en/stable/topics/reading.html


#%% parameters

fpath = 'D:/Data/images/2023/ORTHO_10cm_MNT2022/ORTHO_LCDF_4BD_10cm_MN95_2543500_1213000.tif'

# fpath = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2023_CHXFDS/3_Orthos/ortho_lidar_10cm/2545500_1212500_ortho.tif'

dir_out = 'D:/Data/test/'


#%% read image

src = rasterio.open(fpath)

ima = src.read()

b1 = src.read(1) # red
# b2 = src.read(2) # green
# b3 = src.read(3) # blue
# b4 = src.read(4) # nir
# b5 = src.read(5) # alpha mask


b1 = src.read(1) # red

msk = src.read_masks(1)
msk.shape

mask = src.read(5) == 65535
mask_im = b1 != 65535


src.write_mask(True)


src.write_mask(mask)


src.read_masks(1).all()
src.read_masks(2)

src.close()


np.min(b1)
np.max(b1)
np.mean(b1[mask])
np.mean(b1[mask_im])


show(ima[0:3,:,:], transform=src.transform)
show((src, 1), cmap='viridis')
pyplot.imshow(ima[0:3,:,:])
pyplot.imshow(b5, cmap='pink')


def normalize(x, lower, upper):
    """Normalize an array to a given bound interval"""

    x_max = np.max(x)
    x_min = np.min(x)

    m = (upper - lower) / (x_max - x_min)
    x_norm = (m * (x - x_min)) + lower

    return x_norm

# Normalize each band separately
ima_norm = np.array([normalize(ima[i,:,:], 0, 255) for i in range(ima.shape[0])])
ima_rgb = ima_norm.astype("uint8")


#%% plot image

# Make the first (band) dimension the last
pyplot.imshow(np.moveaxis(ima, 0, -1))

# Plot R/G/B composite
pyplot.imshow(np.moveaxis(ima_rgb[[0,1,2],:,:], 0, -1))

# Plot NIR/R/G composite
pyplot.imshow(np.moveaxis(ima_rgb[[3,1,2],:,:], 0, -1))

# Plot mask
pyplot.imshow(np.moveaxis(ima_rgb[[4],:,:], 0, -1))


#%% write image

# no data are stored as 65535 (white)

fname = pathlib.Path(fpath).stem
fout = rasterio.open(
    dir_out + fname + '_corr2.tif',
    mode = 'w',
    driver = 'GTiff',
    height = ima.shape[1],
    width = ima.shape[2],
    count = ima.shape[0],
    dtype = ima.dtype,
    crs = 'EPSG:2056',
    compress="lzw",
    nodata = 65535,
    transform = src.transform,
    )
    
ima_rgb_out = np.moveaxis(ima, [0, 1, 2], [1, 2, 0])
# ima_rgb_out = np.rollaxis(ima_rgb, axis=2)
    
#fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 0, 1]))
#fout.write(np.moveaxis(ima_rgb, [0, 1, 2], [2, 1, 0]))
fout.write(ima, [1,2,3,4,5])
fout.close()


#%% check results

src2 = rasterio.open('D:/Data/test/ORTHO_LCDF_4BD_10cm_MN95_2543500_1213000_corr.tif')
ima2 = src2.read()
b1_2 = src2.read(1) # red
src2.close()

np.array_equal(ima, ima2)
np.array_equal(b1, b1_2)


#%% DRAFT ---------------------------------------------------------------------------------------------- 


#%% read image

info = tifftools.read_tiff('D:/Projects/intemperie_cdf_20230724£/data/ORTHO_10cm_MNT2022/ORTHO_LCDF_4BD_10cm_MN95_2556000_1216500.tif')


info = tifftools.read_tiff('ORTHO_LCDF_4BD_10cm_MN95_2556000_1216500.tif')


tiles = gpd.read_file("D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/tuiles_lidar_2022.shp", crs='epsg:2056')



info['ifds'][0]['tags'][tifftools.Tag.ImageDescription.value] = {
    'data': 'A dog digging.',
    'datatype': tifftools.Datatype.ASCII
}
exififd = info['ifds'][0]['tags'][tifftools.Tag.EXIFIFD.value]['ifds'][0][0]

exififd['tags'][tifftools.constants.EXIFTag.FNumber.value] = {
    'data': [54, 10],
    'datatype': tifftools.Datatype.RATIONAL
}


tifftools.write_tiff(info, 'photograph_tagged.tif')