# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:21:00 2023

@author: Matthew Parkan, SITN, matthew.parkan@ne.ch
"""

# REFERENCES: 
# https://stackoverflow.com/questions/9543205/matplotlib-get-the-colormap-array

# coding: utf-8
import laspy
import numpy as np
import glob
import pathlib
import re
import matplotlib as mpl



#%% parameters

# path = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/LCDF_LV95_NF02_000052.las'
path = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v3/*.laz' 
files_in = glob.glob(path)
skip_existing = False

# dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/classified/'
dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v3_recolorized/'
dir_out = 'D:/Data/test/'


files_out = glob.glob(dir_out + '/*.las')
files_skip = []


#%% process

files_in = [files_in[42]]



# define colormap
ncolors = 25
d_max = 3
cmap = mpl.colormaps['jet'].resampled(ncolors)


ind_color = np.uint16(np.ndarray.round(ncolors * np.minimum(pc.distance, d_max) / d_max))
cmap(ind_color)

# scale distance
np.min(pc.distance)
np.max(pc.distance)
np.mea()

np.minimum([1,2,32,4,34], [3])
cmap(pc.distance)
cmap(7)

# cmap = mpl.colors.Colormap('jet', N=256)
# mpl.cm.hot(range(256))
# cmap = mpl.colormaps.get_cmap('viridis', 12)
# viridis.colors
# cmap.colors

fpath = files_in[100]
fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/change_detection/v3/2552000_1215500_change.laz'

n = len(files_in)

for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    # check if output file already exists
    fname = pathlib.Path(fpath).stem
    # skip = any(list(map(lambda x: bool(re.search(fname, x)), files_out + files_skip)))  
    
    #if skip & skip_existing:
        #print("Already processed -> Skipping")
        #continue
    
    # read LAS file
    pc = laspy.read(fpath)
    n_pts = len(pc.points)
    
    # overwrite RGB (color target classes only)
    """
    idxl_target = np.isin(pc.classification, [4,5])
    ind_color = np.uint16(np.ndarray.round(ncolors * np.minimum(pc.distance, d_max) / d_max))
    ind_color[~idxl_target] = 0
    rgba = np.uint16(cmap(ind_color) * 65535)
    pc.red = rgba[:,0]
    pc.green = rgba[:,1]
    pc.blue = rgba[:,2]
    """
    
    ## overwrite RGB (color target classes only)
    idxl_target = np.isin(pc.classification, [4,5]) & (pc.distance >= d_max)
    ind_color = np.uint16(np.ndarray.round(ncolors * np.minimum(pc.distance, d_max) / d_max))
    rgba = np.uint16(cmap(ind_color) * 65535)
    pc.red[idxl_target] = 65535
    pc.green[idxl_target] = 0
    pc.blue[idxl_target] = 0

    # write classified point cloud to LAS file
    fpath_out = dir_out + fname + '_recolored.las'
    outFile = laspy.LasData(pc.header)
    outFile.points = pc.points
    outFile.write(fpath_out)
