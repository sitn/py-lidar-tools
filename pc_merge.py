# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:53:59 2023

@author: ParkanM
"""

# https://github.com/laspy/laspy/issues/156
# https://pypi.org/project/laspy/

# Install with LAZ support via both lazrs & laszip
# pip install laspy[lazrs,laszip]

# Import libraries
import os
import pathlib
import glob
from pyproj import CRS
from pyproj.enums import WktVersion
import geopandas as gpd
import matplotlib
import laspy
import numpy as np


#%% parameters


# production
dir_in = 'D:/Data/pointclouds/2023/all/flight_1_raw/*.las' # flight 1
dir_out = 'D:/Data/pointclouds/2023/all/merged/*.las' # flight 2

# test
dir_in = 'D:/Data/pointclouds/2023/all/test_in/*.las' # flight 1
dir_out = 'D:/Data/pointclouds/2023/all/test/*.las' # flight 2



#%% create empty LAS file

files_in = glob.glob(dir_in)
files_out = glob.glob(dir_out)

fnames_in = [os.path.basename(x).split(os.extsep)[0] for x in files_in]
fnames_out = [os.path.basename(x).split(os.extsep)[0] for x in files_out]


#%% append function

def append_to_las(in_las, out_las):
   with laspy.open(out_las, mode='a') as outlas:
      with laspy.open(in_las) as inlas:
         for points in inlas.chunk_iterator(2_000_000):
            outlas.append_points(points)


#%% append points

n = len(files_out)

for index, fpath_out in enumerate(files_out):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    # get file name
    fname = os.path.basename(fpath_out).split(os.extsep)[0]
    
    idxn = [i for i ,e in enumerate(fnames_in) if e == fname]
    
    if len(idxn) > 0:
        
        append_to_las(files_in[idxn[0]], fpath_out)
        
        
        
 
#%% rewrite

files_in = glob.glob('D:/Data/pointclouds/2023/all/flight_1/*.las')

dir_out = 'D:/Data/pointclouds/2023/all/flight_1_raw/'

n = len(files_in)

for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    fname = os.path.basename(fpath)
    
    # read LAS file
    pc = laspy.read(fpath)
                    
    # remove extra dimensions)
    pc.remove_extra_dims(['Distance', 'Group', 'Normal'])

    # write to LAS file
    pc.write(dir_out + fname)
    
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
#%% DRAFT
    
        
        
    idxn = fnames_in.index(fname)
    
    
    idxl = [x == fname for x in fnames_in]
    idxn = np.where(idxl)
    
    fnames_in
    
  
    
    if fname in fnames_in:
        
        append_to_las(fpath_in, fpath_out)
        
        
    
        



    
    idxl = [x == fname for x in fnames_out]
    idxn = np.where(idxl)
    
    if np.any(idxl):
        
        
        
        
    
    
    idxn = [i for i, x in enumerate(t) if x]
    
    idxn = [i for i, fname in enumerate(fnames_out) if fname]
    
    
    files_out[idxl]
    
    '2545500_1212000' == fnames_out
    
    # check for duplicate file names
    if fname in files_in
    
        append_to_las(fpath2, fpath)
       

            

n = len(files_in)

# get file names
fnames = [os.path.basename(x).split(os.extsep)[0] for x in files_in]
occurences = [fnames.count(x) for x in fnames]
skip = [False for i in range(n)]


fpath1 = 'D:/Data/pointclouds/2023/merged/2557000_1217000.laz'  # point format 6
fpath2 = 'D:/Data/pointclouds/2023/merged/LCDF_MN95_NF02_2557000-1217000.laz' # point format 7 (with RGB)
fpath3 = 'D:/Data/pointclouds/2023/merged/2557000_1217000_new.laz'  # point format 7

pc1 = laspy.read(fpath1)
pc1.header.point_format
pc1.header.global_encoding.gps_time_type

pc2 = laspy.read(fpath2)
pc2.header.point_format
pc2.header.global_encoding.gps_time_type



for index, fpath in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    # get file name
    fname = os.path.basename(fpath).split(os.extsep)[0]
    
    # check for duplicate file names
    
    # read LAS file
    pc = laspy.read(fpath1)
    
    laspy.lasappender.LasAppender()
    
    
    
    
    
    
    
    





   with laspy.open(fpath3, mode='a') as outlas:
      with laspy.open(fpath2) as inlas:
         for points in inlas.chunk_iterator(2_000_000):
            outlas.append_points(points)    
        


            

# configure LAS header
header = laspy.LasHeader(point_format=7, version="1.4")
# header = laspy.LasHeader(point_format=3, version="1.2")
crs = CRS.from_epsg(2056)
header.add_crs(crs)
header.offsets = np.append(np.min(boundary, axis=0), 0.0)
header.scales = np.array([0.01, 0.01, 0.01])


#%% process tiles

all_points = laspy.PackedPointRecord.empty(header.point_format)

for fpath in tiles_s.fpath:
    
    print("Processing file: %s" % fpath)
    
    # read LAS file
    pc = laspy.read(fpath)
    
    # point in polygon filter
    idxl_in = path_p.contains_points(np.stack([pc.x, pc.y], axis=0).transpose((1, 0)))
    n_in = np.count_nonzero(idxl_in)
    
    print("Points filtered: %u" % n_in)
    pc.points = pc.points[idxl_in]

    # adjust scalings and offsets
    pc.change_scaling(scales = header.scales, offsets = header.offsets)
    
    # append clipped points to array
    all_points.array = np.append(all_points.array, pc.points.array)
    
    
#%% write LAS file

las = laspy.LasData(header=header, points=all_points)
las.write(fpath_las_out)

print("LAS file written to: %s" % fpath_las_out)

