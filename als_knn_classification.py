# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:37:40 2023

@author: ParkanM
"""


# coding: utf-8
import laspy
import numpy as np
import glob
import pathlib
import re
from sklearn.neighbors import KDTree



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

dir_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/classified/'
files_out = glob.glob(dir_out + '/*.las')

#%% parameters

fpath = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/LCDF_LV95_NF02_000012.las'
fpath_ref = 'D:/Projects/intemperie_cdf_20230724/LIDAR/LAS/2022/LCDF_LV95_NF02_000012.las'

fpath_las_out = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/knn_classification/LCDF_LV95_NF02_000012_knn_class.las'

dir_in = 'D:/Projects/intemperie_cdf_20230724/LIDAR/terrascan_project/corrected/*.las'
dir_ref_in = ''
dir_out = ''

d_max = 1



files_out = glob.glob(dir_out + '/*.las')

#%% read LAS files

files_in = glob.glob(dir_in)

pc = laspy.read(fpath)
points = pc.points
xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()

pc_ref = laspy.read(fpath_ref)
xyz_ref = np.vstack((pc_ref.x, pc_ref.y, pc_ref.z)).transpose()


#%% find k nearest neighbours

# build KDTree
tree = KDTree(xyz_ref, leaf_size=2)

# find neighbours
knn_dist, knn_ind = tree.query(xyz, k=3)
knn_class = pc_ref.classification[knn_ind]

# assign classification
pc.points.classification = knn_class[:,0]

# assign 0 classification to points beyond max range
idxl_dist = knn_dist[:,0] <= d_max
pc.points.classification[~idxl_dist] = 0

#%% write result to LAS file

las = laspy.LasData(header=pc.header, points=pc.points)
las.write(fpath_las_out)










dist, ind = tree.query(xyz[0:5,:], k=3)

pc_ref.classification[ind[:,0]] = 





#%% 

for index, fpath in enumerate(files_in):
# for fpath in files_in:
    
    print("Processing file: %s" % fpath)
    
    #%% check if output file already exists
    fname = pathlib.Path(fpath).stem
    fpath_out = dir_out + fname + '_classified.las'
    skip = any(list(map(lambda x: bool(re.search(fname, x)), files_out + files_skip)))  
    
    if skip:
        print("Already processed -> Skipping")
        continue
    
    #%% read LAS file
    pc = laspy.read(fpath)
    points = pc.points
    xyz = np.vstack((pc.x, pc.y, pc.z)).transpose()
    
    #%% reset classification
    
    # set to not classified
    n = len(pc.points)
    
    pc.classification = np.full(n, 0, dtype=np.int8)
    
    #%% extract last returns
    
    idxl_last = pc.return_number == pc.number_of_returns
    idxn_last = np.where(idxl_last)[0]
    xyz_last = np.vstack((pc.x[idxl_last], pc.y[idxl_last], pc.z[idxl_last])).transpose()
    
    #%% terrain classification
    
    csf.setPointCloud(xyz_last)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.
    
    if len(ground) > 0:
    
        idxn_ground = idxn_last[np.array(ground)]
        pc.classification[idxn_ground] = 2
        # pc.classification[np.array(ground)] = 2
    
    #%% write classified point cloud to LAS file
     
    outFile = laspy.LasData(pc.header)
    
    outFile.points = pc.points # extract ground points, and save it to a las file.
    
    outFile.write(fpath_out)
