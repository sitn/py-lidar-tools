# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:21:00 2023

@author: 
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
csf.params.interations = 600; # integer. Maximum iterations for simulating cloth. The default value is 500. Usually, there is no need to change this value.

#skip = any(list(map(lambda x: bool(re.search('2528000_1203500', x)), files_out)))
#skip = any(file == '2528000_1203500_classified.las' for file in files_out)
#bool(re.search('2528000_1203500', files_out))
#bool(re.search('2528000_1203500', '2528000_1203500_classified'))
#pathlib.Path(files_in[0]).stem
# any(list(map(lambda x: (fname + '_classified.las') in x, files_out)))
 
# fpath = '\\\\nesitn5/geodata/pointclouds/Aeriallidar/Lidar2022_IGN/1_las/Ch1903plus_laz/2546500_1215500.laz'

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


#%% write classified point cloud to LAS file

res = 0.5

x_min = np.min(pc.x)
x_max = np.max(pc.x)
y_min = np.min(pc.y)
y_max = np.max(pc.y)

xe = np.arange(x_min, x_max + res, res)
ye = np.arange(y_min, y_max + res, res)

# transform = Affine.translation(xv[0] - res / 2, yv[0] - res / 2) * Affine.scale(res, res)
transform = Affine.translation(xe[0], ye[-1]) * Affine.scale(res, -res)

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

bibi = transformer.rowcol(pc.x, pc.y)

transformer.rowcol(x_min, y_max)


xe = np.arange(0, 4, 1)
ye = np.arange(0, 3, 1)

x = [0.5, 0.5, 2.5]
y = [0.5, 0.5, 1.5]

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
    