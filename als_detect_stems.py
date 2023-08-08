# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:29:48 2023

@author: ParkanM
"""

import numpy as np
import cv2
import laspy
import matplotlib.pyplot as plt



#%% parameters

fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_rochefort_ch1903p_survey_v220418.las'
fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_chambrelien_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ge_2017_versoix_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry01_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry19_ch1903p_survey.las'
fpath = 'C:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry20e_ch1903p_survey.las'


# fname = os.path.basename(fpath)
fname = Path(fpath).stem
# fext =
fpath_out_pc_subset = "C:/Projects/lidar_tree_detection/input/pointclouds_subsets/%s_sub.las" % (fname)
fpath_out_tree_trunks = "C:/Projects/lidar_tree_detection/input/tree_trunks/%s_tree_trunks.shp" % (fname)



#%% read LAS file

pc = laspy.read(fpath)

print('Point format:', pc.header.point_format)
print('Points from Header:', pc.header.point_count)
print('File source ID:', pc.header.file_source_id)
print('UUID:', pc.header.uuid)
print('Generating software:', pc.header.generating_software)
print('Extra VLR bytes:', pc.header.extra_vlr_bytes)
print('Number of EVLR:', pc.header.number_of_evlrs)

# apply scale and offset
x_s = pc.x * pc.header.scales[0] + pc.header.offsets[0]
y_s = pc.y * pc.header.scales[1] + pc.header.offsets[1]
z_s = pc.z * pc.header.scales[2] + pc.header.offsets[2]
intensity = pc.intensity
luid = pc.luid


# list(pc.point_format.dimension_names)
# set(list(pc.classification))

#%% extract section

idxl_r = pc.return_number == pc.number_of_returns
idxl_c = np.isin(pc.classification, [4,5])
idxl_z = (pc.z >= 610) & (pc.z <= 610.5)

idxl = idxl_r & idxl_c & idxl_z

# plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(pc.x[idxl], pc.y[idxl], s=0.5, c="black", marker=".", alpha=1)


#ax.set_xlim(0, 100)
ax.set_xlabel('Y [m]')

#ax.set_ylim(0, 100)
ax.set_ylabel('X [m]')

title = "FLAI - %s\nn=%s, r2=%s" % (name, n_obs, round(r_corr_sq,2))

ax.set_title(title, fontdict=None, loc='center', pad=None)
       
ax.set_aspect('equal')

#↔ ax.set(xlim=(-3, 3), ylim=(-3, 3))

# save figure
fig.savefig('C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/cross_section.png', dpi=1200)
