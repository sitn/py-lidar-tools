# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:58:57 2023

# references:
https://medium.com/@rdadlaney/basics-of-3d-point-cloud-data-manipulation-in-python-95b0a1e1941e
https://stackoverflow.com/questions/74954954/open3d-o3d-visualization-visualizerwithediting-get-coordinates-from-picked-poi
http://www.open3d.org/docs/release/python_api/open3d.visualization.VisualizerWithVertexSelection.html
    
@author: ParkanM
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import laspy
import matplotlib.pyplot as plt


#%% parameters

fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_rochefort_ch1903p_survey_v220418.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_chambrelien_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ge_2017_versoix_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry01_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry19_ch1903p_survey.las'
fpath = 'D:/Projects/lidar_tree_detection/input/pointclouds/ne_2016_boudry20e_ch1903p_survey.las'


# fname = os.path.basename(fpath)
fname = Path(fpath).stem
# fext =
fpath_out_pc_subset = "D:/Projects/lidar_tree_detection/input/pointclouds_subsets/%s_sub.las" % (fname)
fpath_out_tree_trunks = "D:/Projects/lidar_tree_detection/input/tree_trunks/%s_tree_trunks.shp" % (fname)



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


#%% visualization

n = len(pc.x)
xyz = np.stack([pc.x, pc.y, pc.z], axis=0).transpose((1, 0))
xyz_s = np.stack([x_s, y_s, z_s], axis=0).transpose((1, 0))

           
#%% visualize by acquisition RGB

rgb = np.stack([pc.red, pc.green, pc.blue], axis=0).transpose((1, 0)) / 65535


pcd  = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# plot
o3d.visualization.draw_geometries_with_editing([pcd])



vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # user picks points
picked_points = np.asarray(pcd.points)[vis.get_picked_points()]
vis.destroy_window()

vis.clear_picked_points() 


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


xx = np.array(pick_points(pcd))

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # user picks points
vis.destroy_window()
    
    

    
    