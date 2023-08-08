# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:05:00 2023

@author: ParkanM
"""

#%% Import libraries

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


#%% parameters

fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/"

fig_name = 'ge_2017_versoix'
name = 'Versoix (GE) - 2017'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ge_2017_versoix_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ge-2017-versoix-tree-trunks-v02.shp"

fig_name = 'ne_2016_boudry01'
name = 'Boudry 01 (NE) - 2016'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry01_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2016-boudry-by01-tree-trunks-v02.shp"

fig_name = 'ne_2016_boudry19'
name = 'Boudry 19 (NE) - 2016'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry19_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2022-boudry19-tree-trunks-v02.shp"

fig_name = 'ne_2016_boudry20'
name = 'Boudry 20 (NE) - 2016'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry20e_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2016-boudry20e-tree-trunks-v02.shp"


fig_name = 'ne_2022_boudry01'
name = 'Boudry 01 (NE) - 2022'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry01_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2022-boudry-by01-tree-trunks-v02.shp"


fig_name = 'ne_2022_boudry19'
name = 'Boudry 19 (NE) - 2022'
fpath_in_ref = "C:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry19_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "C:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2022-boudry19-tree-trunks-v02.shp"

fig_name = 'ne_2022_boudry20'
name = 'Boudry 20 (NE) - 2022'
fpath_in_ref = "D:/Projects/lidar_tree_detection/input/tree_trunks/ne_2016_boudry20e_ch1903p_survey_tree_trunks.shp"
fpath_in_pred = "D:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/als-ne-2022-boudry20e-tree-trunks-v02.shp"


#%% read input files

gdf_ref = gpd.read_file(fpath_in_ref)
gdf_pred = gpd.read_file(fpath_in_pred)


#%% find nearest neighbours

nn = gdf_pred.sindex.nearest(gdf_ref.geometry, return_all=False, max_distance=1, return_distance=True)

idxn_nn = np.transpose(nn[0]) # index of nearest neighbour
d_nn = nn[1] # distance to nearest neighbour

dbh_ref = gdf_ref.iloc[idxn_nn[:,0]].diameter
dbh_pred = gdf_pred.iloc[idxn_nn[:,1]].RADIUS * 200

h_ref = gdf_ref.iloc[idxn_nn[:,0]].height
h_pred = gdf_pred.iloc[idxn_nn[:,1]].LENGTH

# create dictionnary
d = {}
d['d_nn'] = d_nn
d['dbh_ref'] = dbh_ref.array
d['dbh_pred'] = dbh_pred.array

# create dataframe from dictionnary
df_stats = pd.DataFrame(data=d)

# filter
idxl = (df_stats.dbh_ref != 0) & (df_stats.dbh_pred != 0)


#%% error assessment

n_obs = len(df_stats[idxl]['dbh_pred'])

r_corr_sq = df_stats[idxl]['dbh_pred'].corr(df_stats[idxl]['dbh_ref'])**2

corr_matrix = np.corrcoef(df_stats[idxl]['dbh_ref'], df_stats[idxl]['dbh_pred'])
corr = corr_matrix[0,1]
R_sq = corr**2


dbh_diff = df_stats[idxl]['dbh_pred'] - df_stats[idxl]['dbh_ref']

dbh_diff_rel = dbh_diff / df_stats[idxl]['dbh_ref']


#%% plot correlation
                                        

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(df_stats[idxl]['dbh_ref'], df_stats[idxl]['dbh_pred'], s=1.2, c="black", marker=".", alpha=1)

# ax.plot([25, 90], [25, 90], ls="--", c=".3")
plt.axline((25, 25), slope=1, color="red", linestyle=(0, (10, 10)), linewidth=0.5)

ax.set_xlim(0, 100)
ax.set_xlabel('True diameter [cm]')

ax.set_ylim(0, 100)
ax.set_ylabel('Predicted diameter [cm]')

title = "FLAI - %s\nn=%s, r2=%s" % (name, n_obs, round(r_corr_sq,2))

ax.set_title(title, fontdict=None, loc='center', pad=None)
       
ax.set_aspect('equal')

#↔ ax.set(xlim=(-3, 3), ylim=(-3, 3))

# save figure
fig.savefig('D:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/%s_dbh_correlation.png' % fig_name, dpi=250)


#%% plot bias

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(df_stats[idxl]['dbh_ref'], 100*dbh_diff_rel, s=1.2, c="black", marker=".", alpha=1)

plt.axline((0, 0), slope=0, color="black", linestyle=(0, (10, 10)), linewidth=0.5)

ax.set_xlim(0, 100)
ax.set_xlabel('True diameter [cm]')

ax.set_ylim(-75, 75)
ax.set_ylabel('Relative bias [%]')

title = "FLAI - %s\nn=%s" % (name, n_obs)

ax.set_title(title, fontdict=None, loc='center', pad=None)
       
ax.set_aspect('equal')

#↔ ax.set(xlim=(-3, 3), ylim=(-3, 3))

# save figure
fig.savefig('D:/Projects/lidar_tree_detection/output/FLAI/tree_trunks/%s_dbh_bias.png' % fig_name, dpi=250)









idxn_nn = np.transpose(gdf_test.sindex.nearest(gdf_ref.geometry))

# distance between points
d_nn = gdf_ref.distance(gdf_test.iloc[idxn_nn[:,1]], align=False)






xyz_ref = gdf_ref.get_coordinates(include_z=True, ignore_index=False, index_parts=False) 
xyz_test = gdf_test.get_coordinates(include_z=True, ignore_index=False, index_parts=False)


result = xyz_test.iloc[idxn_nn[:,1]]










gdf_test.radius[idxn_nn[:,1]] * 2


gdf_test.columns



xyz_ref[diameter]

idxl_valid










idxn_nn = np.transpose(gdf_ref.sindex.nearest(gdf_test.geometry))

points_df.distance(points_df2)





xyz_test

xyz_test.filter(items = idxn_nn, axis=0)
df[df.index.isin(my_list)]



df1 = pd.DataFrame(xyz_test)

df1.x[idxn_nn]


xy_ref = gdf_ref[idxn_nn].x
xy_nn = gdf_test.columns

d_nn gdf_ref[idxn_nn].geometry


pts3 = gdf_test.geometry.unary_union

def near(point, pts=pts3):
     # find the nearest point and return the corresponding Place value
     nearest = gdf_ref.geometry == nearest_points(point, pts)[1]
     return gdf_ref[nearest].Place.get_values()[0]
 
gpd1['Nearest'] = gdf_test.apply(lambda row: near(row.geometry), axis=1)