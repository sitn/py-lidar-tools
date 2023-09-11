# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:53:59 2023

@author: ParkanM
"""

# Import libraries
import os
import pathlib
import glob
import re

#%% parameters

dir_in = 'D:/Data/pointclouds/2023/all/*.laz'


#%% create empty LAS file

files_in = glob.glob(dir_in)


n = len(files_in)

# get file names
fnames = [os.path.basename(x).split(os.extsep)[0] for x in files_in]
occurences = [fnames.count(x) for x in fnames]
skip = [False for i in range(n)]


for index, fpath_src in enumerate(files_in):
    
    print("***********************************************")
    print("Processing file %u / %u: %s" % (index+1, n, fpath))
    
    res = re.search('([0-9]{7}-[0-9]{7})', fname_src)
    
    if res:
        fname_dst = re.sub('-', '_', res.group())
        fpath_dst = re.sub('LCDF.*([0-9]{7}-[0-9]{7})', fname_dst, fpath_src)
        os.rename(fpath_src, fpath_dst)



fpath_src = files_in[348]

res = re.search('([0-9]{7}-[0-9]{7})', fpath_src)
fname_dst = re.sub('-', '_x_', res.group())
fpath_dst = re.sub('LCDF.*([0-9]{7}-[0-9]{7})', fname_dst, fpath_src)
os.rename(fpath_src, fpath_dst)


fpath_dst = 'D:/Data/pointclouds/2023/all/2547000_x_1213000.laz'



fpath = "D:/Data/pointclouds/2023/all/LCDF_MN95_NF02_2551500-1216000.laz"
print(re.search('([0-9]{7}-[0-9]{7})', fpath))





fname = "LCDF_MN95_NF02_2551500-1216000.laz"    
    
# check if string present on a current line
result = re.match('?([0-9]{7}-[0-9]{7})', fname)
result = re.match('\b[0-9]{3}\b', fname)

result = re.match('^.{1,}', fname)

result = re.match('^.{1,}', fname)


result = re.match('.*([0-9]{7}-[0-9]{7})', fname)

print(re.search('([0-9]{7}-[0-9]{7})', fname))





lines = re.sub(columns[4], columns[4][2:], fname)

result = re.match('^LCDF', fname)

