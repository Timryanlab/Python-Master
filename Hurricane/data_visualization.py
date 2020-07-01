# -*- coding: utf-8 -*-
"""
Title: Data Visualization Library
Concept: A library the handles the visualization of localization data
Created on Wed Jul  1 09:07:52 2020

@author: Andrew Nelson
"""
#%% Import Libraries
from localizations_class import *
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import open3d as o3d
#%% Functions
def visualize_localizations(molecules):
    molecules.separate_colors()
    xyz = np.empty((molecules.xf.shape[0],3))
    point_colors = np.empty((molecules.xf.shape[0],3))
    for i in range(molecules.xf_orange.shape[0]):
        if np.abs(molecules.zf_orange[i]) <= 0.5:
            xyz[i,0] = molecules.xf_orange[i] 
            xyz[i,1] = molecules.yf_orange[i]
            xyz[i,2] = molecules.zf_orange[i]
            point_colors[i,:] = [0, 0, 1]
        else:
            xyz[i,0] = 0 
            xyz[i,1] = 0
            xyz[i,2] = 0
            point_colors[i,:] = [0, 0, 1]
    
    for i in range(molecules.xf_red.shape[0]):
        ii = i + molecules.xf_orange.shape[0]
        if np.abs(molecules.zf_red[i]) <= 0.5:
            xyz[ii,0] = molecules.xf_red[i] 
            xyz[ii,1] = molecules.yf_red[i]
            xyz[ii,2] = molecules.zf_red[i]
            point_colors[ii,:] = [1, 0, 0]        
        else:
            xyz[ii,0] = 0 
            xyz[ii,1] = 0
            xyz[ii,2] = 0
            point_colors[ii,:] = [0, 0, 1]
    
    index_of_numbers = np.isfinite(xyz[:,0])
    selected_point_colors = point_colors[index_of_numbers,:]
    selected_xyz = xyz[index_of_numbers,:]
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(selected_xyz)
    pcd.colors = o3d.utility.Vector3dVector(selected_point_colors)
    o3d.visualization.draw_geometries([pcd])
    
#%%
if __name__ == '__main__':
    file_path = 'C:\\Users\\AJN Lab\\Dropbox\\Data\\6-24-20 actin in BEAs\\life_act\\'
    file_name = 'BEA_cell_1_2_localized.pkl'
    result = load_localizations(file_path + file_name)
    
    snr = np.empty((result.xf.shape))
    for i in range(len(snr)):
        snr[i] = result.N[i] / (result.N[i] + (result.pixel_width +1)**2*result.o[i])**0.5
    plt.hist(snr, bins = 100)
    visualize_localizations(result)
