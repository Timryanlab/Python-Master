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
from rolling_ball_subtraction import *
mpl.rcParams['agg.path.chunksize'] = 100000
import open3d as o3d
#%% Functions
def visualize_3d_localizations(molecules):
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

def visualize_2d_projection(molecules, size = 10):
    molecules.separate_colors()
    plt.scatter(molecules.xf_red,
                molecules.yf_red, 
                color = 'red',
                s = size)
    plt.scatter( molecules.xf_orange,
                molecules.yf_orange, 
                color = 'blue',
                s = size)

def show_localized_areas(image,center, pixel_width = 5):
    #This is a function to overlay highlights around the selected pixels
    m,n = center.shape
    plt.imshow(image)
    for i in range(m):
        x = np.array([center[i,1] - pixel_width ,
                      center[i,1] + pixel_width ,
                      center[i,1] + pixel_width ,
                      center[i,1] - pixel_width ,
                      center[i,1] - pixel_width ])
        y = np.array([center[i,0] - pixel_width ,
                      center[i,0] - pixel_width ,
                      center[i,0] + pixel_width ,
                      center[i,0] + pixel_width ,
                      center[i,0] - pixel_width ])
        
        plt.plot(x,y, color='red')
    plt.show()
        
        
    #%%
if __name__ == '__main__':
    file_path = "D:\\Dropbox\\Data\\7-17-20 beas actin\\utrophin\\"
    file_name = 'cell_4_dz_20_r_0_3_localized.pkl'
    result = load_localizations(file_path + file_name)
    #result.make_z_corrections()
    visualize_2d_projection(result, 0.1)
