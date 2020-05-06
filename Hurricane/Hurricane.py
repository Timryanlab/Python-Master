# -*- coding: utf-8 -*-
"""
Title: Hurricane

Concept: This is the localization Pipeline the analyzes tiff / fits files and 
outputs localization data for subsequent analysis
Created on Tue May  5 10:21:58 2020

@author: Andrew Nelson
"""
#%% Import Libraries
from ryan_image_io import *
from rolling_ball_subtraction import *
from numba import cuda
from numba import vectorize, float64
import numpy as np
import matplotlib.pyplot as plt
import os

#%% GPU Accelerated Code
#%%
"""Begin GPU Code"""
@cuda.jit()
def gpu_peaks(image_in, image_out, thresh, pixel_width):
    """
    image_in.shape = image_out.shape 
        
    This code will look in the neighborhood of radius pixw to determine if it is a local maxima
    
    To do this the code will search a neighborhood in a raster scan while keep track of the 
    maxima value it finds above the threshold provided. Once found the ith and jth coordinate are noted
    if they correspond to the cuda i,j then the resulting image_out[i,j,k] = 1 and all rest = 0
    
    This is being followed from https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
    """

    i,j,k = cuda.grid(3) # Get image coordinates
    pixw = pixel_width[0]
    m, n, o = image_in.shape # get row and column
    if (i >= m) or ( j >= n) or (k >=o): # If Any of the threads aren't working on a part of an image kick it out
        return
    

    max_val = thresh[0]
    # Prime values relative to the cuda position
    max_i = i-pixw
    max_j = j-pixw
    
    
    # Loop over the neighborhood in a double loop
    for ii in range(i-pixw,i+pixw):
        for jj in range(j-pixw,j+pixw):
            #Ensure that you are only about to look at an image pixel
            if (ii >= 0) and (ii <m) and (jj>=0) and (jj <n):
                #Basic maximum pixel search, record index locations
                if image_in[ii,jj,k] >= max_val:
                    max_val = image_in[ii,jj,k]
                    max_i = ii
                    max_j = jj
    # Check to see if maximum
    if (max_i == i) and (max_j == j):
        image_out[i,j,k] = 1 #If local maxima turned out to be ideal
    else:
        image_out[i,j,k] = 0 #if not

@cuda.jit()
def gpu_image_segment(image_in, ID_image, segmentation_array, pixel_width):
    """
    This code will segment a 2*pixel_width+1 area around the identified pixel
    image_in = image to be segmented
    ID_image = 0s image with molecule ID at the center pixel of the neighborhood
    defined as above
    segmentation_array
    

    i,j,k = cuda.grid(3) # Get image coordinates
    if (ID_array[])
    """

#%% Functions

def localize_file(file):
    
    #Load Images
    images = load_image_to_array(fpath + fname) # Load image into workspace
    print('Image detected with m = {}, n = {}, and o = {}'.format(images.shape[0],images.shape[1],images.shape[2]))
    
    # Subtract Background using rolling ball subtraction
    print('Subtracting Background')
    gauss_sigma = 2.5
    rolling_ball_radius = 6
    rolling_ball_height = 6
    images_no_background, background_images = rolling_ball_subtraction(images,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
    print('Background Subtracted')
    
    # Peak Identification
    # Accepts background subtracted image and finds the likely spots for molecules  
    # returns a binary image representing areas of likely molecular emission
    
    binary_peak_image = find_peaks(images_no_background, 30, 3)
    
    # Prepare Molecular Data Arrays
    # Accepts binary_peak_image
    # returns centers = nx2 array containing the central pixel of n molecular emissions
    
    # Image Segmentation
    # Accepts: images_no_background, centers, pixel_width
    # Returns: molecules = the hurricane based data structure
    
    
    # Localization
    # Accept in an array where each row represents one localization
    
    # Data Cleanup
    
    # Save relevant data

def find_peaks(image, threshold, pixel_width):
    """ Non-optimized gpu peak finding wrapper"""
    blockdim = (32,32) #Specify number of threads in a 2D grid
    image2 = np.empty_like(image)
    m,n,o = image_size(image1)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    gpu_peaks[griddim, blockdim](np.ascontiguousarray(image1), np.ascontiguousarray(image2), np.ascontiguousarray(threshold), np.ascontiguousarray(pixel_width))
    return image2

#%% Localization Class    
#%% Main Workspace
if __name__ == '__main__':
    fpath = "D:\\Dropbox\\Data\\3-3-20 glut4-vglutmeos\\" # Folder
    fname = "cell10_dz20_r2.tif" # File name
    #%% Load Image
    im = load_image_to_array(fpath + fname, 20, 30) # Load image into workspace
    print(im.shape) # Print shape
    
    #%% Subtract Background using rolling ball subtraction
    print('Subtracting Background')
    gauss_sigma = 2.5
    rolling_ball_radius = 6
    rolling_ball_height = 6
    images_no_background, background_images = rolling_ball_subtraction(im,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
    print('Background Subtracted')
    del im # No reason to keep too much memory
    # Wavelet denoising for peak indentification
    # Saving this for later
    
    #%% Peak Identification
    #GPU Peak Identification
    image1 = images_no_background
    show_as_image(image1[:,:,0])
    plt.title('Background Subtracted Frame')
    plt.figure()
    image2 = find_peaks(image1, 25, 5)
    show_as_image(image2[:,:,0])
    plt.title('Peaks')
    # Image Segmentation
    
    # Localization
    
    # Data Cleanup
    
    # Save relevant data
