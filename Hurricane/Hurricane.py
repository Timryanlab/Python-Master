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
def gpu_image_blank(image_in, seperator, frame_shift):
    """
    A gpu based algorithm to set non-signal Super-res channels to 0 based on their
    alternating frame pattern. 
    """

    i,j,k = cuda.grid(3) # Get image coordinates
    pixw = pixel_width[0]
    m, n, o = image_in.shape # get row and column
    blank_side = (k + frame_shift[0])//2 # k is the current frame_shift = 0 by default, setting value to 1 switches behavior
    if (i >= m) or ( j >= n) or (k >=o): # If Any of the threads aren't working on a part of an image kick it out
        return
    if (blank_side == 0) and (j < seperator[0]): # If you've been marked as a left blank
        image_in[i,j,k] = 0                   #  frame and are a left side pixel set to 0
    if (blank_side == 1) and (j >= seperator[0]): # If you've been marked as a right blank
        image_in[i,j,k] = 0                   #  frame and are a right side pixel set to 0            

@cuda.jit()
def gpu_image_segment(image_in, ID_image, segmentation_array, pixel_width):
    #"""
    pass
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
def count_peaks(peak_image, blanking = 0, seperator = 190):
    """
    

    Parameters
    ----------
    peak_image : (m,n,o) binary image stack
        High pixels mark molecules to be counted and ID'd.
    blanking : int, optional
        Toggling variable to return different behavior. 
            The default is 0. 
            0 = No blanking
            1 = left channel blanks first 
            2 = right channel blanks first
    seperator : int, optional
        DESCRIPTION. The default is 190.

    Returns
    -------
    List([[float, float,int],]) = list([[row index, col index, frame index]])
    Returns a list representing the high pixel of likely molecular emissions

    """
    centers = np.argwhere(peak_image>0)
    mols = centers.shape[0
                         ]
    if blanking ==0:
        return centers
    shift = 0 # Frame shifting variable that allows us to switch behavior between frames
    if blanking == 2: shift = 1 # If we want the first frame to blank on the right, we set shift to 1
    remover = []
    # Remove IDs in blank regions
    for i in range(mols): # We'll loop over the list of centers and remove
        if ((centers[i,2] + shift) // 2 == 0 ) and (centers[i,1] < seperator): #Blanking the left frame
            remover.append(i) # If ID should be abandoned add index to list
        if ((centers[i,2] + shift) // 2 == 1 ) and (centers[i,1] >= seperator): #Blanking the right frame
            remover.append(i) # If ID should be abandoned add index to list
    return np.delete(centers,remover,0) # Return list of centers sans background channel IDs

def localize_file(file):
    """Localize file will do memory batch processing to prevent gpu overload
        this allows localize_images to focus on optimizing dataflow and prioritize 
        analysis speed
    """
    images = load_image_to_array(fpath + fname) # Load image into workspace
    localize_images(images)
    
def localize_images(images):
    
    #Load Images
    print('Image detected with m = {}, n = {}, and o = {}'.format(images.shape[0],images.shape[1],images.shape[2]))
    
    # Data Prep
    # Subtract Background using rolling ball subtraction
    print('Subtracting Background')
    gauss_sigma = 2.5
    rolling_ball_radius = 6
    rolling_ball_height = 6
    images_no_background, background_images = rolling_ball_subtraction(images,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
    del images
    del background_images
    print('Background Subtracted')
    
    # Peak Identification
    # Accepts background subtracted image and finds the likely spots for molecules  
    # returns a binary image representing areas of likely molecular emission
    
    binary_peak_image = find_peaks(images_no_background, 30, 3)
    
    # Prepare Molecular Data Arrays
    # Accepts binary_peak_image blanking and channel separation
    # returns centers = nx2 array containing the central pixel of n molecular emissions
    
    # Image Segmentation
    # Accepts: images_no_background, centers, pixel_width
    # Returns: molecules = the hurricane based data structure
    
    
    # Localization
    # Accept in an array where each row represents one localization
    
    # Data Cleanup
    
    # Save relevant data

def wavelet_denoise(image):
    """Non-optimized """
    pass

    
def find_peaks(image, threshold, pixel_width):
    """ Non-optimized gpu peak finding wrapper"""
    blockdim = (32,32) #Specify number of threads in a 2D grid
    image2 = np.empty_like(image)
    m,n,o = image_size(image)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    #image1 = wavelet_denoise(image)    
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
    # Because the Peak ID is parallelized it's just easier to keep it as part
    # Of the main pipeline despite it's false positives. We can reject those easily
    # On the other side of this by noting what frame we're on
    
    # Peak counting
    centers = count_peaks(image2, 1)
    print(centers)
    # Image Segmentation
    # This can be performed on a GPU and the output can be fed into the localization algorithm
    
    # Localization
    
    # Data Cleanup
    
    # Save relevant data
