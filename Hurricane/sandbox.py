# -*- coding: utf-8 -*-
"""
Title: Exponetial workbook

Concept: A workbook to test the limits of the error function and choose parameters
that satisfy my requirements
Created on Sat May 16 08:51:13 2020

@author: Andrew Nelson
"""
#% Import Libraries
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\')
from ryan_image_io import *
import numpy as np
import cupy as cp
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from numba import cuda
from numba import vectorize, float64
#%%
@cuda.jit
def gpu_pixel_correlate(image_in, image_out, pixel_width):
    """
    image_in.shape = image_out.shape 
    )
    
    This will check the correlation of a pixel signal with it's neighbors and update 
    the pixel value with the value of maximum covariance in the neighborhood
    
    
    """
    
    def mean(x):
        n = len(x)
        s = 0
        for i in range(n):
            s+= x[i]
        return s/n
    
    def var(x):
        n = len(x)

        x2 = x
        mx = mean(x)
        s = 0
        for i in range(n):
            s += (x[i] - mx)**2
        
        return s/(n)
    
    def std(x):
        return var(x)**0.5
    
    def xcorr(x,y):
        mx = mean(x)
        my = mean(y)
        n = len(x)
        xy = 0
        for i in range(n):
            xy += x[i]*y[i]
        mxy = xy/n
        
        
        return (mxy - mx*my)/(std(x)*std(y))
    
    
    
    i,j = cuda.grid(2) # Get image coordinates
    
    m, n, o = image_in.shape # get row and column
    if (i >= m) or ( j >= n): # If Any of the threads aren't working on a part of an image kick it out
        return

    max_corr = -2
    
    c = -2
    if (i - 1 > -1):
        #c = xcorr(image_in[i,j,:],image_in[i-1,j,:])
        if c > max_corr: 
            max_corr = c
    if (i + 1 < m):
        #c = xcorr(image_in[i,j,:],image_in[i+1,j,:])
        if c > max_corr: 
            max_corr = c
    if (j - 1 > -1):
        c = xcorr(image_in[i,j,:],image_in[i,j-1,:])
        if c > max_corr: 
            max_corr = c
    if (j + 1 < n):
        c = xcorr(image_in[i,j,:],image_in[i,j+1,:])
        if c > max_corr: 
            max_corr = c


    image_out[i,j] = max_corr
    mean(image_in[:,1,1])
#%%
def load_and_mean_image(image_path):
    images = load_image_to_array(image_path)
    return np.mean(images,2)

def read_and_prep_mean_images(img_paths):
    imgs = [load_and_mean_image(img_path) for img_path in img_paths]
    img_array = np.array(imgs)
    output = preprocess_input(img_array)
    return(output)

def read_and_prep_images(img_paths):
    imgs = [load_and_mean_image(img_path) for img_path in img_paths]
    img_array = np.array(imgs)
    output = preprocess_input(img_array)
    return(output)

def pixel_correlation_scan(image1, pixel_width = 1):
    m,n,o = image_size(image1)
    image2 = np.zeros((m,n), dtype = float)
    blockdim = (32,32) #Specify number of threads in a 2D grid
    m,n,o = image_size(image1)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1) #Specify the block landscape
    gpu_pixel_correlate[griddim, blockdim](np.ascontiguousarray(image1),  np.ascontiguousarray(image2), pixel_width)
    return image2


#%% Main Workspace
if __name__ == '__main__':
    file_path = 'D:\\Image Processing Folder\\Background Subtractions for Ryan\\'
    file_names = grab_image_files(os.listdir(file_path))
    files = [file_path + file_name for file_name in file_names]
    
    #output = read_and_prep_images(img_paths = files )
    image1 = load_image_to_array(files[0])
    #%%
    image2 = pixel_correlation_scan(image1, 1)
    show_as_image(image2)