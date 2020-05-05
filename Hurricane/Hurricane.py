# -*- coding: utf-8 -*-
"""
Hurricane Function Library

This is a temporary script file.
"""
import numpy as np
from PIL import Image
from numba import cuda

def load_image_to_array(fname):
    """This function will take in a string as a file location and file name
       It will return a Numpy Array of the image at this location"""
    im = Image.open(fname) #Initial call to image datafile
    #Get shape of image
    m,n = np.shape(im)
    o = im.n_frames        
    # Allocate variable memory for image
    tiff_array = np.zeros((m,n,o))
    for i in range(o):
        im.seek(i) #Seek to frame i in image stack
        tiff_array[:,:,i]= np.array(im) #Populate data into preallocated variable
    return tiff_array # return numpy tiff array

# Select Image to Analyze
fpath = "D:\\Dropbox\\Data\\3-3-20 glut4-vglutmeos\\" # Folder
fname = "cell10_dz20_r2.tif" # File name

im = load_image_to_array(fpath + fname) # Load image into workspace
print(im.shape) # Print shape

