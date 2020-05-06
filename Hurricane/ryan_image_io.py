# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:55:54 2020
Image File I/O
This library will allow Ryan Lab codes to handle both Fits and Tiff images for 
subsequent analysis
@author: andre
"""
#%% Import Section
import os
from astropy.io import fits
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%% Helper Functions

def image_size(image):    
    """Quick function to ensure we always get 3 indices"""
    #image size
    try:
        m,n,o = image.shape    
    except:
        m, n = image.shape
        o = 1
    return m,n,o

def load_fits_to_array(file, start=0, stop=0):
    """
    Allows for the loading of fits files by specified frame
    If no frame is specified the full file is loaded as a numpy array
    """
    with fits.open(file) as im1: # Ensure we close the file when we're done reading the data
        if ((start != 0) and (stop == 0)): #In this condition we want 1 frame, return the frame corresponding to start
                im_stack = im1[0].data[start,:,:].astype('float64') # load single frame
                # Reshape fits to expected image indexing of (m,n,o)
                m,n = im_stack.shape 
                o = 1
                image_stack = np.zeros((m,n,1))
                image_stack[:,:,0] = im_stack[:,:]
        if (start != 0) and (stop != 0): # Want a slice of frames
                if start != stop:
                    image_stack = im1[0].data[start:stop+1,:,:].astype('float64') # load frames in array from start to stop
                    # Reshape fits to expected image indexing of (m,n,o)
                    o,m,n = image_stack.shape 
                else:
                    im_stack = im1[0].data[start,:,:].astype('float64') # load single frame
                    # Reshape fits to expected image indexing of (m,n,o)
                    m,n = im_stack.shape 
                    o = 1
                    image_stack = np.zeros((m,n,1))
                    image_stack[:,:,0] = im_stack[:,:]
        if (start == 0) and (stop != 0): # Hear we specify the end frame only
                image_stack = im1[0].data[:stop,:,:].astype('float64') # load frames in array from start to stop
                # Reshape fits to expected image indexing of (m,n,o)
                o,m,n = image_stack.shape 
        if (start == 0) and (stop == 0): # Here we want the whole data set
                image_stack = im1[0].data.astype('float64') # load frames in array from start to stop
                # Reshape fits to expected image indexing of (m,n,o)
                o,m,n = image_stack.shape 
        
        #Preallocate Array for image out
        image_out = np.zeros((m,n,o))
        
        
        # Different rules for a single frame image
        if o > 1:
            for i in range(o):
                image_out[:,:,i] = image_stack[i,:,:]
                
                return image_out
        else:
            return image_stack

def load_image_to_array(file, start = 0, stop = 0): 
    filename, file_extension = os.path.splitext(file) # split filename to check extension
    if file_extension == '.tif':
        return load_tiff_to_array(file, start, stop)
    if file_extension == '.fits':
        return load_fits_to_array(file, start, stop)  
    
def load_tiff_to_array(fname, start = 0, stop = 0):
    """This function will take in a string as a file location and file name
       It will return a Numpy Array of the image at this location"""
    im = Image.open(fname) #Initial call to image datafile
    
    try:
        #Get shape of image
        m,n = np.shape(im)
        o = im.n_frames 
    except:
        o = 1
    
    if stop > o:
        stop = 0
    
    if (start != 0) and (stop == 0): #In this condition we want 1 frame, return the frame corresponding to start
        im.seek(start) #Seek to frame i in image stack
        im_array= np.array(im)
        m,n =im_array.shape
        image_array = np.zeros((m,n,1))
        image_array[:,:,0] = im_array[:,:]
        
    if (start != 0) and (stop != 0): # Want a slice of frames
        if start != stop:
            # Allocate variable memory for image
            image_array = np.zeros((m,n,stop-start+1))
            for i in range(stop-start+1):
                im.seek(i+start) #Seek to frame i in image stack
                image_array[:,:,i]= np.array(im) #Populate data into preallocated variable 
        else:
            im.seek(start) #Seek to frame i in image stack
            im_array= np.array(im)
            m,n =im_array.shape
            image_array = np.zeros((m,n,1))
            image_array[:,:,0] = im_array[:,:]
    if (start == 0) and (stop != 0): # Hear we specify the end frame only
        image_array = np.zeros((m,n,stop))
        for i in range(stop):
            im.seek(i) #Seek to frame i in image stack
            image_array[:,:,i]= np.array(im) #Populate data into preallocated variable
            
    if (start == 0) and (stop == 0): # Here we want the whole data set
        image_array = np.zeros((m,n,stop))
        for i in range(o):
            im.seek(i) #Seek to frame i in image stack
            image_array[:,:,i]= np.array(im) #Populate data into preallocated variable          

    return image_array # return numpy image array

def save_array_as_image(array, file):
    filename, file_extension = os.path.splitext(file) # split filename to check extension
    if file_extension == '.tif':
        save_array_to_tiff(array, file)
    if file_extension == '.fits':
        save_array_to_fits(array, file)
    
def save_array_to_fits(array, file):
    m,n,o = image_size(array)
    re_image = np.zeros((o,m,n))
    for i in range(o):
        re_image[i,:,:] = array[:,:,i]
    hdu = fits.PrimaryHDU(re_image)
    hdu.writeto(file)

def save_array_to_tiff(array, file):
    """Function to save an array as a multipage tiff"""
    m,n,o = image_size(array)
    Im = []
    for i in range(o):
        Im.append(Image.fromarray(array[:,:,i]))
    Im[0].save(file, save_all=True,
               append_images=Im[1:])
    # This appears to work properly
    
def show_as_image(image_frame, title = None):
    """We expect a 2D numpy array and will output the visual grayscale result"
    """
    plt.imshow(image_frame)
    plt.title(title)
    
def grab_image_files(folder):
    """Takes a list of folder contents and returns a list of images only"""
    files = []
    for file in folder:
        if file.find('.tif') + file.find('.fits') > -2:
            files.append(file)
    return files

##%% Main Section Workspace
if __name__ == '__main__':
    pass