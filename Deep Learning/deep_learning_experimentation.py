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

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


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

    cp.cuda.compiler.compile_with_cache
#%% Main Workspace
    """ Goal for the day is to show we can discriminate between background, raw, and 
if __name__ == '__main__':
    file_path = 'D:\\Image Processing Folder\\Background Subtractions for Ryan\\'
    file_names = grab_image_files(os.listdir(file_path))
    files = [file_path + file_name for file_name in file_names]
    
    #output = read_and_prep_images(img_paths = files )
    image1 = load_image_to_array(files[0])
    #%%
    image2 = pixel_correlation_scan(image1, 1)
    show_as_image(image2)