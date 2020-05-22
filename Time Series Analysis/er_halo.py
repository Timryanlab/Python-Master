# -*- coding: utf-8 -*-
"""
Title: ER - Halo Analysis pipeline

This should be a pipeline that allows Ryan to sift through his data more efficiently

Concept:
Created on Fri May  8 15:24:58 2020

@author: Andrew Nelson
"""
#% Import Libraries
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\')

from ryan_image_io import *
from rolling_ball_subtraction import *
import os
import matplotlib.pyplot as plt

THRESHOLD = 25

    
#% Functions
def find_background_pixels(pixels_of_interest, radius = 4):
    """
    

    Parameters
    ----------
    pixels_of_interest : numpy image array
        Binary image showing pixels of interest to be used in calculation.
    radius : int, optional
        radius for deterimining the halo of the pixels of interest. The default is 4.

    Returns
    -------
    pixels_of_background : numpy image array
        Binary image showing pixels of background surrounding pixels of interest.

    """
    pixels_of_background = 0
    return pixels_of_background



def analyze_file_pair(file_pair):
    """
    The major pipeline for data analysis.

    Parameters
    ----------
    file_pair : list
        List of files that will be analyzed together. [ER_FILE, HALO_FILE] expected

    Returns
    -------
    None.

    """
    #Load Files
    er_image = load_image_to_array(file_pair[0])
    halo_image = load_image_to_array(file_pair[1])
    
    # Frame Average Halo
    halo_image_average = np.mean(halo_image,2)
    
    # Rolling ball subtraction of halo image
    halo_background_subtracted = rolling_ball_subtraction(halo_image_average,
                                                          gauss_sigma = 2.5,
                                                          rolling_ball_height = 6,
                                                          rolling_ball_radius = 6)
    
    # Create binary mask of halo cells above a threshold
    pixels_of_interest = halo_background_subtracted >= THRESHOLD
        
    # create the dilated zone around the pixels of interest
    pixels_of_background = find_background_pixels(pixels_of_interest)    
    
    masked_halo_image
    return None
    
def pair_image_files(files):
    """
    

    Parameters
    ----------
    files : list
        List of files from Ryan's Experiments.

    Returns
    -------
    split_files : list
        Returns a paired list of files between halo and ER channel.

    """
    # Get a list of files that contain either ER or Halo
    ER_files = list(filter(lambda x: "ER" in x, files)) 
    HALO_files = list(filter(lambda x: "HALO" in x, files)) 
    #prepare a list for storing results
    split_files = []
    
    for er_file in ER_files: # loop over ER files
        pre_er_index = er_file.index('ER') # Find the ER
        post_er_index = pre_er_index + 2 # Find the end of ER
        for halo_file in HALO_files: # Loop of Halo Files
            pre_halo_index = halo_file.index('HALO') # Find the HALO
            post_halo_index = pre_halo_index + 4 # Find the end of HALO
            # If file names match then add them as a pair
            if (er_file[:pre_er_index] + er_file[post_er_index:]) == (halo_file[:pre_halo_index] + halo_file[post_halo_index:]):
                split_files.append([er_file,halo_file])
            
    # Return what you find            
    return split_files

#% Main Workspace
if __name__ == '__main__':
    file_path = 'D:\\Image Processing Folder\\Background Subtractions for Ryan\\'
    file_save_path = 'D:\\Dropbox\\test batch\\Examples of ER-Halo Masking\\'
    files = grab_image_files(os.listdir(file_path))
    pairs = pair_image_files(files)
    file_pair = pairs[0]
    #Load Files
    er_image = load_image_to_array(file_path + file_pair[0])
    halo_image = load_image_to_array(file_path + file_pair[1])
    scale = 1.7
    show_as_image(halo_image[:,:,100])
    plt.title('Original Frame 0 of Halo')
    plt.figure()
     # Rolling ball subtraction of halo image
    save_array_as_image(  er_image, file_save_path + file_pair[0][:-5] + '.tif')
    save_array_as_image(halo_image, file_save_path + file_pair[1][:-5] + '.tif')
    halo_background_subtracted, background = rolling_ball_subtraction(halo_image,
                                                          gauss_sigma = 2.5,
                                                          rolling_ball_height = 6,
                                                          rolling_ball_radius = 6)
    show_as_image(halo_background_subtracted[:,:,10])
    plt.title("Background subtracted Halo image")
    plt.figure()
    # Frame Average background subtracted Halo
    save_array_as_image(  halo_background_subtracted, file_save_path + file_pair[1][:-5] + '_rbs.tif')
    
    halo_image_average = np.mean(halo_background_subtracted,2)
    
   
    show_as_image(halo_image_average)
    plt.title('Halo Image Average')
    plt.figure()

    # Array of Masks
    
    # Create binary mask of halo cells above a threshold
    pixels_of_interest = halo_image_average >= (np.mean(halo_image_average)+ 0.5* np.std(halo_image_average))
         
    # create the dilated zone around the pixels of interest
    pixels_of_background = find_background_pixels(pixels_of_interest)    
    
    show_as_image(pixels_of_interest)
    plt.title('Thresholded @ {}x std above the median'.format(np.round(i,2)))
    plt.figure()
    save_array_as_image(pixels_of_interest, file_save_path + file_pair[1][:-5] + '_pixels.png' )
