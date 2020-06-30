# -*- coding: utf-8 -*-
"""
Title: Blank irrelevant channel

Concept: A quick program to blank the irrelevant channel in calibration data
Created on Mon Jun 29 07:35:31 2020

@author: Andrew Nelson
"""
#%% Import Libraries
from ryan_image_io import *
from localizations_class import *

#%% Functions

#%% Main Workspace
if __name__ == '__main__':
    colors = ['red', 'orange']
    for color in colors:
        file_path = 'D:\\Dropbox\\Data\\6-23-20 calibrations\\' + color +'\\'
        
        save_file_path = 'D:\\Dropbox\\Data\\6-23-20 calibrations\\Pre_prepped_images\\'
        image_file_list = grab_image_files(file_path)
        channel_split = Localizations().split
        for image_file in image_file_list:
            image_array = load_image_to_array(file_path + image_file)
            if color == 'red':
                image_array[:,channel_split:,:] = 0
            else:
                image_array[:,:channel_split,:] = 0
            save_file_name = save_file_path + image_file[:-4] + '_' + color + '_blank.tif'
            save_array_as_image(image_array, save_file_name)
