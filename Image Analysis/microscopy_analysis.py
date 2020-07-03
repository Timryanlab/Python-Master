# -*- coding: utf-8 -*-
"""
Title: Microscopy Image Analysis 
Concept: A library of regularly used image analysis commands
Created on Fri Jul  3 07:53:08 2020

@author: Andrew Nelson
"""
#% Import Libraries
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\AJN Lab\\Documents\\GitHub\\Python-Master\\Hurricane\\')

import ryan_image_io as rio
import matplotlib.pyplot as plt
#% Functions

#%
if __name__ == '__main__':
    file_path = 'C:\\Users\\AJN Lab\\Dropbox\\Data\\7-2-20 actin-halo in beas\\'
    #file_name = 'utrophin_7.tif'
    darkfield_name = 'mean_darkfield.tif'
    brightfield_name = 'mean_brightfield.tif'
    file_names = rio.grab_image_files(file_path)
    file_names = [file_name, darkfield_name, brightfield_name]
    files = [file_path + file for file in file_names]
    
    brightfield_image = rio.load_image_to_array(files[2])
    darkfield_image = rio.load_image_to_array(files[1])
    image_of_interest = rio.load_image_to_array(files[0])
    '''
    mean_darkfield = darkfield_image.mean(axis=2)
    mean_brightfield = brightbrightfield_image.mean(axis=2)
    normed_brightfield = mean_brightfield - mean_darkfield
    normed_brightfield /= normed_brightfield.max()
    '''
    corrected_image = image_of_interest.mean(axis=2)
    corrected_image -= darkfield_image[:,:,0]
    corrected_image /= brightfield_image[:,:,0]
    rio.show_as_image(corrected_image)
    
    rio.save_array_as_image(corrected_image , files[0][:-4] + '_corrected.tif')
    