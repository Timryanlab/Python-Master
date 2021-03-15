# -*- coding: utf-8 -*-
"""
Title: ROI updating
Concept: Create an algorithm that adjusts an ROI for imageJ in such a way that
prevents researchers from having to manually update their data
Created on Mon Mar  8 08:17:19 2021

@author: Andrew Nelson
"""
#%% Import Libraries
import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, 'C:\\Users\\AJN Lab\\Documents\\GitHub\\TR-Python-Master\\Hurricane\\')
sys.path.insert(1, 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\')
sys.path.insert(1, 'C:\\Users\\AJN Lab\\Documents\\GitHub\\TR-Python-Master\\imageJ rois\\')

import ryan_image_io as rio
import matplotlib.pyplot as plt
import ryan_rois as rr
import numpy as np


#%% Functions

#%% Workshop
if __name__ == '__main__':
    folder = 'D:\\Dropbox\\ROI shifting\\Manually tracked movement\\210301 d2K\\'
    image_name = '210301 d2K NH4Cl glucose.fits'
    zip_name = 'RoiSet 210301 d2K NH4Cl.zip'
    zip_file = folder + zip_name
    rois = rr.read_roi_zip(zip_file)
    image_file = folder + image_name
    image_stack = rio.load_image_to_array(image_file)
    #%% post image loading
    
    
    for roi in rois:
        print(rois[roi]['left'])
        print(rois[roi]['top'])