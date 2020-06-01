# -*- coding: utf-8 -*-
"""
Title:

Concept:
Created on Fri May 22 14:59:00 2020

@author: Andrew Nelson
"""
#%% Import Libraries

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\')
from ryan_image_io import *
import os
import matplotlib.pyplot as plt
import cupy as cp
#%% Functions
loaded_from_source = r'''
extern "C" 

__device__ double mean(double x, int o){
    double s = 0;
    for(int i = 0; i<o;i++){
        s += x[i];
    }
    return s/(double)o;
}


 __global__ void correlation_filter(const double* psf_array, double* out_array, int pix, int m, int n, int o) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = ty + blockIdx.y*blockDim.y;
    int j = tx + blockIdx.x*blockDim.x;
    
    if( i < m && j < n){ // determine if you're operating on an appropriate pixel
        out_array[]
    }
    
}'''


#%% Main Workspace
if __name__ == '__main__':
    pass
