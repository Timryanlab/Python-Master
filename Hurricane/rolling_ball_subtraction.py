# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:06:31 2020
A Cuda Library for Image Analysis
Andrew Nelson
@author: Andrew Nelson
"""
#%% Import Section
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import os
from ryan_image_io import *
import cupy as cp

#%% Important constants
DEVICE_LIMIT = 190*388*2000 #Empirical hard limit in maximum pixels of a setup running a Geforce 1070 TI as the sole graphics card
                            #Due to excessive luxuries of time we'll assume all datasets are dtype = np.float64 
                    

#%%
"""Begin GPU Code"""
@cuda.jit
def gpu_convolve(image_in, kernel, image_out, im_type = 0):
    """
    image_in.shape = image_out.shape 
    kernel.shape = (2*n+1,2*n+1) for some n = (0, 1, 2, ...)
    
    This will convolve image_in with the kernel and return the result as image_out
    The implementation is to wokr on a 3D stack of images conlvoved with a 2D kernel
    
    This is being followed from https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
    """

    i,j,k = cuda.grid(3) # Get image coordinates
    
    m, n, o = image_in.shape # get row and column
    if (i >= m) or ( j >= n) or (k >=o): # If Any of the threads aren't working on a part of an image kick it out
        return
    # Determine the size of the kernel
    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2
    
    #Prime the integration variable
    s = 0
    if (im_type == 2): #dilation
        s_m = 0
    if (im_type == 1): # erosion
        s_m = 1000

    # loop over image ind"icies
    for x in range(kernel.shape[1]):
        for y in range(kernel.shape[0]):
            # get initial positions for kernel scan
            i_k = i - y + delta_rows
            j_k = j - x + delta_cols
            #if index is inside image sum convolution
            if (i_k >= 0) and (j_k >= 0) and (i_k < m) and (j_k < n):
                
                # Makshift switch statement based on type of filter being applied
                if (im_type == 0): #convolution
                    s += image_in[i_k,j_k,k]*kernel[y,x] 
                    
                    
                if (im_type == 1): # erosion
                    s = image_in[i_k,j_k,k] - kernel[y,x] # subtract neighborhood
                    if s < s_m: # look for minimum value
                        s_m = s
                        
                if (im_type == 2): # dilation
                    s = image_in[i_k,j_k,k] + kernel[y,x]
                    if s > s_m:
                        s_m = s
                        
    #Switch statement for 
    if (im_type == 0): #convolution result
        image_out[i,j,k] = s
    if (im_type == 1) or (im_type == 2): # erosion or dilation result
        image_out[i,j,k] = s_m

@cuda.jit()
def gpu_image_subtraction(image1, image2, image3):
    """3D Image Subtraction Algorithm"""
    i,j,k = cuda.grid(3)
    m,n,o = image1.shape
    if (i>=m) or (j>=n) or (k >= o):
        return
    else:
        if (image1[i,j,k] - image2[i,j,k]) >0:
            image3[i,j,k] = image1[i,j,k] - image2[i,j,k]
        else:
            image3[i,j,k] = 0
    #image3[i,j,k] = j
    
#%%
"""Non GPU Code Below this Line"""




def make_gaussian_kernel(sigma = 2.5, kernel_radius = 5): # write a function to create a gaussian kernel whose width is 2sigma + 1
    """
    Returns a volume normalized 2D Gaussian of sigma = sigma and
    image size equal to  2*ceil(sigma)+ 1 pixels x 2*ceil(sigma)+ 1 pixels
    """
    xline = np.linspace(-kernel_radius,kernel_radius, 2*kernel_radius + 1) # Using sigma max ensures encorporating the 2 sigma radius of the gaussian
    yline = np.exp(-xline*xline/(2*sigma*sigma))
    gauss =np.outer(yline,yline)
    return gauss / gauss.sum()

def run_the_conv(image1, kernel, image2, im_type = 0):
    """This is the function that calls the GPU calculations"""
    blockdim = (32,32) #Specify number of threads in a 2D grid
    m,n,o = image_size(image1)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    gpu_convolve[griddim, blockdim](np.ascontiguousarray(image1), np.ascontiguousarray(kernel), np.ascontiguousarray(image2), im_type)
    gpu_convolve[griddim, blockdim](np.ascontiguousarray(image1), np.ascontiguousarray(kernel), np.ascontiguousarray(image2), im_type)
    
def run_the_fast_conv(image1, kernel, image2, im_type = 0):
    """This wrapper runs the gpu calls for the preallocated device arrays"""
    blockdim = (32,32) #Specify number of threads in a 2D grid
    m = image1.shape[0]
    n = image1.shape[1]
    try:
        o = image1.shape[2]
    except:
        o = 1
        
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    gpu_convolve[griddim, blockdim](image1, kernel, image2, im_type) # Make GPU Call
    
def run_the_subtraction(image1, image2, image3):
    """ Wraper for GPU Subtraction algorithm"""
    blockdim = (32,32)
    griddim = (image1.shape[0]// blockdim[0] + 1, image1.shape[1]//blockdim[1] + 1,image1.shape[2])
    gpu_image_subtraction[griddim, blockdim](np.ascontiguousarray(image1),np.ascontiguousarray(image2),np.ascontiguousarray(image3))

def run_the_fast_subtraction(image1, image2, image3):
    """ Optimized subtraction algorithm for device based arrays"""
    blockdim = (32,32)
    m = image1.shape[0]
    n = image1.shape[1]
    try:
        o = image1.shape[2]
    except:
        o = 1
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o)
    gpu_image_subtraction[griddim, blockdim](image1,image2,image3)
    
def slow_rolling_ball(images, radius = 1):
    """
    This function will take a stack of images and blur this with a guassian filter of sigma radius = radius
    It will return a stack of images indexed identically to the input stack
    """
    x = 75
    y = 150
    kernel = make_gaussian_kernel(radius) #Make normalized gaussian kernel
    images_out = np.empty_like(images) # Preallocate array for images
    images_2 = np.empty_like(images)
    print(kernel)
    print("This is the input image")
  #  print(images[(y-5):(y+5),(x-5):(x+5),0])
    
    #Convolution
    run_the_conv(images, kernel, images_out,0) # perform the convolution+
    # The kernel is performing properly for a convolution with a gaussian kernel
    show_as_image(images[:,:,0])
    plt.title("Original")
    plt.figure()
    show_as_image(images_out[:,:,0])
    plt.title("Blurred")
    plt.figure()

    
    #Morphological Opening
    kernel = make_a_sphere(radius = 6, height = 6)
    # The sphere is being created as expected
    #Erosion
    run_the_conv(images_out,kernel,images_2,1)
    show_as_image(images_2[:,:,0])
    plt.title("Eroded")
    plt.figure()
    print("This is the ball")
    print(kernel)
    print("This is the eroded image")
   # print(images_2[(y-5):(y+5),(x-5):(x+5),0])
    run_the_conv(images_2,kernel,images_out,2)
    # The entire convolution algorithm appears to be working as intended, minor
    # differences in rball will result in differences from matlab calculation
    print("This is the dilate image")
    #print(images_2[(y-5):(y+5),(x-5):(x+5),0])
    plt.figure()
    # All code related to the convolution appears to be working currently
    
    show_as_image(images_out[:,:,0])
    plt.title("Dilated (Background)")
    
    run_the_subtraction(images,images_out,images_2)
    plt.figure()
    show_as_image(images_2[:,:,0])
    #Validation the subtraction image works properly (only negative values will appear in final version)
    #print(images[(y-5):(y+5),(x-5):(x+5),0]-images_out[(y-5):(y+5),(x-5):(x+5),0]-images_2[(y-5):(y+5),(x-5):(x+5),0])
    plt.title("Bkgn Subtracted")
    ##
    
    return images_2

def send_to_device(input_variable):
    return cuda.to_device(np.ascontiguousarray(input_variable, dtype = input_variable.dtype))

def make_a_sphere(radius = 6, height = 6):
    """Creates a logical array for use in morphological openings and closings"""
    x = np.linspace(-radius,radius,2*radius + 1)
    xx, yy = np.meshgrid(x, x)
    R = np.sqrt(xx**2 + yy**2)
    H = np.empty_like(R)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if R[i,j] > radius+1:
                H[i,j] = -100000000
            else:
                H[i,j] = height*np.sqrt(radius**2 - np.min([radius**2, R[i,j]**2])) /radius
                #H[i,j] = 1
    return H

def gpu_rolling_ball(d_images, d_images2, d_images3, d_kernel, d_sphere_kernel):
    """Takes in the image arrays used for the rolling ball subtraction
        this is is the fundamental pipeline of the algorithm
    """
    #Convolution
    run_the_fast_conv(d_images, d_kernel, d_images3,0) # perform the convolution+
    #Erosion
    run_the_fast_conv(d_images3, d_sphere_kernel, d_images2,1)
    #Dilation
    run_the_fast_conv(d_images2, d_sphere_kernel, d_images3,2)
    #ubtraction        
    run_the_fast_subtraction(d_images, d_images3, d_images2)
    #Completed Run

def rolling_ball_subtraction(images, gauss_sigma = 2.5, rolling_ball_radius = 6, rolling_ball_height = 6):
    """Performs a rolling ball subtraction on a stack of images
    
       Rolling ball parameters are set through gauss_sigma, rolling_ball_radius, and rolling_ball_height
       Returns 2 stacks, First is the background subtracted images, the second is the stack of backgrounds
    """
    # Get Image Parameters
    m,n,o = image_size(images)
    pixels = m*n*o #assuming 64 bit images from now on
    
    #Morphological Opening structures
    kernel = make_gaussian_kernel(gauss_sigma) #Make normalized gaussian kernel
    # Make a grayscale 'ball'
    sphere_kernel = make_a_sphere(radius = rolling_ball_radius, height = rolling_ball_height) 
    
    #Host side return array allocation
    images2 = np.empty_like(images)
    images3 = np.empty_like(images) # Preallocate array for images
    
    d_kernel = send_to_device(kernel)
    d_sphere_kernel = send_to_device(sphere_kernel)
    #If Statement Based on memory size
    if pixels < DEVICE_LIMIT: #If Memory is low enough run as one chunk 
        #Device Memory Allocation
        d_images = send_to_device(images)
        d_images2 = send_to_device(images2)
        d_images3 = send_to_device(images3)
        #GPU Call
        gpu_rolling_ball(d_images, d_images2, d_images3, d_kernel, d_sphere_kernel)
        #Copy final arrays back to host side
        images2 = d_images2.copy_to_host()
        images3 = d_images3.copy_to_host()
        #
        
        
        
    else: #If the stack is too large handle it in chunks
        rounds = pixels // DEVICE_LIMIT + 1# Divide into manageable sizes
        chunk = o // (rounds)   # Determine dividing chunk size of stacks
        for i in range(rounds+1): # Loop over number of times needed to chunk through data set
            # Ensure we don't go over our stack size
            stride = np.min((o,(i+1)*chunk)) 

            #Parse the image into a subset
            sub_images = images[:,:,i*chunk:stride]
            # Preallocate array for images (This is in addition to the 'return' arrays)
            sub_images3 = np.empty_like(sub_images) 
            sub_images2 = np.empty_like(sub_images)
            #Device Memory Allocation
            d_images = send_to_device(sub_images)
            d_images2 = send_to_device(sub_images2)
            d_images3 = send_to_device(sub_images3)

            #GPU Call
            gpu_rolling_ball(d_images, d_images2, d_images3, d_kernel, d_sphere_kernel)
            #Copy final arrays back to host side
            images2[:,:,i*chunk:stride] = d_images2.copy_to_host()
            images3[:,:,i*chunk:stride] = d_images3.copy_to_host()
            #Completed Run
    # All parts of this code appear to be working properly        
    return images2, images3
     
    


def batch_these_files(files, folder = os.getcwd(), background = False, gauss_sigma = 2.5, rolling_ball_radius = 6,rolling_ball_height= 6):
    """A Batch routine for performing the rolling ball subtraction on a folder of images"""
    try:
    # Prepare the Folders for Saving the Analysis Data
        os.mkdir(folder + '\\sans background\\')
    except:
        pass
    if background: # Only create a background folder if asked
        try:
            os.mkdir(folder + '\\background\\')
        except:
            pass
        
    for file in files: # Loop over the files provided
        images = load_image_to_array(file_path + file) # Load current file into image array
        if background: # Only hold background if asked
            images_no_background, background_images = rolling_ball_subtraction(images,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
            save_array_as_image(images_no_background, folder + '\\sans background\\' + file)    
            save_array_as_image(images_no_background, folder + '\\background\\' + file)    
        else:
            images_no_background, ___ = rolling_ball_subtraction(images,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
            save_array_as_image(images_no_background, folder + '\\sans background\\' + file)    


    
def cupy_rolling_ball(images, gauss_sigma = 2.5, rolling_ball_radius = 6, rolling_ball_height = 6):
    # Get Image Parameters
    m,n,o = image_size(images)
    pixels = m*n*o #assuming 64 bit images from now on
    
    #Morphological Opening structures
    kernel = cp.asarray(make_gaussian_kernel(gauss_sigma)) #Make normalized gaussian kernel
    # Make a grayscale 'ball'
    sphere_kernel = cp.asarray(make_a_sphere(radius = rolling_ball_radius, height = rolling_ball_height) )
    
    #Make device
    images1 = cp.asarray(images)
    images2 = cp.empty_like(images)
    images3 = cp.empty_like(images) # Preallocate array for images
    for i in range(o):
        cupy.cupyx.ndimage.convolve(image1[:,:,i], kernel, images2[:,:,i])
    return cp.asnumpy(images2)


#%% Working Namespace Section of Code
if __name__ == '__main__':
    """
    file_path = 'D:\\Image Processing Folder\\Background Subtractions for Ryan\\'
    files = os.listdir(file_path)
    batch_these_files(files, folder = file_path)
    
    """
    #Demonstration of a rolling ball subtraction on a stack of images    

    # Select Image to Analyze
    fpath = "D:\\Dropbox\\Andrew & Tim shared folder\\Python Code\\" # Folder
    fname = "cell10_dz20_r2.tif" # File name
    im = load_image_to_array(fpath + fname) # Load image into workspace
    
    print("The Image we loaded has {} rows {} columns and {} frames".format(im.shape[0],im.shape[1],1)) # Print shape of image
    
    #%%    Post loading image example section

    # Easy to use call produces background subtracted images and backgrounds through gpu acceleration
    images_no_background, image_background = rolling_ball_subtraction(im[:,:,0:500], gauss_sigma = 2.5, rolling_ball_radius = 6,rolling_ball_height= 6)
    
    # Data Presentation
    show_as_image(images_no_background[:,:,0])
    plt.title('Background Subtracted Frame')
    plt.figure()
    show_as_image(image_background[:,:,0])
    plt.title('Background Frame')
    #Andrew Nelson 5/4/2020
    images_no_background, image_background = cupy_rolling_ball(im[:,:,0:500],
                                                                     gauss_sigma = 2.5, 
                                                                     rolling_ball_radius = 6,
                                                                     rolling_ball_height= 6)