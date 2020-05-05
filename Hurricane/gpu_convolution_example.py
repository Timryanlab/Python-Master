# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:06:31 2020
A Cuda Library for Image Analysis
Andrew Nelson
@author: Andrew Nelson
"""
"""Import Libraries"""
from numba import cuda
from numba import vectorize, float64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
        s_m = 10^9

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
    if (image1[i,j,k] - image2[i,j,k]) >0:
        image3[i,j,k] = image1[i,j,k] - image2[i,j,k]
    else:
        image3[i,j,k] = 0
    #image3[i,j,k] = j
    
#%%
"""Non GPU Code Below this Line"""    
def load_image_to_array(fname):
    """This function will take in a string as a file location and file name
       It will return a Numpy Array of the image at this location"""
    im = Image.open(fname) #Initial call to image datafile
    #Get shape of image
    m,n = np.shape(im)
    o = im.n_frames        
    # Allocate variable memory for image
    image_array = np.zeros((m,n,o))
    for i in range(o):
        im.seek(i) #Seek to frame i in image stack
        image_array[:,:,i]= np.array(im) #Populate data into preallocated variable
    return image_array # return numpy image array

def show_as_image(image_frame):
    """We expect a 2D numpy array and will output the visual grayscale result"
    """
    plt.imshow(image_frame)
    

def make_gaussian_kernel(sigma = 2, kernel_radius = 5): # write a function to create a gaussian kernel whose width is 2sigma + 1
    """
    Returns a volume normalized 2D Gaussian of sigma = sigma and
    image size equal to  2*ceil(sigma)+ 1 pixels x 2*ceil(sigma)+ 1 pixels
    """
    xline = np.linspace(-kernel_radius,kernel_radius, 2*kernel_radius + 1) # Using sigma max ensures encorporating the 2 sigma radius of the gaussian
    yline = np.exp(-xline*xline/(2*sigma*sigma))
    gauss =np.outer(yline,yline)
    return gauss / gauss.sum()
    #return np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])

def run_the_conv(image1, kernel, image2, im_type = 0):
    blockdim = (32,32)
    griddim = (image1.shape[0]// blockdim[0] + 1, image1.shape[1]//blockdim[1] + 1,image1.shape[2])
    gpu_convolve[griddim, blockdim](np.ascontiguousarray(image1), np.ascontiguousarray(kernel), np.ascontiguousarray(image2), im_type)
    
def run_the_subtraction(image1, image2, image3):
    blockdim = (32,32)
    griddim = (image1.shape[0]// blockdim[0] + 1, image1.shape[1]//blockdim[1] + 1,image1.shape[2])
    gpu_image_subtraction[griddim, blockdim](np.ascontiguousarray(image1),np.ascontiguousarray(image2),np.ascontiguousarray(image3))
    
def blur_these_frames(images, radius = 1):
    """
    This function will take a stack of images and blur this with a guassian filter of sigma radius = radius
    It will return a stack of images indexed identically to the input stack
    """
    plt.close('all')
    kernel = make_gaussian_kernel(radius) #Make normalized gaussian kernel
    images_out = np.empty_like(images) # Preallocate array for images
    images_2 = np.empty_like(images)
    #print(kernel)
    
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
    kernel = make_a_sphere(radius = 6)
    #Erosion
    run_the_conv(images_out,kernel,images_2,1)
    show_as_image(images_2[:,:,0])
    plt.title("Eroded")
    plt.figure()
    #print(kernel)

    run_the_conv(images_2,kernel,images_out,2)
    # The entire convolution algorithm appears to be working as intended, minor
    # differences in rball will result in differences from matlab calculation
    
    plt.figure()
    show_as_image(images_out[:,:,0])
    plt.title("Dilated (Background)")
    print("This is the input image")
    print(images_2[0:10,0:10,0])
    run_the_subtraction(images,images_out,images_2)
    show_as_image(images_2[:,:,0])
    plt.title("Bkgn Subtracted")
    print("This is the input image")
    print(images_2[0:10,0:10,0])
    
    return images_2

def make_a_sphere(radius = 6):
    """Creates a logical array for use in morphological openings and closings"""
    kernel = np.zeros((2*radius +1, 2*radius +1), dtype=float) - np.inf
    for i in range(kernel.shape[0]):
        for k in range(kernel.shape[0]):
            if((i-(2*radius+1)/2+0.5)**2 + (k - (2*radius+1)/2+0.5)**2) <= (radius)**2:
                kernel[i,k] = np.sqrt((radius)**2 - ((i-(2*radius+1)/2+0.5)**2 + (k - (2*radius+1)/2+0.5)**2))/2 + radius/2
    return kernel
    
def gpu_rolling_ball_subtraction(d_image, d_image2, d_image3, d_gauss, d_ball):
    """"We are expecting input of device arrays for:
        d_image The original image
        d_image2 Allocated space for background subtracted image
        d_image3 preallocated space for the background approximation calculated
        
        This function will utilize the GPU algorithmic process
    """
    #Perform a Gaussian Blur
    run_the_conv(d_image, d_gauss, d_image2, 0)
    #Perform an image erosion
    run_the_conv(d_image2, d_ball, d_image3, 1)
    #Perform an image dilation
    run_the_conv(d_image3, d_ball, d_image2, 2)
    subtract_the_images(d_image,d_image2,d_image3)
    
def rolling_ball_subtraction(image, gauss_radius = 1.1, sphere_radius = 6) :
    """We take in an image stack on the host machine
        Take into account memory considerations and return a background
        subtracted image with the potential for returning the background as well
        Inputs : 
            image: Image to correct
            gauss_radius = blurring radius for gaussian convolution
            sphere_radius = radius for sphere neighborhood
        Returns:
            image2 = Background 
            image3 = Background corrected image
    """
    #image size
    m,n,o = image.shape
    image_memory = m*n*o*64 #assuming 64 bit images from now on
    cuda.get_current_devicet().reset()
    
    
    #Build Shaping Elements and transfer to device
    gauss_kernel = make_gaussian_kernel(gauss_radius)
    ball = make_a_sphere(sphere_radius)
    d_gauss = cuda.to_device(np.ascontiguousarray(gauss_kernel))
    d_ball = cuda.to_device(np.ascontiguousarray(ball))
    device_memory = cuda.get_current_device().primary_context.get_memory_info()
    if image_memory*3.01 < device_memory: # this process will use 3 image arrays to complete if memory exists to hold all at once allocate and go
        # Preallocate device arrays for fastest analysis
        d_image = cuda.to_device(np.ascontiguousarray(image))
        d_image2 = cuda.device_array((image.shape[0],image.shape[1],image.shape[2]),dytpe = image.dtype)
        d_image3 = cuda.device_array((image.shape[0],image.shape[1],image.shape[2]),dytpe = image.dtype)
        gpu_rolling_ball_subtraction(d_image, d_image2, d_image3, d_gauss, d_ball)
        return d_image3.copy_to_host(), d_image2.copy_to_host()
    else:
        image2 = np.empty_like(image)
        image3 = np.empty_like(image)
        rounds = image_memory // (3.01 * device_memory) + 1 # Ensure a little buffer on memory
        chunk = o // rounds
        for i in range(rounds):
            # Preallocate device arrays for fastest analysis
            jump = np.min((o,(i+1)*chunk)) # check to see which number is smaller to ensure not 'jumping' into pixels we don't have
            image_sub = image[:,:,i*chunk:jump]
            d_image = cuda.to_device(np.ascontiguousarray(image_sub))
            d_image2 = cuda.device_array((image_sub.shape[0],image_sub.shape[1],image_sub.shape[2]),dytpe = image_sub.dtype)
            d_image3 = cuda.device_array((image_sub.shape[0],image_sub.shape[1],image_sub.shape[2]),dytpe = image_sub.dtype)
            gpu_rolling_ball_subtraction(d_image, d_image2, d_image3, d_gauss, d_ball)
            image2[:,:,i*chunk:jump] = d_image2.copy_to_host()
            image3[:,:,i*chunk:jump] = d_image3.copy_to_host()
        return image3, image2
        
#%%        
# Example Of Blurring Super Resolution Data
# Select Image to Analyze
fpath = "D:\\Dropbox\\Data\\3-3-20 glut4-vglutmeos\\" # Folder
fname = "cell10_dz20_r2.tif" # File name

im = load_image_to_array(fpath + fname) # Load image into workspace
print("The Image we loaded has {} rows {} columns and {} frames".format(im.shape[0],im.shape[1],im.shape[2])) # Print shape
#%%
#Demonstration of gaussian blur
blurred_images = blur_these_frames(im[:,:,11:12], 1.2)

#Subsection of total image for memory analysis
image1 = im[:,:,11]
#%%
show_as_image(image2)
plt.figure()
show_as_image(background)



    