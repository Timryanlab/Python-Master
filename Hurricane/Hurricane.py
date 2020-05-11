# -*- coding: utf-8 -*-
"""
Title: Hurricane

Concept: This is the localization Pipeline the analyzes tiff / fits files and 
outputs localization data for subsequent analysis
Created on Tue May  5 10:21:58 2020

@author: Andrew Nelson
"""
#%% Import Libraries
from ryan_image_io import *
from rolling_ball_subtraction import *
from numba import cuda
from numba import vectorize, float64
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import os

#%% GPU Accelerated Code
#%%
"""Begin GPU Code"""
@cuda.jit()
def gpu_peaks(image_in, image_out, thresh, pixel_width):
    """
    image_in.shape = image_out.shape 
        
    This code will look in the neighborhood of radius pixw to determine if it is a local maxima
    
    To do this the code will search a neighborhood in a raster scan while keep track of the 
    maxima value it finds above the threshold provided. Once found the ith and jth coordinate are noted
    if they correspond to the cuda i,j then the resulting image_out[i,j,k] = 1 and all rest = 0
    
    This is being followed from https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
    """

    i,j,k = cuda.grid(3) # Get image coordinates
    pixw = pixel_width[0]
    m, n, o = image_in.shape # get row and column
    if (i >= m) or ( j >= n) or (k >=o): # If Any of the threads aren't working on a part of an image kick it out
        return
    

    max_val = thresh[0]
    # Prime values relative to the cuda position
    max_i = i-pixw
    max_j = j-pixw
    
    
    # Loop over the neighborhood in a double loop
    for ii in range(i-pixw,i+pixw):
        for jj in range(j-pixw,j+pixw):
            #Ensure that you are only about to look at an image pixel
            if (ii >= 0) and (ii <m) and (jj>=0) and (jj <n):
                #Basic maximum pixel search, record index locations
                if image_in[ii,jj,k] >= max_val:
                    max_val = image_in[ii,jj,k]
                    max_i = ii
                    max_j = jj
    # Check to see if maximum
    if (max_i == i) and (max_j == j):
        image_out[i,j,k] = 1 #If local maxima turned out to be ideal
    else:
        image_out[i,j,k] = 0 #if not

@cuda.jit()
def gpu_image_blank(image_in, seperator, frame_shift):
    """
    A gpu based algorithm to set non-signal Super-res channels to 0 based on their
    alternating frame pattern. 
    """

    i,j,k = cuda.grid(3) # Get image coordinates
    pixw = pixel_width[0]
    m, n, o = image_in.shape # get row and column
    blank_side = (k + frame_shift[0])//2 # k is the current frame_shift = 0 by default, setting value to 1 switches behavior
    if (i >= m) or ( j >= n) or (k >=o): # If Any of the threads aren't working on a part of an image kick it out
        return
    if (blank_side == 0) and (j < seperator[0]): # If you've been marked as a left blank
        image_in[i,j,k] = 0                   #  frame and are a left side pixel set to 0
    if (blank_side == 1) and (j >= seperator[0]): # If you've been marked as a right blank
        image_in[i,j,k] = 0                   #  frame and are a right side pixel set to 0            

@cuda.jit()
def gpu_image_segment(images, psf_image_array, centers, pixel_width):
    """
    

    Parameters
    ----------
    images : numpy array [m,n,o]
        Numpy Image array to be segmented.
    psf_image_array : numpy array shape [N,(2*pixel_width + 1)^2]
        Final image array for localization.
    centers : numpy array [N,3]
        Array of N 3D locations of peaks to be segmented.
    pixel_width : int
        Radial width of segmentation. 

    Returns
    -------
    None.

    """
    i,j,k = cuda.grid(3) # Get location in data array x, y span  [0 , 1, 2 ....]
    if (i >= pixel_width[0]*2 + 1) and (j >= pixel_width[0]*2 + 1) and (k >= centers.shape[0]):
        return
    else:
        ii = i - pixel_width[0]
        jj = j - pixel_width[0]
        if (i<psf_image_array.shape[0]) and (j<psf_image_array.shape[1]) and (k<psf_image_array.shape[2]):
            psf_image_array[i,j,k] = images[centers[k,0] + ii, centers[k,1] + jj, centers[k,2]]
            #psf_image_array[i,j,k] = k
        
    

#%% Functions
def count_peaks(peak_image, blanking = 0, seperator = 190):
    """
    

    Parameters
    ----------
    peak_image : (m,n,o) binary image stack
        High pixels mark molecules to be counted and ID'd.
    blanking : int, optional
        Toggling variable to return different behavior. 
            The default is 0. 
            0 = No blanking
            1 = left channel blanks first 
            2 = right channel blanks first
    seperator : int, optional
        DESCRIPTION. The default is 190.

    Returns
    -------
    List([[float, float,int],]) = list([[row index, col index, frame index]])
    Returns a list representing the high pixel of likely molecular emissions

    """
    centers = np.argwhere(peak_image>0)
    mols = centers.shape[0]
    if blanking ==0:
        return centers
    else:
        shift = 0 # Frame shifting variable that allows us to switch behavior between frames
        if blanking == 2: shift = 1 # If we want the first frame to blank on the right, we set shift to 1
        remover = []
        # Remove IDs in blank regions
        for i in range(mols): # We'll loop over the list of centers and remove
            if ((centers[i,2] + shift) % 2 == 0 ) and (centers[i,1] < seperator): #Blanking the left frame  
                remover.append(i) # If ID should be abandoned add index to list
            if ((centers[i,2] + shift) % 2 == 1 ) and (centers[i,1] >= seperator): #Blanking the right frame
                remover.append(i) # If ID should be abandoned add index to list
        return np.delete(centers,remover,0) # Return list of centers sans background channel IDs

def localize_file(file):
    """Localize file will do memory batch processing to prevent gpu overload
        this allows localize_images to focus on optimizing dataflow and prioritize 
        analysis speed
    """
    images = load_image_to_array(fpath + fname) # Load image into workspace
    localize_images(images)
    
def localize_images(images):
    
    #Load Images
    print('Image detected with m = {}, n = {}, and o = {}'.format(images.shape[0],images.shape[1],images.shape[2]))
    
    # Data Prep
    # Subtract Background using rolling ball subtraction
    print('Subtracting Background')
    gauss_sigma = 2.5
    rolling_ball_radius = 6
    rolling_ball_height = 6
    images_no_background, background_images = rolling_ball_subtraction(images,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
    del images
    del background_images
    print('Background Subtracted')
    
    # Peak Identification
    # Accepts background subtracted image and finds the likely spots for molecules  
    # returns a binary image representing areas of likely molecular emission
    wavelet_images = make_wavelets(images_no_background)
    binary_peak_image = find_peaks(wavelet_images, 30, 3)
    
    # Prepare Molecular Data Arrays
    # Accepts binary_peak_image blanking and channel separation
    # returns centers = nx3 array containing the central pixel of n molecular emissions
    centers = count_peaks(image2, 2)
   
    # Image Segmentation
    # Accepts: images_no_background, centers, pixel_width
    # Returns: molecules = the hurricane based data structure
    
    
    # Localization
    # Accept in an array where each row represents one localization
    
    # Data Cleanup
    
    # Save relevant data

def wavelet_denoise(image):
    """Non-optimized """
    pass

def segment_image_into_psfs(image, centers, pixel_width):
    """
    CPU side wrapper for a GPU segmentation of an image stack

    Parameters
    ----------
    images : numpy array [m,n,o]
        Numpy Image array to be segmented.
    centers : numpy array [N,3]
        Array of N 3D locations of peaks to be segmented.
    pixel_width : int
        Radial width of segmentation. 

    Returns
    -------
    psf_image_array : numpy array shape [N,(2*pixel_width + 1)^2]
        Final image array for localization.

    """
    # Variable preallocation
    psf_image_array = np.zeros((2*pixel_width + 1,2*pixel_width + 1,centers.shape[0]))
    
   #GPU Specification
    if 2*pixel_width + 1 > 32:
        blockdim = (32,32) #We can use a linearized array here
        griddim = ((2*pixel_width + 1) // blockdim[0] + 1,  centers.shape[0]) #Specify the block landscape
    else:
        values = np.array([32, 16, 8, 4, 2])
        block_size = values[np.argwhere(values < 2*pixel_width + 1)[0] - 1][0]
        if block_size != 32:
            D3_increase = int(1024/(block_size**2)) # block_size and 1024 have common factors so expect integer returns
            blockdim = (block_size,block_size,D3_increase)
            griddim = (1,1,centers.shape[0]//D3_increase + 1)
            gpu_image_segment[griddim, blockdim](np.ascontiguousarray(image),np.ascontiguousarray(psf_image_array), np.ascontiguousarray(centers), np.ascontiguousarray(pixel_width))
        else:
            blockdim = (32,32)
            griddim = (1,1,centers.shape[0])
            gpu_image_segment[griddim, blockdim](np.ascontiguousarray(image),np.ascontiguousarray(psf_image_array), np.ascontiguousarray(centers), np.ascontiguousarray(pixel_width))            
    #GPU Call
    
    
    return psf_image_array
   
def find_peaks(image, threshold = -1, pixel_width = 5):
    """ Non-optimized gpu peak finding wrapper"""
    if threshold == -1:
        threshold = np.mean(image)
    blockdim = (32,32) #Specify number of threads in a 2D grid
    image2 = np.empty_like(image)
    m,n,o = image_size(image)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    #image1 = wavelet_denoise(image)    
    gpu_peaks[griddim, blockdim](np.ascontiguousarray(image), np.ascontiguousarray(image2), np.ascontiguousarray(threshold), np.ascontiguousarray(pixel_width))
    return image2

def gauss_and_derivatives(psf, fit_vector):
    pixel_width = int((psf.shape[0]-1)/2)
    x_build = np.linspace(-pixel_width,pixel_width,2*pixel_width +1)
    X , Y = np.meshgrid(x_build, x_build)
    derivatives = [0 , 0 , 0 , 0 , 0 , 0]
    second_derivatives = [0 , 0 , 0 , 0 , 0 , 0]
    x_gauss = 0.5 * (erf( (X - fit_vector[0] + 0.5) / (np.sqrt(2 * fit_vector[3]**2))) - erf( (X - fit_vector[0] - 0.5) / (np.sqrt(2 * fit_vector[3]**2))))
    y_gauss = 0.5 * (erf( (Y - fit_vector[1] + 0.5) / (np.sqrt(2 * fit_vector[4]**2))) - erf( (Y - fit_vector[1] - 0.5) / (np.sqrt(2 * fit_vector[4]**2))))
    psf_guess = fit_vector[2]*x_gauss*y_gauss + fit_vector[5]
    derivatives = [ np.multiply(fit_vector[2]/(np.sqrt(np.pi*2*fit_vector[3]**2))*(np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2)) -np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2))),y_gauss), # x-derivative
                    np.multiply(fit_vector[2]/(np.sqrt(np.pi*2*fit_vector[4]**2))*(np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2)) -np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2))),x_gauss), # y-derivative
                    np.multiply(x_gauss,y_gauss), # N Derivative
                    fit_vector[2]/(np.sqrt(np.pi*2)*fit_vector[3]**2)*np.multiply((np.multiply((X - fit_vector[0] - 0.5) , np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2))) - np.multiply((X - fit_vector[0] + 0.5) , np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))),y_gauss), # x-sigma derivative
                    fit_vector[2]/(np.sqrt(np.pi*2)*fit_vector[4]**2)*np.multiply((np.multiply((Y - fit_vector[1] - 0.5) , np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2))) - np.multiply((Y - fit_vector[1] + 0.5) , np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))),x_gauss), # y-sigma derivative
                    1] # derivative background
    second_derivatives = [ fit_vector[2]/np.sqrt(2*np.pi)*fit_vector[3]**3*np.multiply(y_gauss, np.multiply((X - fit_vector[0] - 0.5), np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2))) - np.multiply((X - fit_vector[0] + 0.5), np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))), # x-derivative
                           fit_vector[2]/np.sqrt(2*np.pi)*fit_vector[4]**3*np.multiply(x_gauss, np.multiply((Y - fit_vector[1] - 0.5), np.exp(-(X-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2))) - np.multiply((Y - fit_vector[1] + 0.5), np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))), # y-derivative
                           0, # N 2nd Derivative
                           fit_vector[2]/(np.sqrt(2*np.pi))*
                           y_gauss*((((X - fit_vector[0] - 0.5)**3)*np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2))-
                                     ((X - fit_vector[0] + 0.5)**3)*np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))/(fit_vector[3]**5) - 
                                    2/(fit_vector[3]**3)*(((X - fit_vector[0] - 0.5)**1)*np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2))-
                                                            ((X - fit_vector[0] + 0.5)**1)*np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))), # x-sigma 2nd derivative
                           fit_vector[2]/(np.sqrt(2*np.pi))*
                           x_gauss*((((Y - fit_vector[1] - 0.5)**3)*np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2))-
                                     ((Y - fit_vector[1] + 0.5)**3)*np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))/(fit_vector[4]**5) - 
                                    2/(fit_vector[4]**3)*(((Y - fit_vector[1] - 0.5)**1)*np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2))-
                                                            ((Y - fit_vector[1] + 0.5)**1)*np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))), # y-sigma 2nd derivative
                           0] # 2nd derivative background
    dx = np.sum(derivatives[0]*(psf/psf_guess -1))
    dy = np.sum(derivatives[1]*(psf/psf_guess -1))
    dn = np.sum(derivatives[2]*(psf/psf_guess -1))
    dsx = np.sum(derivatives[3]*(psf/psf_guess -1))
    dsy = np.sum(derivatives[4]*(psf/psf_guess -1))
    db = np.sum(derivatives[5]*(psf/psf_guess -1))
    
    ddx = np.sum(second_derivatives[0]*(psf/psf_guess - 1) - derivatives[0]**2*psf/psf_guess**2)
    ddy = np.sum(second_derivatives[1]*(psf/psf_guess - 1) - derivatives[1]**2*psf/psf_guess**2)
    ddn = np.sum(-derivatives[2]**2*psf/psf_guess**2)
    ddsx = np.sum(second_derivatives[3]*(psf/psf_guess - 1) - derivatives[3]**2*psf/psf_guess**2)
    ddsy = np.sum(second_derivatives[4]*(psf/psf_guess - 1) - derivatives[4]**2*psf/psf_guess**2)
    ddb = np.sum(-psf/psf_guess**2)
    
    corrections = [- dx/ddx,
                   - dy/ddy,
                   - dn/ddn,
                   - dsx/ddsx,
                   - dsy/ddsy,
                   - db/ddb]
    
    return corrections
                         
def fit_psf(psf):
    """
    

    Parameters
    ----------
    psf : numpy 2D image array
        2D image of a point spread function to be fit.

    Returns
    -------
    fit_vector : array of fits
        Resulting fit with elements [ xf, yf, N, sx, sy, O].

    """
    pixel_width = int((psf.shape[0]-1)/2)
    
    #Build mesh grid for vector calculation
    x_build = np.linspace(-pixel_width,pixel_width,2*pixel_width +1)
    X , Y = np.meshgrid(x_build, x_build)
    fit_vector = [np.sum(np.multiply(psf,X))/np.sum(psf) , np.sum(np.multiply(psf,Y))/np.sum(psf), np.sum(psf), 1.2, 1.2, np.min(psf)] # Initial vector guess [xc, yc, N, sx, sy, background]
    cycles = 20
    tracks = np.zeros((cycles,6))
    for i in range(cycles): # Here is the analysis
        corrections = gauss_and_derivatives(psf,fit_vector)
        for j in range(6):
            fit_vector[j] += corrections[j]
        tracks[i,:] = fit_vector
    plt.figure()
    plt.plot(tracks[:,0], color = 'k')
    plt.plot(tracks[:,1], color = 'g')
    plt.plot(tracks[:,3], color = 'c')
    plt.plot(tracks[:,4], color = 'b')
    plt.title("Settling of fitting variables over fitting iterations")
    plt.ylabel('Fitting variable in various units')
    plt.xlabel('Fitting iteration')
    plt.show()
    

    
    return fit_vector
    
    
#%% Localization Class    
#%% Main Workspace
if __name__ == '__main__':
    fpath = "D:\\Dropbox\\Data\\3-3-20 glut4-vglutmeos\\" # Folder
    fname = "cell10_dz20_r2.tif" # File name
    #%% Load Image
    im = load_image_to_array(fpath + fname, 20) # Load image into workspace
    print(im.shape) # Print shape
    
    #%% Subtract Background using rolling ball subtraction
    print('Subtracting Background')
    gauss_sigma = 2.5
    rolling_ball_radius = 6
    rolling_ball_height = 6
    pixel_width = 5
    blanking = 2
    threshold = 35
    images_no_background, background_images = rolling_ball_subtraction(im,  gauss_sigma, rolling_ball_radius, rolling_ball_height)
    print('Background Subtracted')
    del im # No reason to keep too much memory
    del background_images
    # Wavelet denoising for peak indentification
    # Saving this for later
    
    #%% Peak Identification
    #GPU Peak Identification
    
    show_as_image(images_no_background[:,:,0])
    plt.title('Background Subtracted Frame')
    plt.figure()
    image2 = find_peaks(images_no_background, threshold, pixel_width)
    show_as_image(image2[:,:,0])
    plt.title('Peaks')
    # Because the Peak ID is parallelized it's just easier to keep it as part
    # Of the main pipeline despite it's false positives. We can reject those easily
    # On the other side of this by noting what frame we're on
    
    # Peak counting
    centers = count_peaks(image2, blanking)
    del image2
    # Image Segmentation
    # This can be performed on a GPU and the output can be fed into the localization algorithm
    psf_image_array = segment_image_into_psfs(images_no_background, centers, pixel_width)

    # Localization
    
    psf = psf_image_array[:,:,2]
    
    fits = fit_psf(psf)
    print("For The Generated Image, it was found with position ({},{}) , with {} photons. ".format(np.round(fits[0]+6,3),np.round(6+fits[1],3),np.round(fits[2],3)))
    print("The width was {} pixels in 'x', and {} pixels in 'y', background = {}. ".format(np.round(fits[3],3),np.round(fits[4],3),np.round(fits[5],3)))
    show_as_image(psf)    
    # Save relevant data