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
from localization_kernels import *
from numba import cuda
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from localizations_class import *
import open3d as o3d
import pickle
# Defintions for GPU codes to double precision
EXP = 2.71828182845904523536028747135
PI  = 3.14159265358979311599796346854


LOCALIZE_LIMIT = 190*388*400
# GPU Accelerated Code
#
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
            if (ii >= 0) and (ii <m) and (jj>= 0) and (jj <n):
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

# Functions
def count_peaks(peak_image, blanking = 0, seperator = 190, pixel_width = 5):
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
    # Create Apron to ensure no rois near the periphery are selected
    peak_image[:pixel_width+1,:,:] = False
    peak_image[:,:pixel_width+1,:] = False
    peak_image[-pixel_width-1:,:,:] = False
    peak_image[:,-pixel_width-1:,:] = False
    
    centers = np.argwhere(peak_image>0)
    ind = np.argsort(centers[:,2])
    
    mols = centers.shape[0]
    if blanking ==0:
        remover = []
        for i in range(mols): # We'll loop over the list of centers and remove
            if centers[ind[i],1] >= seperator: #Blanking the right frame  
                remover.append(i) # If ID should be abandoned add index to list
        return np.delete(centers[ind,:],remover,0)
        #return centers
    else:
        shift = 0 # Frame shifting variable that allows us to switch behavior between frames
        if blanking == 2: shift = 1 # If we want the first frame to blank on the right, we set shift to 1
        remover = []
        # Remove IDs in blank regions
        for i in range(mols): # We'll loop over the list of centers and remove
            if ((centers[ind[i],2] + shift) % 2 == 0 ) and (centers[ind[i],1] < seperator): #Blanking the left frame  
                remover.append(i) # If ID should be abandoned add index to list
            if ((centers[ind[i],2] + shift) % 2 == 1 ) and (centers[ind[i],1] >= seperator): #Blanking the right frame
                remover.append(i) # If ID should be abandoned add index to list
        return np.delete(centers[ind,:],remover,0) # Return list of centers sans background channel IDs


    
def cpu_localize_images(file_name):
    """
    CPU version of MLE Asym. Gaussian fitting algorithm

    Parameters
    ----------
    file_name : string
        File path/name of image to localize.

    Returns
    -------
    None.

    """
    #Load Images
    images = load_image_to_array(file_name) # Load image into workspace
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
    psf_image_array = segment_image_into_psfs(images_no_background, centers, pixel_width)
    
    # Localization
    # Accept in an array where each row represents one localization
    fits = fit_psf(psf, True)
    
    # Data Cleanup
    
    # Save relevant data

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
    if 2*pixel_width + 1 >= 32:
        blockdim = (32,32) #We can use a linearized array here
        griddim = ((2*pixel_width + 1) // blockdim[0] + 1,  centers.shape[0]) #Specify the block landscape
    else:
        values = np.array([32, 16, 8, 4, 2])
        block_size = values[np.argwhere(values < 2*pixel_width + 1)[0] - 1][0]
        D3_increase = int(1024/(block_size**2)) # block_size and 1024 have common factors so expect integer returns
        blockdim = (block_size,block_size,D3_increase)
        griddim = (1,1,centers.shape[0]//D3_increase + 1)
    gpu_image_segment[griddim, blockdim](np.ascontiguousarray(image),np.ascontiguousarray(psf_image_array), np.ascontiguousarray(centers), np.ascontiguousarray(pixel_width))            
    #GPU Call

def gpu_image_into_psfs(d_image_2, d_psf_array, d_centers, pixel_width):
    """
    Python wrapper for the cuda call to segment out images

    Parameters
    ----------
    d_image_2 : cupy double array pointer
        Background Subtracted data frames.
    d_psf_array : cupy double array pointer
        empty array to be populated by image segments.
    d_centers : cupy int array pointer
        List of [[row,col,frame],] that indicates central pixels for segmentation in d_image_2.
    pixel_width : int
        radius of segmented cut out, final segmentation is (2*pixel_width+1) x (2*pixel_width+1) in size.

    Returns
    -------
    None.

    """
     #GPU Specification
    if 2*pixel_width + 1 >= 32:
        blockdim = (32,32) #We can use a linearized array here
        griddim = ((2*pixel_width + 1) // blockdim[0] + 1,  centers.shape[0]) #Specify the block landscape
    else:
        values = np.array([32, 16, 8, 4, 2])
        block_size = values[np.argwhere(values < 2*pixel_width + 1)[0] - 1][0]        
        D3_increase = int(1024/(block_size**2)) # block_size and 1024 have common factors so expect integer returns
        blockdim = (block_size,block_size,D3_increase)
        griddim = (1,1,d_centers.shape[0]//D3_increase + 1)
    d_pixel_width = cp.array(pixel_width)
    gpu_image_segment[griddim, blockdim](d_image_2,d_psf_array, d_centers, d_pixel_width)
    del d_pixel_width          

   
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

def gpu_find_peaks(d_image_2, d_image_3, threshold, pixel_width):
    '''
    Python Wrapper for GPU based peak finding algorithm

    Parameters
    ----------
    d_image_2 : cupy double array
        Image in which to search for peaks.
    d_image_3 : cupy boolean array
        Binary image to indicate central pixels of ROIs.
    threshold : double
        Pixel threshold limit to qualify an ROI.
    pixel_width : int
        radius of neighborhood to search, final neighborhood is (2*pixel_width+1) x (2*pixel_width+1) in size.

    Returns
    -------
    None.

    '''
    blockdim = (32,32) #Specify number of threads in a 2D grid
    
    m,n,o = image_size(d_image_2)
    griddim = (m// blockdim[0] + 1, n//blockdim[1] + 1,o) #Specify the block landscape
    gpu_peaks[griddim, blockdim](d_image_2, d_image_3,np.ascontiguousarray(threshold), np.ascontiguousarray(pixel_width))



def gauss_and_derivatives(psf, fit_vector, crlbs = False):
    """
    

    Parameters
    ----------
    psf : numpy array
        input image to model.
    fit_vector : numpy array
        Vector containing fit positions.

    Returns
    -------
    corrections : TYPE
        DESCRIPTION.

    """
    pixel_width = int((psf.shape[0]-1)/2)
    x_build = np.linspace(-pixel_width,pixel_width,2*pixel_width +1)
    X , Y = np.meshgrid(x_build, x_build)
    derivatives = [0 , 0 , 0 , 0 , 0 , 0]
    second_derivatives = [0 , 0 , 0 , 0 , 0 , 0]
    x_gauss = 0.5 * (erf( (X - fit_vector[0] + 0.5) / (np.sqrt(2 * fit_vector[3]**2))) - erf( (X - fit_vector[0] - 0.5) / (np.sqrt(2 * fit_vector[3]**2))))
    y_gauss = 0.5 * (erf( (Y - fit_vector[1] + 0.5) / (np.sqrt(2 * fit_vector[4]**2))) - erf( (Y - fit_vector[1] - 0.5) / (np.sqrt(2 * fit_vector[4]**2))))
    psf_guess = fit_vector[2]*x_gauss*y_gauss + fit_vector[5]
    derivatives = [ (fit_vector[2]/(np.sqrt(np.pi*2*fit_vector[3]**2))*
                     (np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2)) -
                      np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2))))*y_gauss, # x-derivative
                   
                    (fit_vector[2]/(np.sqrt(np.pi*2*fit_vector[4]**2))*
                     (np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2)) -
                      np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2))))*x_gauss, # y-derivative
                    
                    x_gauss*y_gauss, # N Derivative
                    
                    fit_vector[2]/(np.sqrt(np.pi*2)*fit_vector[3]**2)*(
                        (X - fit_vector[0] - 0.5) * np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2)) - 
                        (X - fit_vector[0] + 0.5) * np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))
                    *y_gauss, # x-sigma derivative
                    
                    fit_vector[2]/(np.sqrt(np.pi*2)*fit_vector[4]**2)*(
                        (Y - fit_vector[1] - 0.5) * np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2)) - 
                        (Y - fit_vector[1] + 0.5) * np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))
                   * x_gauss, # y-sigma derivative
                   
                    1] # derivative background
    
    second_derivatives = [ fit_vector[2]/np.sqrt(2*np.pi)*fit_vector[3]**3*
                          (y_gauss*((
                              (X - fit_vector[0] - 0.5)* np.exp(-(X-fit_vector[0] - 0.5)**2/(2*fit_vector[3]**2))) - 
                              (X - fit_vector[0] + 0.5)* np.exp(-(X-fit_vector[0] + 0.5)**2/(2*fit_vector[3]**2)))), # x-derivative
                           fit_vector[2]/np.sqrt(2*np.pi)*fit_vector[4]**3*
                           (x_gauss*((
                              (Y - fit_vector[1] - 0.5)* np.exp(-(Y-fit_vector[1] - 0.5)**2/(2*fit_vector[4]**2))) - 
                              (Y - fit_vector[1] + 0.5)* np.exp(-(Y-fit_vector[1] + 0.5)**2/(2*fit_vector[4]**2)))), # y-derivative
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
    if crlbs:
        information_matrix = np.empty((6,6))
        for i in range(6):
            for j in range(6):
                information_matrix[i,j] = np.sum(derivatives[i]*derivatives[j]/psf_guess)
        
        crlbs = np.empty(7)
        if is_invertible(information_matrix):
            information_inverse = np.linalg.inv(information_matrix)
            for i in range(6):
                crlbs[i] = information_inverse[i][i]
            crlbs[6] = np.sum(psf*np.log(psf_guess/(psf+0.00000000001)) - (psf_guess + psf))
        else: 
            crlbs =[-1,-1,-1,-1,-1,-1,1]
        return crlbs
    else:
        return corrections

def fit_psf_array(psf_image_array):
    """
    Wrapper to handle an array of PSF images

    Parameters
    ----------
    psf_image_array : numpy double array
        array of PSF images to be fit w/ MLE approx.

    Returns
    -------
    numpy double array
        Array of fitting parameters.

    """
    fits = []
    for i in range(psf_image_array.shape[2]):#loop over all molecules in array
        fits.append(fit_psf(psf_image_array[:,:,i]))
    return np.array(fits)
        
def fit_psf(psf, figures = False):
    """
    A CPU based maximum likelihood estimation of a gaussian spatial distribution 
    of photons on a pixel grid

    Parameters
    ----------
    psf : numpy 2D image array
        2D image of a point spread function to be fit.
    figures: boolean
        Determines whether or not to show figures of a fit

    Returns
    -------
    fit_vector : array of fits
        Resulting fit with elements [ xf, yf, N, sx, sy, O].

    """
    pixel_width = int((psf.shape[0]-1)/2)
    
    #Build mesh grid for vector calculation
    x_build = np.linspace(-pixel_width,pixel_width,2*pixel_width +1)
    X , Y = np.meshgrid(x_build, x_build)
    fit_vector = [np.sum(np.multiply(psf,X))/np.sum(psf) , np.sum(np.multiply(psf,Y))/np.sum(psf), np.sum(psf), 2, 2, np.min(psf)] # Initial vector guess [xc, yc, N, sx, sy, background]
    cycles = 20
    tracks = np.zeros((cycles,6))
    for i in range(cycles): # Here is the analysis
        corrections = gauss_and_derivatives(psf,fit_vector)
        for j in range(6):
            fit_vector[j] += corrections[j]
        tracks[i,:] = fit_vector
    if figures == True:
        plt.figure()
        plt.plot((tracks[:,0]-tracks[-1,0])/tracks[-1,0], color = 'k', label = 'xf')
        plt.plot((tracks[:,1]-tracks[-1,1])/tracks[-1,1], color = 'g', label = 'yf')
        plt.plot((tracks[:,3]-tracks[-1,3])/tracks[-1,3], color = 'c', label = 'sx')
        plt.plot((tracks[:,4]-tracks[-1,4])/tracks[-1,4], color = 'b', label = 'sy')
        plt.legend()
        plt.title("Settling of fitting variables over fitting iterations")
        plt.ylabel('(F(i) - F(end))/F(end)')
        plt.xlabel('Fitting iteration')
        plt.show()
    
    return fit_vector

def remove_bad_fits(fitting_vector):
    '''
    

    Parameters
    ----------
    fitting_vector : numpy double array
        A lit .

    Returns
    -------
    None.

    '''
    # remove NaNs
    index = []
    for i in range(fitting_vector.shape[0]):
        if ~np.isnan(fitting_vector[i,0]):
            if np.abs(fitting_vector[i,0]) < 5 and np.abs(fitting_vector[i,1]) < 5:
                if np.abs(fitting_vector[i,3]) < 6 and np.abs(fitting_vector[i,4]) < 6:

                    if np.abs(fitting_vector[i,3]) > 0.6 and np.abs(fitting_vector[i,4]) > 0.6:
                        if fitting_vector[i,2] > 10 and fitting_vector[i,2] < 10000000:
                            if fitting_vector[i,5] > -100 and fitting_vector[i,5] < 100:

                                index.append(i)
    return index

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def get_error_values(psf_array, fitting_vector):
    M = psf_array.shape[2]
    crlb_vectors = np.empty((M,7))
    for i in range(M):
        crlb_vectors[i,:] = gauss_and_derivatives(psf_array[:,:,i], fitting_vector[i,:], crlbs = True)
    return crlb_vectors

def localize_image_stack(file_name, pixel_size = 0.130, gauss_sigma = 2.5, rolling_ball_radius = 6, rolling_ball_height = 6, pixel_width = 5, blanking = 2, threshold = 35, start = 0, finish = 0, angs = [0,0]):
    images = load_image_to_array(file_name, start, finish)
    m,n,o = image_size(images)
    pixels = m*n*o
    molecules = Localizations()
    angs = np.array([molecules.red_angle, molecules.orange_angle])
    if pixels <= LOCALIZE_LIMIT:
        # run the localization in one chunk
        fits, crlbs, frames  =localize_image_slices(images, 
                                                  pixel_size,
                                                  gauss_sigma,
                                                  rolling_ball_radius,
                                                  rolling_ball_height,
                                                  pixel_width, 
                                                  blanking,
                                                  threshold,
                                                  molecules.split,
                                                  angs)
        frames += start
    else:
        rounds = pixels // LOCALIZE_LIMIT + 1# Divide into manageable sizes
        chunk = o // (rounds)   # Determine dividing chunk size of stacks
        
        for i in range(rounds): # Loop over number of times needed to chunk through data set
            # Ensure we don't go over our stack size
            stride = np.min((o,(i+1)*chunk)) 
            print(i*chunk*100/o)
            #Parse the image into a subset
            sub_images = images[:,:,i*chunk:stride]
            slice_fits, slice_crlbs, slice_frames  = localize_image_slices(sub_images, 
                                                                           pixel_size,
                                                                           gauss_sigma,
                                                                           rolling_ball_radius,
                                                                           rolling_ball_height,
                                                                           pixel_width,
                                                                           blanking,
                                                                           threshold,
                                                                           molecules.split,
                                                                           angs)
            if i == 0:
                fits = slice_fits
                crlbs = slice_crlbs
                frames = slice_frames
            else:
                fits = np.concatenate((fits,slice_fits))
                crlbs = np.concatenate((crlbs,slice_crlbs))
                frames = np.concatenate((frames,slice_frames+start))


    molecules.store_fits(fits, crlbs, frames)
    return molecules

def wavelet_denoising(images, levels = 3):
    """Images is the stack of images to denoise, levels tells the number of iterations for the denoising"""
    baselet = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
    m,n,o = image_size(images)
    out_images = np.empty_like(images)
    for frame in range(o):
        image = images[:,:,frame]
        waves  = np.empty((m,n,levels))
        for i in range(levels):
            wavelet = baselet
            for j in range(baselet.shape[0] - 1):
                wavelet = np.insert(wavelet,baselet.shape[0] - j - 1,np.zeros(2**i-1))
            blurred_image = np.copy(image) # Initially put the blurred_image defined outside of the levels loop
            for j in range(m):
                blurred_image[j,:] = np.convolve(blurred_image[j,:],wavelet,mode='same')
            
            for j in range(n):
                blurred_image[:,j] = np.convolve(blurred_image[:,j],wavelet,mode='same')
                
            waves[:,:,i] = (image - blurred_image) # we threshold above 2
            image = np.copy(blurred_image) # use blurred image for the next
        out_images[:,:,frame] = waves[:,:,1]*np.where(waves[:,:,1] > 2, 1, 0) # we select the second size scale for psf detection
    return out_images
        
def localize_image_slices(image_slice, pixel_size = 0.130, gauss_sigma = 2.5, rolling_ball_radius = 6, rolling_ball_height = 6, pixel_width = 5, blanking = 2, threshold = 35, split = 0, angs = np.array([0,0])):
    """
    Will use MLE localization algorithm to analyze images such that maximum uptime
    is kept on the gpu

    Parameters
    ----------
    file_name : Str
        Name of image file to localize.

    Returns
    -------
    None.

    """

    m,n,o = image_size(image_slice)    
    kernel = make_gaussian_kernel(gauss_sigma) #Make normalized gaussian kernel
    # Make a grayscale 'ball'
    sphere_kernel = make_a_sphere(radius = rolling_ball_radius, 
                                  height = rolling_ball_height) 

    # Allocate Device Arrays 
    # Note that we will not have host side image arrays
    d_image_2 = cuda.device_array_like(image_slice)
    d_image_3 = cuda.device_array_like(image_slice) # We require 3 image arrays for analysis

    
    # Copy data over 
    d_images = send_to_device(image_slice)
    d_kernel = send_to_device(kernel)
    d_sphere_kernel = send_to_device(sphere_kernel)
    
    #GPU Call for the rolling_ball_subtraction
    gpu_rolling_ball(d_images, d_image_2, d_image_3, d_kernel, d_sphere_kernel)
    
    # background subtracted image is on d_image_2
    # original image is on d_images and background is on d_image_3
    del d_image_3
    d_image_3 = cp.empty((m,n,o),dtype = cp.bool)
    gpu_find_peaks(d_image_2, d_image_3, threshold, pixel_width)
    
    # Copy peak data back onto host
    bkn_subbed = cp.asnumpy(d_image_2)
    peaks = cp.asnumpy(d_image_3)
    '''
    bkn_subbed = cp.asnumpy(d_image_2)
    show_as_image(bkn_subbed[:,:,0])
    peaks[:pixel_width+1,:,:] = False
    peaks[:,:pixel_width+1,:] = False
    peaks[-pixel_width-1:,:,:] = False
    peaks[:,-pixel_width-1:,:] = False
    show_as_image(peaks[:,:,0])'''
    # Clean up GPU memory
    del d_image_3, d_images, d_kernel, d_sphere_kernel
    
    # Start peak finding analysis
    centers = count_peaks(peaks, blanking, seperator = split)
    hot_pixels = centers.shape[0] # number of areas found to localize
    # Set up device memory for next round of computation
    
    d_rotation = cp.array([angs[int(centers[i,1] <= split)] for i in range(hot_pixels)])# determine the 'color' of the molecule based on it's initial location
    #print([angs[centers[i,1] <= split] for i in range(hot_pixels)])
    d_centers = send_to_device(centers)
    d_psfs = cp.empty((2*pixel_width+1, 2*pixel_width +1,centers.shape[0]))
    d_fitting_vectors = cp.empty((centers.shape[0],6))
    
    # Segment Image
    gpu_image_into_psfs(d_image_2, d_psfs, d_centers, pixel_width)
    del d_centers
    del d_image_2
    
    # Perform Fit
    fitting_kernel((1024,),( hot_pixels//1024 + 1,),(d_psfs, d_fitting_vectors, d_rotation, hot_pixels, 20))
    
    # Copy data back onto CPU
    psfs = cp.asnumpy(d_psfs)
    fitted_vectors = cp.asnumpy(d_fitting_vectors)
    
    # Clean up GPU memory
    del d_psfs
    del d_fitting_vectors, d_rotation
    # Adjust for global context
    list_of_good_fits = remove_bad_fits(fitted_vectors)
    fitted_vectors[:,0] += centers[:,1]
    fitted_vectors[:,1] += centers[:,0]
    frames = centers[list_of_good_fits,2]
    keep_vectors = fitted_vectors[list_of_good_fits,:]
    keep_psfs = psfs[:,:,list_of_good_fits]
    # At this point fitting_vectors should contain coordinates of all fitted localizations in the dataset
    
    crlb_vectors = get_error_values(keep_psfs, keep_vectors) # perform a single run through of MLE to compute error values
    
    return keep_vectors, crlb_vectors, frames
    


def localize_folder(folder):
    # get list of image files
    image_files = grab_image_files(fpath)
    
    for file in image_files:
        file_name = fpath + file
        result =  localize_image_stack(file_name, 
                                    pixel_size = 0.130, 
                                    gauss_sigma = 2.5, 
                                    rolling_ball_radius = 6, 
                                    rolling_ball_height = 6, 
                                    pixel_width = 5, 
                                    blanking = 2, 
                                    threshold = 150,
                                    start = 0,
                                    finish = 0)
        save_localizations(result, file_name)
        
#%% Main Workspace
if __name__ == '__main__':
    fpath = "D:\\Dropbox\\Data\\7-17-20 beas actin\\utrophin\\" # Folder
    # get list of image files
    image_files = grab_image_files(fpath)
    
    for file in image_files:
        file_name = fpath + file
        result =  localize_image_stack(file_name, 
                                    pixel_size = 0.130, 
                                    gauss_sigma = 2.5, 
                                    rolling_ball_radius = 6, 
                                    rolling_ball_height = 6, 
                                    pixel_width = 5, 
                                    blanking = 0, 
                                    threshold = 70,
                                    start = 0,
                                    finish = 0)
        save_localizations(result, file_name)
    '''    
    # go one by one through localization analysis and save the files
    fname = "BEA_cell_1_2.tif" # File name
    
    file_name = fpath + fname

    result =  localize_image_stack(file_name, 
                                    pixel_size = 0.130, 
                                    gauss_sigma = 2.5, 
                                    rolling_ball_radius = 6, 
                                    rolling_ball_height = 6, 
                                    pixel_width = 5, 
                                    blanking = 2, 
                                    threshold = 35,
                                    start = 0,
                                    finish = 10)
    result.separate_colors()
    save_localizations(result, file_name)
    xyz = np.empty((result.xf.shape[0],3))
    #orange_xyz = np.empty((result.color.sum(),3))
    point_colors = np.empty((result.xf.shape[0],3))
    for i in range(result.xf_orange.shape[0]):
        if np.abs(result.zf_orange[i]) <= 0.6:
            xyz[i,0] = result.xf_orange[i] 
            xyz[i,1] = result.yf_orange[i]
            xyz[i,2] = result.zf_orange[i]
            point_colors[i,:] = [0, 0, 1]
        else:
            xyz[i,0] = 0 
            xyz[i,1] = 0
            xyz[i,2] = 0
            point_colors[i,:] = [0, 0, 1]
    
    for i in range(result.xf_red.shape[0]):
        ii = i + result.xf_orange.shape[0]
        if np.abs(result.zf_red[i]) <= 0.6:
            xyz[ii,0] = result.xf_red[i] 
            xyz[ii,1] = result.yf_red[i]
            xyz[ii,2] = result.zf_red[i]
            point_colors[ii,:] = [1, 0, 0]        
        else:
            xyz[ii,0] = 0 
            xyz[ii,1] = 0
            xyz[ii,2] = 0
            point_colors[ii,:] = [0, 0, 1]
    
    index_of_numbers = np.isfinite(xyz[:,0])
    selected_point_colors = point_colors[index_of_numbers,:]
    selected_xyz = xyz[index_of_numbers,:]
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(selected_xyz)
    pcd.colors = o3d.utility.Vector3dVector(selected_point_colors)
    o3d.visualization.draw_geometries([pcd])
    '''
    
    
    '''
    psfs, truths = simulate_psf_array(1000)
    fits =  fit_psf_array(psfs)
    centers = np.zeros((truths.shape[0],3))
    #%%
    list_of_good_fits = remove_bad_fits(fits,centers, truths)
    results = truths[list_of_good_fits,:] - fits[list_of_good_fits,:]
    results[:,2] /= truths[list_of_good_fits,2]
    
    errors = get_error_values(psfs, fits)
    '''
                                        
    #%% The code below is an example of localizing the data
    '''
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
    
    fits = fit_psf(psf, True)
    print("For The Generated Image, it was found with position ({},{}) , with {} photons. ".format(np.round(fits[0]+6,3),np.round(6+fits[1],3),np.round(fits[2],3)))
    print("The width was {} pixels in 'x', and {} pixels in 'y', background = {}. ".format(np.round(fits[3],3),np.round(fits[4],3),np.round(fits[5],3)))
    show_as_image(psf)    
    # Save relevant data
    #fits = fit_psf_array(psf_image_array)
    #%% GPU Fitting Section
    fit_gpu = fit_psf_array_on_gpu(psf_image_array, 0 , 20)
    '''
    