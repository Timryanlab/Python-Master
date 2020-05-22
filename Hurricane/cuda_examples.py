# -*- coding: utf-8 -*-
"""
Title: Cuda Localization

Concept:
Created on Thu May 21 13:34:41 2020

@author: Andrew Nelson
"""
#%% Import Libraries
from ryan_image_io import *
from Hurricane import *
import cupy as cp
#%% Functions

'''
Notes on this function
Indexing is weird because of how I index in python. Lesson learned but sticking with my convention for now. 
frame = i
column = frame + number_of_frames*column
row = frame + number_of_frames*column + row*number_of_frames*number_of_columns
'''


fitting_kernel = cp.RawKernel(r'''
extern "C" 
 __global__ void fitting_kernel(const double* psf_array, double* fit_array, double ang, int images, int cycles) {
    const int pix = 11; // pixel size
	__shared__ double xgrid[pix*pix];			// allocate xpix and ypix variables to the shared memory of the blocks
	__shared__ double ygrid[pix*pix];			// this will reduce calls to global device memory if we didn't need rotation we could reduce the variables needed for calculation
    int tx = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (tx == 0){ // create xgrid and ygrid we want to create the grid regardless of whether the index is crunching on an image
		for (int i = 0; i <pix; i++){
			for(int j = 0; j <pix; j++){
				double x = (double)j - ((double)pix-1)/2;
				double y = (double)i - ((double)pix-1)/2;
				xgrid[j*pix + i] = x*cos(ang) - y*sin(ang);
				ygrid[j*pix + i] = x*sin(ang) + y*cos(ang);
			}
		}
	}
    
    if( index < images){ // ensure that the thread is working on a molecule
        double fits[6] = {0, 0, 0, 2.0, 2.0, 2000}; // preallocate fitting array
        double psf[pix*pix];
        double pi = 3.14159265358979323846;
        // populate fitting array with initial guesses
        for(int ii = 0; ii <pix*pix; ii++){
            int i = ii % pix;
            int j = ii / pix;
            psf[ii] = psf_array[index + (j + i*pix)*images]; // this is a critical step in the process. Here the index comes in bizzare, but needs to be arranged such that it is a column major process
            fits[0] += xgrid[ii]*psf[ii]; // Estimate center of mass
            fits[1] += ygrid[ii]*psf[ii]; // Estimate center of mass
            fits[2] += psf[ii];           // Estimate initial photon emission load
            if( fits[5] > psf[ii]){fits[5] = psf[ii];} // estimate offset
        }
        fits[0] = fits[0]/fits[2]; // Get center of mass by dividing through the sum        
        fits[1] = fits[1]/fits[2]; // Center of mass y component
        
        // At this point, we have loaded our PSF into our thread and populated our fit vectors with proper initial guess
        // This has been confirmed through reading out the following line of code and comparing with CPU versions of the process        
        // for(int i= 0; i < 6; i++){fit_array[i+ index*6] = fits[i];}   AJN 5/21/20 everything written in this kernel after this line is untested
        
        // Begin the fitting loops
        for(int cycle = 0; cycle < cycles; cycle ++){
            
            double psf_pixel, Ex, Ey, xp, xm, yp, ym;
            double dx, dy, dn, dsx, dsy, db;
            double ddx, ddy, ddn, ddsx, ddsy, ddb;
            // Calculate pixel values for derivatives, 2nd derivatives, errorfunctions and u
			for (int row = 0; row < pix; row++){	// FOR 2  loops over all rows
				for (int col = 0; col < pix; col++){	// FOR 3 loops over all columns
                    // Initialize calculation arrays
                    double derivative[6] = {0, 0, 0, 0, 0, 0};
                    double derivative_2[6] = {0, 0, 0, 0, 0, 0};
                    
                    xp = (xgrid[row + col*pix] - fits[0] + 0.5);
                    xpg = exp(-powf(xp,2)/(2*powf(fits[3],2)));
                    xm = (xgrid[row + col*pix] - fits[0] - 0.5); // these terms are very regularly used in the subsequent calculations
                    xmg = exp(-powf(xm,2)/(2*powf(fits[3],2)));
                    
                    yp = (ygrid[row + col*pix] - fits[1] + 0.5);
                    ypg = exp(-powf(yp,2)/(2*powf(fits[4],2)));
                    ym = (ygrid[row + col*pix] - fits[1] - 0.5);
                    ymg = exp(-powf(ym,2)/(2*powf(fits[4],2)));
                    
                    // Define the estimated gaussian
                    Ex = 0.5 * (erf(xp / sqrt(2.0 * fits[3] * fits[3])) - erf(xm / sqrt(2.0 * fits[3] * fits[3]))); // x component gaussian
					Ey = 0.5 * (erf(yp / sqrt(2.0 * fits[4] * fits[4])) - erf(ym / sqrt(2.0 * fits[4] * fits[4]))); // Y component gaussian
					u = fits[2]*Ex*Ey + fits[5]; // pixel of estimated PSF
                    
                    // Calculate pixel contributions to derivative
                    derivative[0] = Ey*fits[2]/(sqrt(pi*2)*fits[3])*(xmg - xpg);
                    derivative[1] = Ex*fits[2]/(sqrt(pi*2)*fits[4])*(ymg - ypg);
                    derivative[2] = Ex*Ey;
                    derivative[3] = Ey*fits[2]/(sqrt(pi)*powf(fits[3],2))*( xm* xmg - xp* xpg);
                    derivative[4] = Ex*fits[2]/(sqrt(pi)*powf(fits[4],2))*( ym* ymg - yp* ypg);
                    derivative[5] = 1;
                    
                    derivative_2[0] = Ey*fits[2]/sqrt(2*pi)*powf(fits[3],3))*(xm*xmg - xp*xpg);
                    derivative_2[1] = Ex*fits[2]/sqrt(2*pi)*powf(fits[4],3))*(ym*ymg - yp*ypg);
                    derivative_2[2] = 0;
                    derivative_2[3] = Ey*fits[2]/sqrt(2*pi)*((powf(xm,3)*xmg - powf(xp, 3)*xpg)/powf(fits[3],5) - 2*(xm*xmg - xp*xpg)/powf(fits[3],3));
                    derivative_2[4] = Ex*fits[2]/sqrt(2*pi)*((powf(ym,3)*ymg - powf(yp, 3)*ypg)/powf(fits[4],5) - 2*(ym*ymg - yp*ypg)/powf(fits[4],3));
                    derivative_2[5] = 0;
                    
                    // I'm reasonably confident that these work'
                
                }
            }
        }
    }
}
''', 'fitting_kernel')

def gpu_psf_fit(images):
    m,n,o = image_size(images)
    fits = cp.empty((o,6))
        

#%% Main Workspace
if __name__ == '__main__':
    fpath = "D:\\Dropbox\\Data\\3-3-20 glut4-vglutmeos\\" # Folder
    fname = "cell10_dz20_r2.tif" # File name
    #%% Load Image
    im = load_image_to_array(fpath + fname,  20) # Load image into workspace
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
    
    fits = fit_psf(psf, True)
    print("For The Generated Image, it was found with position ({},{}) , with {} photons. ".format(np.round(fits[0]+6,3),np.round(6+fits[1],3),np.round(fits[2],3)))
    print("The width was {} pixels in 'x', and {} pixels in 'y', background = {}. ".format(np.round(fits[3],3),np.round(fits[4],3),np.round(fits[5],3)))
    show_as_image(psf)    
    # Save relevant data
    fits = fit_psf_array(psf_image_array)
    #psf_image_array[2,1,3] = 26
    #psf_image_array[1,0,0] = 26
    #psf_image_array[0,1,0] = 26
    #psf_image_array[0,0,1] = 26
    gpu_fits = cp.empty_like(cp.asarray(fits))
    gpu_psf = cp.asarray(psf_image_array)
    fitting_kernel((4,),(1,),(gpu_psf,gpu_fits, 0, gpu_psf.shape[2]))
    print(gpu_fits)
