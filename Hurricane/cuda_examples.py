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
import matplotlib.pyplot as plt
import scipy
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
        double fits[6] = {0, 0, 0, 1.6, 1.6, 200}; // preallocate fitting array
        double psf[pix*pix];
        double pi = 3.14159265358979323846; // close enough
        
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
        for(int i = 0; i<pix; i++){
        
        }
        fits[0] = fits[0]/fits[2]; // Get center of mass by dividing through the sum        
        fits[1] = fits[1]/fits[2]; // Center of mass y component
        
        // At this point, we have loaded our PSF into our thread and populated our fit vectors with proper initial guess
        // This has been confirmed through reading out the following line of code and comparing with CPU versions of the process        
        // for(int i= 0; i < 6; i++){fit_array[i+ index*6] = fits[i];}   AJN 5/21/20 everything written in this kernel after this line is untested
        
        // Begin the fitting loops
        for(int cycle = 0; cycle < cycles; cycle ++){ // Begin Fitting Cycle Loop
            
            
            double d_1[6] = {0, 0, 0, 0, 0, 0};
            double d_2[6] = {0, 0, 0, 0, 0, 0};
            
            // Calculate pixel values for derivatives, 2nd derivatives, errorfunctions and u
			for (int row = 0; row < pix; row++){	// Begin Row Loop
				for (int col = 0; col < pix; col++){	// Begin column loop
                    // Initialize calculation arrays
                    double derivative[6] = {0, 0, 0, 0, 0, 0};
                    double derivative_2[6] = {0, 0, 0, 0, 0, 0};
                    
                    double psf_pixel = psf[row + col*pix];
                    
                    // these terms are very regularly used in the subsequent calculations
                    double xp = (xgrid[row + col*pix] - fits[0] + 0.5); // short for 'x-plus' indicates that relative half pixel shift in the positive direction
                    double xpg = exp(-powf(xp,2)/(2*powf(fits[3],2)));  // This is the corresponding gaussian value
                    double xm = (xgrid[row + col*pix] - fits[0] - 0.5); // by contrast the negative direction
                    double xmg = exp(-powf(xm,2)/(2*powf(fits[3],2)));
                    
                    double yp = (ygrid[row + col*pix] - fits[1] + 0.5); // Corresponding y values
                    double ypg = exp(-powf(yp,2)/(2*powf(fits[4],2)));
                    double ym = (ygrid[row + col*pix] - fits[1] - 0.5);
                    double ymg = exp(-powf(ym,2)/(2*powf(fits[4],2)));
                    
                    // Define the estimated gaussian
                    double Ex = 0.5 * (erf(xp / sqrt(2.0 * powf(fits[3], 2))) - erf(xm / sqrt(2.0 * powf(fits[3], 2)))); // x component gaussian
					double Ey = 0.5 * (erf(yp / sqrt(2.0 * powf(fits[4], 2))) - erf(ym / sqrt(2.0 * powf(fits[4], 2)))); // Y component gaussian
					double u = fits[2]*Ex*Ey + fits[5]; // pixel of estimated PSF
                    
                    // Calculate pixel contributions to derivative
                    derivative[0] = Ey*fits[2]/(sqrt(2*pi)*fits[3])*(xmg - xpg);
                    derivative[1] = Ex*fits[2]/(sqrt(2*pi)*fits[4])*(ymg - ypg);
                    derivative[2] = Ex*Ey;
                    derivative[3] = Ey*fits[2]/(sqrt(2*pi)*powf(fits[3],2))*( xm* xmg - xp* xpg);
                    derivative[4] = Ex*fits[2]/(sqrt(2*pi)*powf(fits[4],2))*( ym* ymg - yp* ypg);
                    derivative[5] = 1;
                    
                    derivative_2[0] = Ey*fits[2]/(sqrt(2*pi)*powf(fits[3],3))*(xm*xmg - xp*xpg);
                    derivative_2[1] = Ex*fits[2]/(sqrt(2*pi)*powf(fits[4],3))*(ym*ymg - yp*ypg);
                    derivative_2[2] = 0;
                    derivative_2[3] = Ey*fits[2]/sqrt(2*pi)*((powf(xm,3)*xmg - powf(xp, 3)*xpg)/powf(fits[3],5) - 2*(xm*xmg - xp*xpg)/powf(fits[3],3));
                    derivative_2[4] = Ex*fits[2]/sqrt(2*pi)*((powf(ym,3)*ymg - powf(yp, 3)*ypg)/powf(fits[4],5) - 2*(ym*ymg - yp*ypg)/powf(fits[4],3));
                    derivative_2[5] = 0;
                    
                    // I'm reasonably confident that these work' From here we just have to add the correction to the sum
                    for (int i = 0 ; i<6; i++){d_1[i] +=  derivative[i] * (psf_pixel / u - 1); }
                    for (int i = 0 ; i<6; i++){d_2[i] +=  derivative_2[0]*(psf_pixel / u - 1) - powf(derivative[0], 2) * psf_pixel/powf(u, 2); }

                
                } // Finish Column loop
            } // Finish Row Loop
            
            for(int i = 0; i<6; i++){fits[i] -= d_1[i]/d_2[i];} // make the corrections
        } // Finish Fitting Cycle Loop
        // assign final fitting parameters to output vector
        for(int i= 0; i < 6; i++){fit_array[i+ index*6] = fits[i];}
    } // Finish PSF calculation
} // Finish Kernel
''', 'fitting_kernel')

def gpu_psf_fit(images):
    m,n,o = image_size(images)
    fits = cp.empty((o,6))
        
def simulate_psf_array(N = 100, pix = 5):
    psf_image_array = np.zeros((2*pix+1,2*pix+1,N))
    x_build = np.linspace(-pix,pix,2*pix +1)
    X , Y = np.meshgrid(x_build, x_build)
    truths = np.zeros((N,6))
    
    for i in range(N):
        # Start by Determining your truths
        truths[i,0] = np.random.uniform(-0.5, 0.5)
        truths[i,1] = np.random.uniform(-0.5, 0.5)
        truths[i,2] = np.random.uniform(1000, 3000)
        truths[i,3] = np.random.uniform(1.4, 1.401)
        truths[i,4] = np.random.uniform(1.4, 1.401)
        truths[i,5] = np.random.uniform(1, 3)
        
        x_gauss = 0.5 * (scipy.special.erf( (X - truths[i,0] + 0.5) / (np.sqrt(2 * truths[i,3]**2))) - scipy.special.erf( (X - truths[i,0] - 0.5) / (np.sqrt(2 * truths[i,3]**2))))
        y_gauss = 0.5 * (scipy.special.erf( (Y - truths[i,1] + 0.5) / (np.sqrt(2 * truths[i,4]**2))) - scipy.special.erf( (Y - truths[i,1] - 0.5) / (np.sqrt(2 * truths[i,4]**2))))
        #print(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5]))
        psf_image_array[:,:,i] = np.random.poisson(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5]))
    return psf_image_array, truths
def display_results(gpu_results):
    starting_with = gpu_results.shape[0]
    gpu_results = gpu_results[~np.isnan(gpu_results[:,0]),:]
    print('Recovered {}% of localization'.format(np.round(100*gpu_results.shape[0]/starting_with,2)))
    x_bins = np.linspace(-0.5,0.5,100)
    plt.hist(gpu_results[:,0], bins= x_bins, alpha = 0.5, label='x-difference')
    plt.hist(gpu_results[:,1], bins= x_bins, alpha = 0.5, label='y-difference')
    plt.legend(loc='upper right')
    plt.title('X-Y Localization Errors')
    plt.xlabel('Difference in Pixels')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure()
    
    xbins = np.linspace(-0.4,0.1,100)
    plt.hist(gpu_results[:,2], bins= xbins, label='n-difference')
    #plt.hist(gpu_results[:,5], bins= xbins, label='b-difference')
    plt.xlabel('Fractional Error in Gaussian Volume(Photons)')
    plt.ylabel('Frequency')
    plt.title('N Fractional Errors')
    plt.show()
    
    plt.figure()
    xbins = np.linspace(-0.4,0.8,100)
    plt.hist(gpu_results[:,3], bins= xbins, label='sx-difference')
    plt.hist(gpu_results[:,4], bins= xbins, label='sy-difference')
    plt.legend(loc='upper right')
    plt.title('Sigma Errors')
    plt.show()
    
def test_gpu_fit(N = 100, show = False):
    psfs, truths = simulate_psf_array(N)
    print('Simulated')
    gpu_fits = cp.empty_like(cp.asarray(truths))
    gpu_psf = cp.asarray(psfs)
    fitting_kernel((N,),(1,),(gpu_psf,gpu_fits, 0, gpu_psf.shape[2], 20))
    gpu_differences = truths - cp.asnumpy(gpu_fits)
    gpu_results = gpu_differences
    gpu_results[:,2] /= truths[:,2]
    if show: 
        display_results(gpu_results)
    return gpu_results, truths

#%% Main Workspace
if __name__ == '__main__':
    results, truths = test_gpu_fit(10000, True)
