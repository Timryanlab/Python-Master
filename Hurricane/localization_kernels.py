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
from scipy.special import erf
import time
import warnings

#%% Functions

warnings.filterwarnings("ignore")
'''
Notes on this function
Indexing is weird because of how I index in python. Lesson learned but sticking with my convention for now. 
frame = i
column = frame + number_of_frames*column
row = frame + number_of_frames*column + row*number_of_frames*number_of_columns
'''


fitting_kernel = cp.RawKernel(r'''
extern "C" 
 __global__ void fitting_kernel(const double* psf_array, double* fit_array, double* ang_array, int images, int cycles) {
    const int pix = 11; // pixel size
	__shared__ double xgrid[pix*pix];			// allocate xpix and ypix variables to the shared memory of the blocks
	__shared__ double ygrid[pix*pix];			// this will reduce calls to global device memory if we didn't need rotation we could reduce the variables needed for calculation
    int tx = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    double ang = ang_array[index];
	if (tx == 0){ // create xgrid and ygrid we want to create the grid regardless of whether the index is crunching on an image
		for (int i = 0; i <pix; i++){
			for(int j = 0; j <pix; j++){
				double x = (double)j - ((double)pix-1)/2;
				double y = ((double)i - ((double)pix-1)/2);
				xgrid[j*pix + i] = x*cos(ang) - y*sin(ang);
				ygrid[j*pix + i] = x*sin(ang) + y*cos(ang);
			}
		}
	}
    
    if( index < images){ // ensure that the thread is working on a molecule
        double fits[6] = {0, 0, 0, 1.3, 1.3, 200}; // preallocate fitting array
        double psf[pix*pix];
        double pi = 3.14159265358979323846; // close enough
        
        // populate fitting array with initial guesses
        for(int ii = 0; ii <pix*pix; ii++){
            int i = ii % pix;
            int j = ii / pix;
            psf[ii] = psf_array[index + (j + (i)*pix)*images]; // this is a critical step in the process. Here the index comes in bizzare, but needs to be arranged such that it is a column major process
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
                    for (int i = 0 ; i<6; i++){d_2[i] +=  derivative_2[i]*(psf_pixel / u - 1) - powf(derivative[i], 2) * psf_pixel/powf(u, 2); }

                
                } // Finish Column loop
            } // Finish Row Loop
            for(int i = 0; i<6; i++){fits[i] -= d_1[i]/d_2[i];} // make the corrections
        } // Finish Fitting Cycle Loop
        // assign final fitting parameters to output vector
        float x = fits[0]*cos(-ang) - fits[1]*sin(-ang);
        float y = fits[0]*sin(-ang) - fits[1]*cos(-ang);
        fits[0] = x;
        fits[1] = y;
        for(int i= 0; i < 6; i++){fit_array[i+ index*6] = fits[i];}
    } // Finish PSF calculation
} // Finish Kernel
''', 'fitting_kernel')


        


def display_results(gpu_results):
    starting_with = gpu_results.shape[0]
    #gpu_results = gpu_results[~np.isnan(gpu_results[:,0]),:]
    print('Recovered {}% of localization'.format(np.round(100*gpu_results.shape[0]/starting_with,2)))
    x_bins = np.linspace(-0.2,0.2,100)
    plt.hist(gpu_results[:,0], bins= x_bins, alpha = 0.5, label='x-difference')
    plt.hist(gpu_results[:,1], bins= x_bins, alpha = 0.5, label='y-difference')
    plt.legend(loc='upper right')
    plt.title('X-Y Localization Errors')
    plt.xlabel('Difference in Pixels')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure()
    
    xbins = np.linspace(-0.1,0.1,100)
    plt.hist(gpu_results[:,2], bins= xbins, label='n-difference')
    #plt.hist(gpu_results[:,5], bins= xbins, label='b-difference')
    plt.xlabel('Fractional Error in Gaussian Volume(Photons)')
    plt.ylabel('Frequency')
    plt.title('N Fractional Errors')
    plt.show()
    
    plt.figure()
    xbins = np.linspace(-0.2,0.2,100)
    plt.hist(gpu_results[:,3], bins= xbins, alpha = 0.5, label='sx-difference')
    plt.hist(gpu_results[:,4], bins= xbins, alpha = 0.5,  label='sy-difference')
    plt.legend(loc='upper right')
    plt.title('Sigma Errors')
    plt.show()
    
def test_gpu_fit(N = 100, show = False):
    t = time.clock()
    psfs, truths = simulate_psf_array(N)
    te = time.clock() - t
    print('Simulated took {} s'.format(te))

    gpu_fits = fit_psf_array_on_gpu(psfs)
    #gpu_differences = truths - cp.asnumpy(gpu_fits)
    gpu_differences = truths - gpu_fits
    gpu_results = gpu_differences
    gpu_results[:,2] /= truths[:,2]
    if show: 
        display_results(gpu_results)
    print('Final Timing: it took {}s to run all of this'.format(time.clock()-t))
    return gpu_results, truths


def fit_psf_array_on_gpu(psf_array, rotation = 0, cycles = 20):
    
    N = psf_array.shape[2] # Get number of images to localize
    fits = np.empty((N,6)) # Allocate fits array
    if N <= 2*10**5: # If less than 100k, localize together
        t0 = time.clock()
        gpu_fits = cp.empty_like(cp.asarray(fits))
        gpu_psf = cp.asarray(psf_array)
        d_rotation = cp.asarray(0*np.ones(fits.shape[0]))
        t_load = time.clock()
        if N <1024: # Broadcasting rules
            fitting_kernel((N,),(1,),(gpu_psf,gpu_fits, d_rotation, gpu_psf.shape[2], cycles))
        else:
            fitting_kernel((1024,),( N//1024 + 1,),(gpu_psf,gpu_fits, d_rotation, gpu_psf.shape[2], cycles))
        t_fit = time.clock()
        cpu_fits = cp.asnumpy(gpu_fits)
        t_return = time.clock()
        
        print(' Localized {} molecules'.format(N))
        print('Timings: Loading onto the GPU took {}s'.format(t_load - t0))
        print('Timings: Fitting  on  the GPU took {}s'.format(t_fit - t_load))
        print('Timings: Loading  off the GPU took {}s'.format(t_return - t_fit))
        
        return cpu_fits
    else: # more than 100k divide up the array
        rounds = int(np.ceil(float(N) / (2*10**5))) # This will be the number of rounds required to analyze the array
        # We'll divide the rounds via 100k chunks to maximize computation speed
        times = np.empty((rounds,4))
        for i in range(rounds):
            start = i*2*10**5
            end = np.min([(i+1)*2*10**5,N])
            print('Localizing Molecules {} through {}'.format(start, end))
            times[i,0]= time.clock()
            gpu_fits = cp.empty_like(cp.asarray(fits[start:end,:]))
            gpu_psf = cp.asarray(psf_array[:,:,start:end])
            times[i,1] = time.clock()
            d_rotation = cp.asarray(rotation*np.ones(fits.shape[0]))
            if (end-start) <1024: # Broadcasting rules
                fitting_kernel(((end-start),),(1,),(gpu_psf,gpu_fits, d_rotation, gpu_psf.shape[2], cycles))
            else:
                fitting_kernel((1024,),( (end-start)//1024 + 1,),(gpu_psf,gpu_fits, d_rotation, gpu_psf.shape[2], cycles))
            times[i,2] = time.clock()
            fits[start:end,:] = cp.asnumpy(gpu_fits)
            times[i,3] = time.clock()
        
        print(' Localization of 200k molecules'.format(N))
        loads = times[:,1] - times[:,0]
        fiters = times[:,2] - times[:,1]
        returns = times[:,3] - times[:,2]
        
        loads = loads[~np.isnan(loads)]
        fiters = fiters[~np.isnan(fiters)]
        returns = returns[~np.isnan(returns)]
        
        print('Timings: Loading onto the GPU took {} +/- {}s'.format(loads.mean(), loads.std()/len(loads)**0.5))
        print('Timings: Fitting  on  the GPU took {} +/- {}s'.format(fiters.mean(), fiters.std()/len(fiters)**0.5))
        print('Timings: Loading  off the GPU took {} +/- {}s'.format(returns.mean(), fits.std()/len(returns)**0.5))
        return fits
    
#%% Main Workspace
if __name__ == '__main__':
    results, truths = test_gpu_fit(500, True)


