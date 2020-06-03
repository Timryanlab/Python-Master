# -*- coding: utf-8 -*-
"""
Title:

Concept:
Created on Thu May 28 15:06:51 2020

@author: Andrew Nelson
"""
#% Import Libraries
from ryan_image_io import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from localization_kernels import *

from mpl_toolkits.mplot3d import Axes3D

#% Functions
def simulate_psf_array(N = 100, pix = 5):
    psf_image_array = np.zeros((2*pix+1,2*pix+1,N))
    x_build = np.linspace(-pix,pix,2*pix +1)
    X , Y = np.meshgrid(x_build, x_build)
    truths = np.zeros((N,6))
    loc1 = Localizations()
    
    orange, red = loc1.get_sigma_curves()
    loc1.split = 0
    for i in range(N):
        # Start by Determining your truths
        truths[i,0] = np.random.uniform(-0.5, 0.5)
        truths[i,1] = np.random.uniform(-0.5, 0.5)
        truths[i,2] = np.random.uniform(1000, 3000)
        ind = np.random.randint(0,orange.shape[1])
        if truths[i,0] <= loc1.split:    
            truths[i,3] = orange[1,ind]
            truths[i,4] = orange[2,ind]
        else:
            truths[i,3] = red[1,ind]
            truths[i,4] = red[2,ind]
        truths[i,3] += np.random.normal(0,0.001)
        truths[i,4] += np.random.normal(0,0.001)
        truths[i,5] = np.random.uniform(1, 3)
        
        x_gauss = 0.5 * (erf( (X - truths[i,0] + 0.5) / (np.sqrt(2 * truths[i,3]**2))) - erf( (X - truths[i,0] - 0.5) / (np.sqrt(2 * truths[i,3]**2))))
        y_gauss = 0.5 * (erf( (Y - truths[i,1] + 0.5) / (np.sqrt(2 * truths[i,4]**2))) - erf( (Y - truths[i,1] - 0.5) / (np.sqrt(2 * truths[i,4]**2))))
        #print(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5]))
        psf_image_array[:,:,i] = np.random.poisson(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5]))
    return psf_image_array, truths

#%% Localization Class
class Localizations:
    def __init__(self, filename, pixel_size = 0.13):
        self.
        self.xf = np.array([])
        self.yf = np.array([])
        self.zf = np.array([])
        self.N = np.array([])
        self.sx = np.array([])
        self.sy = np.array([])
        self.o = np.array([])
        self.frames = np.array([])
        self.pixel_size = pixel_size
        self.xf_error = np.array([])
        self.yf_error = np.array([])
        self.sx_error = np.array([])
        self.sy_error = np.array([])
        cal_fpath = 'C:\\Users\\andre\\Documents\\GitHub\\Python-Master\\Hurricane\\'
        self.cal_files = [cal_fpath + 'z_calib.mat',  # axial calibration
                          cal_fpath + '2_color_calibration.mat']  # 2 color calibration
        self.store_calibration_values()
        
    def show_localizations(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
    
        ax.scatter(self.xf, self.yf, self.zf)
        
        ax.set_xlabel('X um')
        ax.set_ylabel('Y um')
        ax.set_zlabel('Z um')
        fig.show()
    
    def show_axial_sigma_curve(self):
        fig = plt.figure()
        (ax1, ax2) = fig.subplots(1,2) 
        
        ax1.scatter(self.zf[self.color], self.sx[self.color], color = 'orange', alpha = 0.8)
        ax1.scatter(self.zf[self.color], self.sy[self.color], color = 'blue', alpha = 0.8)
        ax1.set_xlabel('Axial Position in um')
        ax1.set_ylabel('Gaussian Width in um')
        
        
        ax2.scatter(self.zf[~self.color], self.sx[~self.color], color = 'orange', alpha = 0.8)
        ax2.scatter(self.zf[~self.color], self.sy[~self.color], color = 'blue', alpha = 0.8)
        ax2.set_xlabel('Axial Position in um')
        ax2.set_ylabel('Gaussian Width in um')

        fig.show()
        
    def store_fits(self, fitting_vectors, crlb_vectors, frames):
        self.xf = self.pixel_size*(fitting_vectors[:,0])
        self.yf = self.pixel_size*(fitting_vectors[:,1])
        self.zf = np.empty_like(self.xf)
        self.N = fitting_vectors[:,2]
        self.sx = np.abs(self.pixel_size*fitting_vectors[:,3])
        self.sy = np.abs(self.pixel_size*fitting_vectors[:,4])
        self.o = fitting_vectors[:,5]
        self.frames = frames
        self.color = self.xf <= self.split*self.pixel_size  # given the 2 color system we can use a boolean variable to relay color info
                                       # False = Red channel True = Orange Channel
        self.get_z_from_widths()
        self.make_z_corrections()
        self.xf_error = self.pixel_size*crlb_vectors[:,0]**0.5
        self.yf_error = self.pixel_size*crlb_vectors[:,1]**0.5
        self.sx_error = self.pixel_size*crlb_vectors[:,3]**0.5
        
    def get_z_from_widths(self):
        # Depending on color, we should load either orange or red z params
        x = np.linspace(-0.5,0.5,1000) # gives nanometer level resolution, well below actual resolution
        
        # Build curve for orange calibration
        xs = self.orange_z_calibration[0]
        gx = self.orange_z_calibration[1]
        dx = self.orange_z_calibration[2]
        ax = self.orange_z_calibration[3]
        bx = self.orange_z_calibration[4]
        
        ys = self.orange_z_calibration[5]
        gy = self.orange_z_calibration[6]
        dy = self.orange_z_calibration[7]
        ay = self.orange_z_calibration[8]
        by = self.orange_z_calibration[9]
        
        orange_sigma_x_curve = self.pixel_size*xs*(1 + ((x-gx)/dx)**2 + ax*((x-gx)/dx)**3 + bx*((x-gx)/dx)**4)**0.5
        orange_sigma_y_curve = self.pixel_size*ys*(1 + ((x-gy)/dy)**2 + ay*((x-gy)/dy)**3 + by*((x-gy)/dy)**4)**0.5
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, orange_sigma_x_curve)
        ax.scatter(x, orange_sigma_y_curve)
        ax.set_xlabel('Axial Position in um')
        ax.set_ylabel('Sigma Width Position in um')
        fig.show()
        '''
        # Repeat for red curves
        xs = self.red_z_calibration[0]
        gx = self.red_z_calibration[1]
        dx = self.red_z_calibration[2]
        ax = self.red_z_calibration[3]
        bx = self.red_z_calibration[4]
        
        ys = self.red_z_calibration[5]
        gy = self.red_z_calibration[6]
        dy = self.red_z_calibration[7]
        ay = self.red_z_calibration[8]
        by = self.red_z_calibration[9]
        
        red_sigma_x_curve = self.pixel_size*xs*(1 + ((x-gx)/dx)**2 + ax*((x-gx)/dx)**3 + bx*((x-gx)/dx)**4)**0.5
        red_sigma_y_curve = self.pixel_size*ys*(1 + ((x-gy)/dy)**2 + ay*((x-gy)/dy)**3 + by*((x-gy)/dy)**4)**0.5
        
        del xs, gx, dx, ax, bx, ys, gy, dy, ay, by
        
        for i in range(len(self.xf)):
            if self.color[i]: # If molecule was found in the orange channel, use orange calibration parameters
                D = ((self.sx[i]**0.5 - orange_sigma_x_curve**0.5)**2 +(self.sy[i]**0.5 - orange_sigma_y_curve**0.5)**2)**0.5
                if ~np.isnan(D.min()):
                    index = np.argwhere(D == D.min())
                    self.zf[i] = x[index[0][0]]
                else:
                    self.zf[i] = -0.501
            if ~self.color[i]: # If molecule was found in the orange channel, use orange calibration parameters
                D = ((self.sx[i]**0.5 - red_sigma_x_curve**0.5)**2 +(self.sy[i]**0.5 - red_sigma_y_curve**0.5)**2)**0.5
                if ~np.isnan(D.min()):
                    index = np.argwhere(D == D.min())
                    self.zf[i] = -x[index[0][0]]
                else:
                    self.zf[i] = -0.501
        
    def get_sigma_curves(self):
        # Depending on color, we should load either orange or red z params
        x = np.linspace(-0.5,0.5,1000) # gives nanometer level resolution, well below actual resolution
        
        # Build curve for orange calibration
        xs = self.orange_z_calibration[0]
        gx = self.orange_z_calibration[1]
        dx = self.orange_z_calibration[2]
        ax = self.orange_z_calibration[3]
        bx = self.orange_z_calibration[4]
        
        ys = self.orange_z_calibration[5]
        gy = self.orange_z_calibration[6]
        dy = self.orange_z_calibration[7]
        ay = self.orange_z_calibration[8]
        by = self.orange_z_calibration[9]
        
        orange_sigma_x_curve = xs*(1 + ((x-gx)/dx)**2 + ax*((x-gx)/dx)**3 + bx*((x-gx)/dx)**4)**0.5
        orange_sigma_y_curve = ys*(1 + ((x-gy)/dy)**2 + ay*((x-gy)/dy)**3 + by*((x-gy)/dy)**4)**0.5
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, orange_sigma_x_curve)
        ax.scatter(x, orange_sigma_y_curve)
        ax.set_xlabel('Axial Position in um')
        ax.set_ylabel('Sigma Width Position in um')
        fig.show()
        '''
        # Repeat for red curves
        xs = self.red_z_calibration[0]
        gx = self.red_z_calibration[1]
        dx = self.red_z_calibration[2]
        ax = self.red_z_calibration[3]
        bx = self.red_z_calibration[4]
        
        ys = self.red_z_calibration[5]
        gy = self.red_z_calibration[6]
        dy = self.red_z_calibration[7]
        ay = self.red_z_calibration[8]
        by = self.red_z_calibration[9]
        
        red_sigma_x_curve = xs*(1 + ((x-gx)/dx)**2 + ax*((x-gx)/dx)**3 + bx*((x-gx)/dx)**4)**0.5
        red_sigma_y_curve = ys*(1 + ((x-gy)/dy)**2 + ay*((x-gy)/dy)**3 + by*((x-gy)/dy)**4)**0.5
        
        return np.array([x, orange_sigma_x_curve, orange_sigma_y_curve]), np.array([x, red_sigma_x_curve, red_sigma_y_curve])
        
    def store_calibration_values(self):
        
        #Store axial Calibration file into memory
        mat_dict = loadmat(self.cal_files[0])
        orange = mat_dict['orange']
        self.orange_z_calibration = orange[0][0][0][0]
        self.orange_angle = orange[0][0][3][0][0]
        self.orange_refraction_tilt = orange[0][0][1][0][0]   
        self.orange_x_tilt = orange[0][0][2][0][0][0][0]
        self.orange_y_tilt = orange[0][0][2][0][0][1][0]
        orange_z0_tilt = orange[0][0][2][0][0][2][0]
        self.orange_z_tilt = orange_z0_tilt[1:] + np.mean(np.diff(orange_z0_tilt))/2
        
        orange = mat_dict['red']
        self.red_z_calibration = orange[0][0][0][0]
        
        self.red_refraction_tilt = orange[0][0][1][0][0]   
        self.red_x_tilt = orange[0][0][2][0][0][0][0]
        self.red_y_tilt = orange[0][0][2][0][0][1][0]
        red_z0_tilt = orange[0][0][2][0][0][2][0]
        self.red_angle = orange[0][0][3][0][0]
        self.red_z_tilt = red_z0_tilt[1:] + np.mean(np.diff(red_z0_tilt))/2
        
        # Story 2 color overlay information into memory
        mat_dict = loadmat(self.cal_files[1])
        self.orange_2_red_x_weights = mat_dict['o2rx'][:,0]
        self.orange_2_red_y_weights = mat_dict['o2ry'][:,0]
        self.split = mat_dict['split'][0][0]
        
        
#% Main Workspace
if __name__ == '__main__':
    fpath = 'C:\\Users\\andre\\Documents\\GitHub\\Matlab-Master\\Hurricane\\hurricane_functions\\z_calib.mat'

    
    loc1 = Localizations()
    loc1.split = 0
    psfs, truths = simulate_psf_array(1000)
    fits = fit_psf_array_on_gpu(psfs)
    fits[:,3:4] = np.abs(fits[:,3:4])
    list_of_good_fits = remove_bad_fits(fits)
    keep_vectors = fits[list_of_good_fits,:]
    keep_psfs = psfs[:,:,list_of_good_fits]
    crlb_vectors = get_error_values(keep_psfs, keep_vectors)
    loc1.store_fits(keep_vectors, crlb_vectors, np.array(range(keep_vectors.shape[0])))
    loc1.show_axial_sigma_curve()
    