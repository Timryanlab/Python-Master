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
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import pickle

#% Functions
def simulate_psf_array(N = 100, rot = 0, pix = 5):
    """So this will simulate a PSF array that when transferred to the GPU can be
    fit by the localization algorithm located in 'localization_kernels"""
    
    psf_image_array = np.zeros((2*pix+1,2*pix+1,N)) # Preallocate array
    x_build = np.linspace(-pix,pix,2*pix +1) # Get ready to make mesh grid
    X , Y = np.meshgrid(x_build, x_build) # mesh grid
    X1 = np.cos(rot)*X - np.sin(rot)*Y
    Y1 = np.sin(rot)*X + np.cos(rot)*Y
    X = X1
    Y = Y1
    truths = np.zeros((N,6)) # preallocate truths array
    loc1 = Localizations() # Grab calibration file values
    
    orange, red = loc1.get_sigma_curves() # Grab the axial calibration values 
    loc1.split = 0
    for i in range(N):
        # Start by Determining your truths
        truths[i,0] = np.random.uniform(-1, 1)*0 + 0.8
        truths[i,1] = np.random.uniform(-1, 1)*0 + 0.6 # Position
        x = 0
        y = 0
        truths[i,2] = np.random.uniform(1000, 3000) # Photons
        ind = np.random.randint(0,orange.shape[1]) # Because we're selecting from a curve, we choose a random position
        if truths[i,0] <= loc1.split: # determine calibration based off position
            truths[i,3] = orange[1,ind]/loc1.pixel_size
            truths[i,4] = orange[2,ind]/loc1.pixel_size
        else:
            truths[i,3] = red[1,ind]/loc1.pixel_size
            truths[i,4] = red[2,ind]/loc1.pixel_size
        truths[i,3] += np.random.normal(0,0.001) 
        truths[i,4] += np.random.normal(0,0.001) # Make some noise!
        truths[i,5] = np.random.uniform(1, 3) # Offset
        
        # Build your x/y guass components
        x_gauss = 0.5 * (erf( (X - x + 0.5) / (np.sqrt(2 * truths[i,3]**2))) - erf( (X - x - 0.5) / (np.sqrt(2 * truths[i,3]**2))))
        y_gauss = 0.5 * (erf( (Y - y + 0.5) / (np.sqrt(2 * truths[i,4]**2))) - erf( (Y - y - 0.5) / (np.sqrt(2 * truths[i,4]**2))))
        #print(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5]))
        psf_image_array[:,:,i] = np.random.poisson(np.round(truths[i,2]*x_gauss*y_gauss + truths[i,5])) # make some noise and return it
    return psf_image_array, truths

#%% Localization Class
class Localizations:
    def __init__(self, file_name='temp_name.tif', pixel_size = 0.13, pixel_width = 5):
        self.name = file_name
        self.xf = np.array([])
        self.yf = np.array([])
        self.zf = np.array([])
        self.N = np.array([])
        self.sx = np.array([])
        self.sy = np.array([])
        self.o = np.array([])
        self.frames = np.array([])
        self.pixel_size = pixel_size
        self.pixel_width = pixel_width
        self.xf_error = np.array([])
        self.yf_error = np.array([])
        self.sx_error = np.array([])
        self.sy_error = np.array([])
        # Load Calibration Files
        
        cal_fpath = 'C:\\Users\\' + get_computer_name() + '\\Documents\\GitHub\\Python-Master\\Hurricane\\'
        self.cal_files = [cal_fpath + '3d_calibration.pkl',  # Matlab axial calibration
                          cal_fpath + '2_color_calibration.mat',  # 2 color calibration
                          cal_fpath + 'z_calib.mat'] # Python 3D axial Calibration

        self.store_calibration_values() 
        
    def show_all(self):
        """Perform a 3D scatter plot of X-Y-Z localization data"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
    
        ax.scatter(self.xf, self.yf, self.zf)
        
        ax.set_xlabel('X um')
        ax.set_ylabel('Y um')
        ax.set_zlabel('Z um')
        fig.show()
    
    def show_axial_sigma_curve(self):
        """ Show z vs. Sigma curves"""
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
        """When provided with localization data, store it into the class"""
        self.xf = self.pixel_size*(fitting_vectors[:,0])
        self.yf = self.pixel_size*(fitting_vectors[:,1])
        self.zf = np.empty_like(self.xf)
        self.N = fitting_vectors[:,2]
        self.sx = np.abs(self.pixel_size*fitting_vectors[:,3])
        self.sy = np.abs(self.pixel_size*fitting_vectors[:,4])
        self.o = fitting_vectors[:,5]
        self.frames = frames
        self.color = self.xf >= self.split*self.pixel_size  # given the 2 color system we can use a boolean variable to relay color info
                                       # False = Red channel True = Orange Channel
        self.get_z_from_widths() # Z assignment
        self.make_z_corrections() # X-Y corrections based on astigmatism tilt
        
        self.xf_error = self.pixel_size*crlb_vectors[:,0]**0.5
        self.yf_error = self.pixel_size*crlb_vectors[:,1]**0.5
        self.sx_error = self.pixel_size*crlb_vectors[:,3]**0.5
    
    def make_z_corrections(self):
        
        for i in range(self.xf.shape[0]):
            
            if self.color[i]:
                self.xf[i] -= self.model_orange_x_axial_correction(self.zf[i])/self.pixel_width
                self.yf[i] -= self.model_orange_y_axial_correction(self.zf[i])/self.pixel_width
                self.zf[i] /= self.orange_refraction_correction
            else:
                self.xf[i] -= self.model_red_x_axial_correction(self.zf[i])/self.pixel_width
                self.yf[i] -= self.model_red_y_axial_correction(self.zf[i])/self.pixel_width
                self.zf[i] /= self.red_refraction_correction

        
    def get_z_from_widths(self):
        # Depending on color, we should load either orange or red z params
        x = np.linspace(-0.8,0.8,1600) # gives nanometer level resolution, well below actual resolution
        orange_sigma_x_curve = self.model_orange_sx(x)
        orange_sigma_y_curve = self.model_orange_sy(x)
 

        red_sigma_x_curve = self.model_red_sx(x)
        red_sigma_y_curve = self.model_red_sy(x)

        for i in range(len(self.xf)):
            if self.color[i]: # If molecule was found in the orange channel, use orange calibration parameters
                D = ((self.sx[i]**0.5 - orange_sigma_x_curve**0.5)**2 +(self.sy[i]**0.5 - orange_sigma_y_curve**0.5)**2)**0.5
                if ~np.isnan(D.min()):
                    index = np.argwhere(D == D.min())
                    self.zf[i] = x[index[0][0]]
                else:
                    self.zf[i] = -1
            if ~self.color[i]: # If molecule was found in the orange channel, use orange calibration parameters
                D = ((self.sx[i]**0.5 - red_sigma_x_curve**0.5)**2 +(self.sy[i]**0.5 - red_sigma_y_curve**0.5)**2)**0.5
                if ~np.isnan(D.min()):
                    index = np.argwhere(D == D.min())
                    self.zf[i] = x[index[0][0]]
                else:
                    self.zf[i] = -1
                     
    def get_sigma_curves(self):
        # Depending on color, we should load either orange or red z params
        x = np.linspace(-0.5,0.5,1000) # gives nanometer level resolution, well below actual resolution
        
        orange_sigma_x_curve = self.model_orange_sx(x)
        orange_sigma_y_curve = self.model_orange_sy(x)
        red_sigma_x_curve = self.model_red_sx(x)
        red_sigma_y_curve = self.model_red_sy(x)
        
        orange_sigma_curves = np.array([x, 
                                        orange_sigma_x_curve, 
                                        orange_sigma_y_curve])
        red_sigma_curves = np.array([x, 
                                     red_sigma_x_curve, 
                                     red_sigma_y_curve])
        
        return orange_sigma_curves, red_sigma_curves
        
    def store_calibration_values(self):
        
        #Store axial Calibration file into memory
        # Calibrations based off matlab file
       
        # python 3D Calibration 
        with open(self.cal_files[0], 'rb') as f:
            calibration_dict = pickle.load(f)
        self.model_orange_sx = calibration_dict['model_orange_sx']
        self.model_orange_sy = calibration_dict['model_orange_sy']
        self.model_orange_x_axial_correction = calibration_dict['model_orange_x_axial_correction']
        self.model_orange_y_axial_correction = calibration_dict['model_orange_y_axial_correction']
        self.orange_angle = calibration_dict['orange_angle']
        self.orange_refraction_correction = calibration_dict['orange_refraction_correction']
        
        self.model_red_sx = calibration_dict['model_red_sx']
        self.model_red_sy = calibration_dict['model_red_sy']
        self.model_red_x_axial_correction = calibration_dict['model_red_x_axial_correction']
        self.model_red_y_axial_correction = calibration_dict['model_red_y_axial_correction']
        self.red_angle = calibration_dict['red_angle']
        self.red_refraction_correction = calibration_dict['red_refraction_correction']
        
        # Story 2 color overlay information into memory
        mat_dict = loadmat(self.cal_files[1])
        self.orange_2_red_x_weights = mat_dict['o2rx'][:,0]
        self.orange_2_red_y_weights = mat_dict['o2ry'][:,0]
        self.split = mat_dict['split'][0][0]
    
    def separate_colors(self):
        """This makes the overlay correction between the channels in X and Y, soon to be Z"""
        self.xf_red = self.xf[~self.color]
        self.yf_red = self.yf[~self.color]
        self.zf_red = self.zf[~self.color]
        self.xf_orange = self.xf[self.color]/self.pixel_size
        self.yf_orange = self.yf[self.color]/self.pixel_size
        self.zf_orange = self.zf[self.color]
        self.x = np.array([self.xf_orange**3, 
                         self.yf_orange**3, 
                         self.xf_orange**2*self.yf_orange, 
                         self.xf_orange*self.yf_orange**2,
                         self.xf_orange**2, 
                         self.yf[self.color]**2,
                         self.xf_orange*self.yf_orange,
                         self.xf_orange, 
                         self.yf_orange,
                         self.xf_orange*0 + 1])
        self.xf_orange = np.matmul(self.orange_2_red_x_weights,self.x)*self.pixel_size
        self.yf_orange = np.matmul(self.orange_2_red_y_weights,self.x)*self.pixel_size
    
def save_localizations(mol_to_save, file_name):
    with open(file_name[:-4] + '_localized.pkl', 'wb') as f:
        pickle.dump(mol_to_save,f)        
        
def load_localizations(file_name):
    with open(file_name, 'rb') as f:
        result = pickle.load(f)
    return result
        
#% Main Workspace
if __name__ == '__main__':
    fpath = 'C:\\Users\\andre\\Documents\\GitHub\\Matlab-Master\\Hurricane\\hurricane_functions\\z_calib.mat'

    
    loc1 = Localizations('Example')
    loc1.split = 0
    psfs, truths = simulate_psf_array(100000, rot = np.pi/4)
    fits = fit_psf_array_on_gpu(psfs, rotation = np.pi/4)
    fits[:,3:4] = np.abs(fits[:,3:4])
    list_of_good_fits = remove_bad_fits(fits)
    keep_vectors = fits[list_of_good_fits,:]
    keep_psfs = psfs[:,:,list_of_good_fits]
    crlb_vectors = get_error_values(keep_psfs, keep_vectors)
    loc1.store_fits(keep_vectors, crlb_vectors, np.array(range(keep_vectors.shape[0])))

    kept_truths = truths[list_of_good_fits,:]
    plt.hist(kept_truths[:,0] - keep_vectors[:,0], bins = 200)
    print((kept_truths[:,0] - keep_vectors[:,0]).std())
