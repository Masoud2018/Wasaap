# IMPORTS

import os
import numpy as np
import cv2
from typing import Any
import numpy as np

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lib.wf_sensing import WFS
from lib.zernike import Wavefront, eval_cartesian
from lib.retrieve_intensity import retrieve_intensity


import wfs_processor_FLASH as wfs
from config_files import config_mr_beam,config_im_optics


# INPUT

reference_image = config_im_optics.input_config['reference_image']
filepath =  config_im_optics.input_config['filepath']

pixel_size = config_im_optics.input_config['pixel_size']
pitch_size = config_im_optics.input_config['pitch_size']
aoi_size = int(pitch_size/pixel_size)+2
focal_length = config_im_optics.input_config['focal_length']
wavelength =  config_im_optics.input_config['wavelength']
magnification = config_im_optics.input_config['magnification']


# CONFIGURATION --> True : CONV OR DIV ILLUMINATION 
# IN OUR WORK OTHERWISE MEANS  DIRECT (~PLANAR) BEAM
IS_DIV_CONV = True


# PLOT FINAL REULTS:
PLOT = True

# READ REFERENCE 

reference_image = wfs.read_file(reference_image)[0]

# READ DATA AND PROCESS

files = os.listdir(filepath)


for filenum, file in enumerate(files):
    if(file.endswith('.npy')):
        print('File {} of {}: {}'.format(filenum, len(files), file))
        data_filepath = filepath+file
        
        ##recover intensity
        img = wfs.read_file(data_filepath)[0]

        #Parameters
        I_ret = retrieve_intensity(img,pixel_size,pitch_size)
        plt.imshow(I_ret,vmin=0.1,vmax=I_ret.max(),cmap='CMRmap')
        plt.colorbar()
        plt.title('Intensity')
        plt.pause(2)
        plt.close()



        shifts,sx,sy = wfs.process_file(data_filepath, aoi_size, reference_image, 
        focal_length, pixel_size, pitch_size, wavelength, magnification,
        no_div_conv=IS_DIV_CONV)

        
        if IS_DIV_CONV:
            shifts = np.zeros((sx.shape[0],sx.shape[1],2))
            shifts[:,:,0] = sy
            shifts[:,:,1] = sx 


        # PLOTTING SLOPES
        fig , ax = plt.subplots(1,2)
        plt.suptitle('Slopes')
        ax[0].imshow(shifts[:,:,0],cmap='CMRmap')
        ax[1].imshow(shifts[:,:,1],cmap='CMRmap')
        plt.pause(5)
        plt.close()


        # SET UP HSWFS INSTANCE
        print('Setting up wavefront sensor...', end=' ', flush=True)
        sensor = WFS(relative_shifts=shifts)
        print('Done!', flush=True)


        # FIT WAVEFRONT TO SHIFTS  -- FIRST 3 COEFFS ARE SET TO ZERO.
        print('Fitting wavefront...', end=' ', flush=True)
        coefficients = sensor.fit_wavefront(n_zernike=12)
        coefficients[0:3] = np.zeros(3,dtype=np.float)
        wavefront = Wavefront(coefficients=coefficients)
        print('Done!', flush=True)
    

        # -------------------------------------------------------------------------
        # PLOT RESULTS 
        # -------------------------------------------------------------------------

        if PLOT:
            print('Plotting results...', end=' ', flush=True)

            # Plot Zernike Coeff.

            factor = np.pi/360  * 1e6 # --> CONVERTS RADIAN TO MICROMETER
            plt.bar(np.arange(3,3+len(coefficients[3:])),coefficients[3:]*factor)
            dim = np.arange(int(3),int(3+len(coefficients[3:])))
            plt.xticks(dim)
            plt.ylabel('um')
            plt.title('Zernike coeff.')
            plt.pause(10)
            plt.close()
            print('Done!', flush=True)




            # Evaluate the wavefront on a grid and plot it
            # J=0,1,2 and J=4 are excluded!
            
            coefficients[0:3] = np.zeros(3,dtype=np.float) #--> piston-tilt-top (Ansi system)
            coefficients[4] = 0 #--> Defocus (Ansi system)
            wavefront = Wavefront(coefficients=coefficients)
            
            x_0, y_0 = WFS.get_unit_disk_meshgrid(resolution=256)
            wf_grid = eval_cartesian(wavefront.cartesian, x_0=x_0, y_0=y_0)
            limit = 1.1 * np.nanmax(np.abs(wf_grid))
            xy_extent = sy.shape[0]*pitch_size/1e-3
            plt.imshow(wf_grid, interpolation='nearest', cmap='CMRmap',
                                vmin=-limit, vmax=limit, extent=[0,xy_extent,0,xy_extent])
            plt.pause(4)
            plt.close()
