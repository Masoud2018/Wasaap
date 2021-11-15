
from lib.pinhole_generator import pinholes
from lib.fourier_centeral import ff , iff
import numpy as np


def retrieve_intensity(img,pixel_size,pitch_size):


    # IMGs are symmetric, assymetry needs a code change!

    N_x = img.shape[0]
    N_y = img.shape[1]


    # Real and Fourier coordinates:
    H_N_x = int(N_x/2)-1
    H_N_y = int(N_y/2)-1

    x = np.linspace((-H_N_x+1)*pixel_size,(H_N_x)*pixel_size,N_x)
    y = np.linspace((-H_N_y+1)*pixel_size,(H_N_y)*pixel_size,N_y)

    [x,y] = np.meshgrid(x,y)
    du_x = 2 * np.pi/(N_x*pixel_size)
    du_y = 2 * np.pi/(N_y*pixel_size)
    u_x = np.linspace((-H_N_x+1)*du_x,(H_N_x)*du_x,N_x)
    u_y = np.linspace((-H_N_y+1)*du_y,(H_N_x)*du_y,N_y)
    [ux,uy] = np.meshgrid(u_y,u_x)
    Q_mesh = [ux,uy]

    # PITCH SIZE IN FOURIER SPACE
    Q_y = N_y*pixel_size/pitch_size

    # FOURIER TRANSFORMATION 
    F_I = ff(img)

    # ISOLATING CENTRAL LOBE IN CIRCLE TO 1/4th OF Q - 
    # IF Q_Y THEN du_y, otherwise changes to X&x.
    F_I_ret = F_I * pinholes(Q_mesh).CircPinhole(int(Q_y/4)*du_x) 
    I_ret = iff(F_I_ret)
    I_ret = np.abs(I_ret)
    I_ret[I_ret<0.05*I_ret.max()]=0
    I_ret = I_ret/I_ret.max()

    return(I_ret)