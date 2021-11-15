import numpy as np
from lib.fourier_centeral import ff , iff


class propagation:
    
    def __init__(self,field,wavelength,distance,numerical_grid,pixel_size):
        self.E = field
        self.z = distance 
        self.N = numerical_grid
        self.lam = wavelength
        self.dx = pixel_size
        

    def angular_spectrum(self):

        zcrit = self.N * self.dx**2/self.lam
        if np.abs(self.z) < zcrit:
     
            k = 2*np.pi/self.lam
            du = 2*np.pi / (self.N * self.dx)
            u = np.linspace((-self.N/2+1)*du,(self.N/2)*du,self.N)
            [ux,uy] = np.meshgrid(u,u)
            kernel = np.exp(-1j*(ux**2+uy**2)*self.z/(2*k))
            complex_field = iff(ff(self.E) * kernel)
            complex_field =  complex_field 
            
        return(complex_field)
    
    def fresnel(self):
        
        X = np.linspace((-self.N/2+1)*self.dx,(self.N/2)*self.dx,self.N)
        [XX,YY] = np.meshgrid(X,X)
        q_phase = np.exp(1j*2*np.pi/self.lam*(XX**2+YY**2)/self.z)
        complex_field =  ff(self.E*q_phase)
        
        return(complex_field)

        
    def franhoufer(self):
        
        complex_field = 1j*np.exp(1j*2*np.pi/self.lam*self.z)/self.lamb/self.z*ff(self.E)
        
        return(complex_field)