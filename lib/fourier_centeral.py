
import numpy as np
import numpy.fft as nfft

# Fourier transormesr:
def ff(a):
    return(nfft.fftshift(nfft.fft2(nfft.ifftshift(a),norm = 'ortho')))
def iff(a):
    return(nfft.fftshift(nfft.ifft2(nfft.ifftshift(a),norm = 'ortho')))