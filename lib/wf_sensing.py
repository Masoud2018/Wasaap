
import numpy as np
import scipy as sp
from typing import Tuple


from lib.zernike import derive, eval_cartesian, j_to_mn, Wavefront, \
    ZernikePolynomial
from lib.fast_zernike import zernike_derivative_cartesian




class WFS:


    def __init__(self,
                 relative_shifts: np.ndarray,
                 aperture_size: int = 32,
                 ) -> None:

        self.relative_shifts = relative_shifts
        self.aperture_size = aperture_size

        self.grid_size = self.relative_shifts.shape[0]

    def fit_wavefront(self,
                      n_zernike: int = 9,
                      ) -> np.ndarray:
        

        n_ap = self.grid_size**2


        p = np.concatenate((self.relative_shifts[:, :, 0].reshape(n_ap),
                            self.relative_shifts[:, :, 1].reshape(n_ap)))

       
        d = np.full((n_zernike, 2 * n_ap), np.nan)

        x_0 = 1 / np.sqrt(2) * np.linspace((1 / self.grid_size - 1),
                                           (1 - 1 / self.grid_size),
                                           self.grid_size).reshape(1, -1)
        x_0 = np.repeat(x_0, self.grid_size, axis=0)
        y_0 = 1 / np.sqrt(2) * np.linspace((1 - 1 / self.grid_size),
                                           (1 / self.grid_size - 1),
                                           self.grid_size).reshape(-1, 1)
        y_0 = np.repeat(y_0, self.grid_size, axis=1)

  
        for row_idx, j in enumerate(range(1, n_zernike+1)):

            m, n = j_to_mn(j)

    
            if j <= 135:
                x_derivatives: np.ndarray = \
                    zernike_derivative_cartesian(m, n, x_0, y_0, 'x')
                y_derivatives: np.ndarray = \
                    zernike_derivative_cartesian(m, n, x_0, y_0, 'y')
            else:
                zernike_polynomial = ZernikePolynomial(m=m, n=n).cartesian
                x_derivatives = \
                    eval_cartesian(derive(zernike_polynomial, 'x'), x_0, y_0)
                y_derivatives = \
                    eval_cartesian(derive(zernike_polynomial, 'y'), x_0, y_0)

            d[row_idx] = np.concatenate((x_derivatives.flatten(),
                                         y_derivatives.flatten()))

       
        e = d @ d.transpose()


        a = sp.linalg.lstsq(a=e, b=d @ p)[0]

        a = np.insert(a, 0, 0)

        return a

    @staticmethod
    def get_unit_disk_meshgrid(
        resolution: int,
            ) -> Tuple[np.array, np.array]:
        """
        Get a (Cartesian) mesh grid of positions on the unit disk, that is,
        all positions with with a Euclidean distance <= 1 from (0, 0).

        Args:
            resolution: An integer specifying the size of the mesh grid,
                that is, the number of points in each dimensions.

        Returns:
            A mesh grid consisting of the tuple `x_0`, `y_0`, which are each
            numpy arrays of shape `(resolution, resolution)`. For positions
            that are on the unit disk, they contain the coordinates of the
            position; otherwise, they contain `np.nan`.
        """

        # Create a meshgrid of (Cartesian) positions: [-1, 1] x [-1, 1]
        x_0, y_0 = np.meshgrid(np.linspace(-1, 1, resolution),
                            np.linspace(-1, 1, resolution))

        # Create a mask for the unit disk (only select position with radius <= 1)
        unit_disk_mask = np.sqrt(x_0**2 + y_0**2) <= 1

        # Mask out all the position that are not on the unit disk
        x_0[~unit_disk_mask] = np.nan
        y_0[~unit_disk_mask] = np.nan

        return x_0, y_0

    @staticmethod
    def get_psf(wavefront: Wavefront,
                resolution: int = 2048,
                ) -> np.ndarray:
        
        #Compute the point spread function (PSF) that corresponds to the
        #given `wavefront`. 
        
        # Get a grid of the unit disk
        x_0, y_0 = get_unit_disk_meshgrid(resolution=resolution)

        # Compute the wavefront on a grid of the given resolution, and cast
        # np.nan to 0, because the FFT cannot deal with NaN
        wf_grid = eval_cartesian(expression=wavefront.cartesian,
                                 x_0=x_0,
                                 y_0=y_0)
        wf_grid = np.nan_to_num(wf_grid)

        # Compute the pupil function. In our simple case, this is simply the
        # unit disk, i.e., it is 1 where the radius is <= 1, and 0 elsewhere
        pupil = np.logical_not(np.logical_or(np.isnan(x_0), np.isnan(y_0)))
        pupil = pupil.astype(np.float)


        lambda_ = 13.5e-9

        # Compute the corresponding point spread function (PSF)
        psf = pupil * np.exp(2 * np.pi * 1j / lambda_ * wf_grid/2/np.pi)
        psf = np.fft.fft2(psf)
        psf = np.abs(psf)**2
        psf = np.fft.fftshift(psf) / np.max(psf)

        return psf

    