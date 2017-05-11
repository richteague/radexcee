"""

Part of the radexcee.py package.

A class to read in and provide simple interpolation of the pre-calculated RADEX
grids created with radex_grid.py. Requires the use of limepy to read in the
molecular weight.

The inputs are:

    j - transition in the LAMDA file (this is not necessarily the lower quantum
        number as is the case for linear rotators).
    w - width (stdev) of the line including the thermal broadening in [km/s].
    t - kinetic temperature of the slab [K].
    r - log(n(H2)), number density of the main collider in [log(/ccm)].
    s - log(N(mol)), column density of the emitting molecule in [log(/sqcm)].

The returns are:

    interpolate_intensity() - integrated intensity in [K km/s].
    interpolate_tau() - optical depth at the line centre.
    slab() - spectra of the line [K] along a velocity axis in [km/s]. If no
             spectral axis is provided, will use a +/- 4 * dV axis with 50
             channels.

"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from limepy.analysis.collisionalrates import ratefile


class radexgrid:

    def __init__(self, path, **kwargs):
        """Load up the pre-calculated rates using get_radex_grid.py."""
        self.path = path
        self.filename = self.path.split('/')[-1]
        self.molecule = self.filename.split('_')[0]
        self.mu = ratefile(self.molecule).mu

        # The grid follows the standards in calc_radexgrid.py with the filename
        # providing the information about the axes:
        #       species_widths_temperatures_densities_columns.npy
        # where each variable contains the three values,
        #       minval_maxval_nvals,
        # with number in the '%.2f' format.

        self.grid = np.load(self.path)
        self.grid = np.where(np.isfinite(self.grid), self.grid, 0.0)
        self.parse_filename()

        # Default values to help with the interpolation.
        # To make sure that the slabs are sufficiently optically thin, we can
        # check, if checktau == True, if the optical depth of each slab is less
        # than or equal to tauthick.

        self.checktau = kwargs.get('checktau', False)
        self.tauthick = kwargs.get('tauthick', 0.66)
        self.verbose = kwargs.get('verbose', False)
        self.fwhm = 2. * np.sqrt(np.log(2) * 2)

        return

    def parse_filename(self):
        """Parse grid axes from filename."""
        self.vals = [float(v) for v in self.filename[:-4].split('_')[1:]]
        self.width = np.linspace(self.vals[0], self.vals[1], self.vals[2])
        self.temp = np.linspace(self.vals[3], self.vals[4], self.vals[5])
        self.dens = np.linspace(self.vals[6], self.vals[7], self.vals[8])
        self.sigma = np.linspace(self.vals[9], self.vals[10], self.vals[11])
        return

    def indices(self, j, w, t, r, s):
        """Returns the indices required for the interpolation."""
        w = np.interp(w, self.width, np.arange(self.vals[2]))
        t = np.interp(t, self.temp, np.arange(self.vals[5]))
        r = np.interp(r, self.dens, np.arange(self.vals[8]))
        s = np.interp(s, self.sigma, np.arange(self.vals[11]))
        return [[w], [t], [r], [s]]

    def interpolate_intensity(self, j, w, t, r, s):
        """Returns integrated intensity in [K km/s]. Width is not FWHM."""
        idxs = self.indices(j, self.fwhm * w, t, r, s)
        inten = map_coordinates(self.grid[j, 0], idxs, order=1, mode='nearest')
        return inten * w * np.sqrt(2. * np.pi)

    def interpolate_tau(self, j, w, t, r, s):
        """Interpolate the optical depth grid."""
        idxs = self.indices(j, w, t, r, s)
        return map_coordinates(self.grid[j, 1], idxs, order=1, mode='nearest')

    def slab(self, j, w, t, r, s, velax=None):
        """Emission from a slab model."""
        if velax is None:
            velax = np.linspace(-w, w, 50) * 4.
        I = self.interpolate_intensity(j, w, t, r, s)
        if self.checktau:
            if self.interpolate_tau(j, w, t, r, s) > self.tauthick:
                if self.verbose:
                    print('WARNING: Slab is optically thick.')
        return I * self.normgaussian(velax, w)

    @staticmethod
    def gaussian(x, dx, A, x0=0.0):
        '''Gaussian function.'''
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2.))

    @staticmethod
    def normgaussian(x, dx, x0=0.0):
        '''Normalised Gaussian function.'''
        return np.exp(-0.5 * np.power((x-x0)/dx, 2.)) / np.sqrt(2.*np.pi) / dx
