"""Class for the pre-calculated RADEX grids calcualted with
`get_radex_grid.py`. Provides the functions to interpolate said grid and return
the total integrated intensity and the optical depth."""

import os
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from analyseLIME.readLAMDA import ratefile


class radexgrid:

    def __init__(self, path, **kwargs):
        """Load up the pre-calculated rates using get_radex_grid.py."""
        self.path = path
        self.filename = self.path.split('/')[-1]
        if self.filename[-4:] != '.npy':
            raise ValueError('Must be a .npy file.')
        self.grid = np.load(self.path)
        self.grid = np.where(np.isfinite(self.grid), self.grid, 0.0)
        self.mu = self.read_molecular_weight()
        self.parse_filename()
        self.fwhm = 2. * np.sqrt(np.log(2) * 2)
        return

    def parse_filename(self):
        """Parse grid axes from filename."""
        self.vals = self.filename[:-4].split('_')[1:]
        self.vals = [float(v) for v in self.vals]
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

    def read_molecular_weight(self):
        """Read the molecular weight from collisional rates."""
        mol = self.filename.split('_')[0]
        path = os.getenv('RADEX_DATAPATH')
        rates = ratefile(path+mol+'.dat')
        return rates.mu
