""" Class for the pre-calculated RADEX grids. """

import os
import numpy as np
import scipy.constants as sc
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
        """
        Parses the pre-calculated grid axes from the filename. First is
        molecule name, then, in couples of min value, max value and number of
        points, are linewidth, temperature, logdensity and logsigma.
        Assumes the input is a .npy file and density and column density are log.
        """
        self.vals = self.filename[:-4].split('_')[1:]
        self.vals = [float(v) for v in self.vals]
        self.linewidth = np.linspace(self.vals[0], self.vals[1], self.vals[2])
        self.temperature = np.linspace(self.vals[3], self.vals[4], self.vals[5])
        self.logdensity = np.linspace(self.vals[6], self.vals[7], self.vals[8])
        self.logsigma = np.linspace(self.vals[9], self.vals[10], self.vals[11])
        return

    # Interpolation routines. Uses linear interpolation but higher orders can
    # be specified through 'order=2'. 
    # j : index of the transition.
    # t : [non-thermal width [m/s], 
    #      kinetic temperature [K], 
    #      log10(nH2 number density [cm^-3]), 
    #      log10(column density [cm^-2])]
    
    def interpolate_indices(self, j, vals):
        """Returns the values required for the interpolation."""
        width = self.total_linewidth(vals) * self.fwhm
        w = np.interp(width, self.linewidth, np.arange(self.vals[2]))
        t = np.interp(vals[1], self.temperature, np.arange(self.vals[5]))
        r = np.interp(vals[2], self.logdensity, np.arange(self.vals[8]))
        s = np.interp(vals[3], self.logsigma, np.arange(self.vals[11]))
        return [[w], [t], [r], [s]]

    def interpolate_intensity(self, j, vals):
        """Interpolate the intensity grid."""
        indices = self.interpolate_indices(j, vals)
        return map_coordinates(self.grid[j,0], indices, order=1, mode='nearest')
    
    def interpolate_tau(self, j, vals):
        """Interpolate the optical depth grid."""
        indices = self.interpolate_indices(j, vals)
        return map_coordinates(self.grid[j,1], indices, order=1, mode='nearest')

    # Model spectra.
    # For a given velocity axis, 'velax' [km/s], generate a spectrum from 
    # a slab specified by 't'. If attenuate is True then the returned 
    # spectrum is attenuated by a factor of (1 - exp(-tau)).
 
    def gaussian(self, x, a, b, c):
        """Simple Gaussian function, width is Doppler-b."""
        return a * np.exp(-0.5 * np.power((x - b) / c, 2))

    def get_spectra(self, j, vals, velax, attenuate=False):
        """Single Gaussian spectrum."""
        #TODO: Remove attenuate if possible.
        peak_intensity = self.interpolate_intensity(j, vals)
        line_center = 0.0
        linewidth = self.total_linewidth(vals)
        line = self.gaussian(velax, peak_intensity, line_center, linewidth)
        if attenuate:
            return line * (1. - self.get_tau(j, vals, velax))
        else:   
            return line
            
    def get_tau(self, j, t, velax):
        """Returns the (Gaussian) optical depth profile for the slab. """
        peak_opacity = self.interpolate_tau(j, t)
        line_center = 0.0
        linewidth = self.total_linewidth(t)
        return self.gaussian(velax, peak_opacity, line_center, linewidth)
    
    def total_linewidth(self, vals, kms=True):
        """Returns total Doppler-b linewidth."""
        linewidth = np.hypot(vals[0], self.thermal_linewidth(vals))
        if kms:
            linewidth /= 1e3
        return linewidth
    
    def thermal_linewidth(self, vals):
        """Returns the thermal Doppler-b linewidth."""
        return np.sqrt(2. * sc.k * vals[1] / self.mu / sc.m_p)
        
    # Additional functions.

    def randomValues(self, turb_only=True, fwhm=False, kms=False):
        """
        Return random width, temperature, density and column value for
        the given grid. Useful for testing emcee routines.
        """
        vals = np.squeeze([self.randomValue(i) for i in range(4)])
        if turb_only:
            dv_full = vals[0]**2
            dv_therm = self.thermal_fwhm(vals[1])**2
            if dv_therm >= dv_full:
                vals[0] = 0
            else:
                vals[0] = np.sqrt(dv_full - dv_therm)
        if not fwhm:
            vals[0] /= 2. * np.sqrt(np.log(2) * 2)
        if not kms:
            vals[0] *= 1e3
        return vals
     
    def randomValue(self, i):
        """Returns a random value for the ith variable."""
        return np.random.ranf() * (self.vals[1+(i*3)] - self.vals[(i*3)]) + self.vals[i*3]
    
    def read_molecular_weight(self):
        """Read the molecular weight from collisional rates."""
        mol = self.filename.split('_')[0]
        path = os.getenv('RADEX_DATAPATH')
        rates = ratefile(path+mol+'.dat')
        return rates.mu

