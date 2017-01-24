'''
Updated version of radexcee used to fit multiple line emission.

Input:
    > .npy file of pre-calculated intensities and optical depths.
    > dictionary containing entries with an integer key of the lower
      transition number. Each entry is a 2D array with the velocity and
      the brightness temperature.

emcee:
    > theta = [Tkin, n(H2), N(mol), dTkin, x0_0, dx_0, ..., x0_i, dx_i]
      where the final centres and widths are in increasing transition.
    > Assumes that RMS and flux calibration are [%] of the peak value.

Output:
    > samples after the burn-in.
    > statistics of the samples?
'''

import emcee
import numpy as np
from radexgridclass import radexgrid


class radexcee:

    def __init__(self, observations, grid_path, **kwargs):
        '''
        observations - dictionary of lines to fit.
        grid_path - path to pre-calculated intensities.
        '''

        # Input
        self.obs = observations
        self.trans = [k for k in sorted(observations.keys()) if type(k) is int]
        self.ntrans = len(self.trans)
        self.grid = radexgrid(grid_path)

        # Defaults
        self.maxdt = kwargs.get('maxdt', 2.0)
        self.nslabs = kwargs.get('nslabs', 5)
        return

    def lnpriors(self, theta):
        '''Log-prior function.'''

        # Minimum and maximum temperatures.
        if theta[3] < 0:
            return -np.inf
        if self.grid.vals[3] > (theta[0] - theta[3]):
            return -np.inf
        if (theta[0] + theta[3]) > self.grid.vals[5]:
            return -np.inf
        if (theta[3] / theta[0]) > self.maxdt:
            return -np.inf

        # Number density.
        if not self.grid.vals[6] <= theta[1] <= self.grid.vals[8]:
            return -np.inf

        # Column density.
        # TODO: Make a harder upper limit for this. Tau check?
        if self.grid.vals[9] > self.slab_column(theta[2]):
            return -np.inf
        if 16. < theta[2]:
            return -np.inf

        # Line center and line widths.
        for i, j in enumerate(self.trans):
            if not self.obs[j][0][0] <= theta[4+(2*i)] <= self.obs[j][0][-1]:
                return -np.inf
            if theta[5+(2*i)] > (self.obs[j][0][-1] - self.obs[j][0][0]) * 0.5:
                return -np.inf

        return 0.0

    def lnprob(self, theta, rms, fluxcal):
        '''Log-probability function.'''
        if not np.isfinite(self.lnpriors(theta)):
            return -np.inf
        return self.lnlike(theta, rms, fluxcal)

    def lnlike(self, theta, rms, fluxcal):
        '''Log-likelihood function.'''
        rms = self.check_iterable(rms, self.ntrans)
        fluxcal = self.check_iterable(fluxcal, self.ntrans)
        models = [self.single_spectrum(i, theta, fluxcal[i]) for i in range(self.ntrans)]
        chi2 = [self.lnchi2(models[i], self.obs[self.trans[i]][1], rms[i], fluxcal[i])
                for i in range(self.ntrans)]
        return np.nansum(chi2)

    def lnchi2(self, model, observation, rms, fluxcal):
        '''Returns log-chi_squared value.'''
        uncertainty = np.hypot(rms, fluxcal) * np.nanmax(observation)
        lnx2 = ((observation - model) / uncertainty)**2
        lnx2 -= 0.5 * np.log(2. * np.pi * uncertainty**2)
        return -0.5 * np.nansum(lnx2)

    def check_iterable(self, val, length):
        '''Check if val is iterable and of the correct length.'''
        try:
            if len(val) != length:
                raise ValueError('len(val) != ntrans')
        except TypeError:
            val = [val for _ in range(length)]
        return val

    def single_spectrum(self, i, theta, fluxcal):
        '''Returns the combined spectra for the i transition.'''

        # Properties for the ensemble of slabs.
        j = self.trans[i]
        temps = np.linspace(-theta[3], theta[3], self.nslabs) + theta[0]


        # Model the emission from each slab, combine them and include
        # flux calibration rescaling if appropriate.
        spectra = [self.grid.get_spectra(j, t, theta[2], theta[1], theta[4+(i*2)],
                                         theta[5+(i*2)], self.obs[j][0])
                                         for t in temps]
        spectrum = np.sum(spectra, axis=0)
        return spectrum * (1. + np.random.randn() * fluxcal)


    def slab_column(self, logSigma):
        '''Column density of each slab.'''
        return np.log10(np.power(10, logSigma) / self.nslabs)
