'''
Updated version of radexcee used to fit multiple line emission.
Provides all the functions to build (simple) Gaussian line profiles given a
set of physical parameters: temperature, local density and column density. The
resulting profile is the product of several 0-D slab models.

Input:
    > .npy file of pre-calculated intensities and optical depths.
    > dictionary containing entries with an integer key of the lower
      transition number. Each entry is a 2D array with the velocity and
      the brightness temperature.

emcee:
    > theta = [Tkin, n(H2), N(mol), dTkin, V0_0, dV_0, ..., V0_i, dV_i]
      where the final centres and widths are in increasing transition.
    > Assumes that RMS and flux calibration are [%] of the peak value.

Output:
    > samples after the burn-in.
    > statistics of the samples?
'''

import numpy as np
from radexgridclass import radexgrid
import scipy.constants as sc
from scipy.optimize import curve_fit


class radexcee:

    def __init__(self, observations, grid_path, **kwargs):
        '''Initialise.'''

        # Parse the information from the provided dictionary.
        # Note that the velocity axis should be [km/s].

        self.obs = observations
        self.trans = [k for k in sorted(observations.keys()) if type(k) is int]
        self.ntrans = len(self.trans)
        self.grid = radexgrid(grid_path)
        self.nslabs = kwargs.get('nslabs', 5)

        # Do an intial fit to the data which is used for the priors in the
        # fitting. self.priors is an [N x 2] array where N = self.ntrans * 2.
        # Each row is a best-fit value and its (symmetric) uncertainty.

        self.priors = [self.fit_spectrum(i) for i in range(self.ntrans)]
        self.priors = np.hstack([p for p in self.priors]).T
        if self.priors.shape[0] != self.ntrans * 2:
            raise ValueError('Odd number of priors.')
        print 'Successfully estimated priors.'

        return

    def fit_spectrum(self, i):
        '''Fits the spectrum.'''

        line = self.obs[self.trans[i]]

        # Make a crude estimate for the starting position of curve_fit.
        # The width assumes the line is fully Gaussian.

        peak = np.nanmax(line[1])
        centre = line[0][line[1].argmax()]
        width = peak / np.sqrt(2. * np.pi) / np.trapz(line[1], line[0])
        uncertainty = np.ones(line[0].size) * self.obs['info']['rms'][i]
        uncertainty *= peak

        popt, pcov = curve_fit(self.gaussian, line[0], line[1],
                               p0=[centre, width, peak],
                               sigma=uncertainty,
                               absolute_sigma=True,
                               maxfev=10000)

        # Return the width and centre values with 1 sigma variance.

        return np.squeeze([popt[:2], np.diag(pcov)[:2]])

    def lnpriors(self, theta):
        '''Log-prior function.'''

        # For the temperatures, H2 density and column density we have
        # uninformative priors. Here we just care if they are in the provided
        # grid of intensities. If not, return -np.inf.

        # For the centres and widths of the lines, we have already tried to fit
        # them with a simple Gaussian function. These results will provide the
        # priors for these parameters. Note that the input value for the width
        # is the non-thermal component. We will calculate the total line width
        # from the central temperature, although in practice is +/- 5 K
        # difference will not make a considerable differencee to the width.

        # As we are working with the log-prior, we add up the results from each
        # of the 6+ parameters. Each parameter is assumed to have a Gaussian
        # shaped prior with symmetric uncertainties.

        # Minimum and maximum temperatures. Make sure that dT is positive and
        # that the range of temperatures will be in the temperature grid.

        if theta[3] < 0:
            return -np.inf

        tmax = theta[0] + theta[3]
        tmin = theta[0] - theta[3]
        if tmin < self.grid.temp[0]:
            return -np.inf
        if tmax > self.grid.temp[-1]:
            return -np.inf

        # Number density of the main collider.

        if theta[1] < self.grid.dens[0]:
            return -np.inf
        if theta[1] > self.grid.dens[-1]:
            return -np.inf

        # Column density. Use here the slab column density as this is what's
        # used to calculate the intensity of each slab. TODO: Include a harder
        # upper limit as at high optical depths this results in weirdness.

        if theta[2] < self.grid.sigma[0]:
            return -np.inf
        if theta[2] > 16.:
            return -np.inf

        # If up to here, then all uninformative priors have been met. Calculate
        # the log-prior value for each of the line centres and widths. Note we
        # don't have to check if these values are in the provided velocity axes
        # as they're defined everywhere.

        lnprior = 0.0
        for i in range(len(theta[4:])):

            # Note that every other parameter is a line width. We must convert
            # this from the non-thermal component to the full component. This
            # is found by i%2 == 1. Also check that the non-thermal component
            # is positive.

            if i % 2:
                if theta[4+i] < 0:
                    return -np.inf
                param = self.total_width(theta[0], theta[4+i])
            else:
                param = theta[4+i]

            pprior = self.priors[i]
            lnprior -= np.log(pprior[1] * np.sqrt(2. * np.pi))
            lnprior -= 0.5 * np.power((param - pprior[0]) / pprior[1], 2)

        return lnprior

    def lnprob(self, theta, rms, fluxcal):
        '''Log-probability function.'''
        lnprior = self.lnpriors(theta)
        if not np.isfinite(lnprior):
            return -np.inf
        return lnprior + self.lnlike(theta, rms, fluxcal)

    def lnlike(self, theta, rms, fluxcal):
        '''Log-likelihood function.'''
        rms = self.check_iterable(rms, self.ntrans)
        fluxcal = self.check_iterable(fluxcal, self.ntrans)
        models = [self.single_spectrum(i, theta, fluxcal[i])
                  for i in range(self.ntrans)]
        chi2 = [self.lnchi2(models[i], self.obs[self.trans[i]][1],
                            rms[i], fluxcal[i]) for i in range(self.ntrans)]
        return np.nansum(chi2)

    def lnchi2(self, model, observation, rms, fluxcal):
        '''Returns log-chi_squared value.'''
        uncertainty = np.hypot(rms, fluxcal) * np.nanmax(observation)
        lnx2 = -0.5 * ((observation - model) / uncertainty)**2
        lnx2 -= np.log(uncertainty * np.sqrt(2. * np.pi))
        return np.nansum(lnx2)

    def check_iterable(self, val, length):
        '''Check if val is iterable and of the correct length.'''
        try:
            if len(val) != length:
                raise ValueError('len(val) != ntrans')
        except TypeError:
            val = [val for _ in range(length)]
        return val

    def single_spectrum(self, i, theta, fluxcal):
        '''Returns a spectra for the i transition.'''

        # The spectra are modelled as the simple summation of a number of 0D
        # slab models. Each slab, with equal column density, is considered to
        # have the same physical properties other than kinetic temperature.

        # The emission from each slab is modelled from pre-calculated RADEX
        # intensities. The line width used is a combination of the thermal and
        # non-thermal components. By interpolating the provided grid, an
        # intensity for each slab is calculated. This intensity is then
        # distributed across the velocity axis assuming a Gaussian form.

        # The final spectrum is just the sum of all these spectra. As the
        # emission is considered optically thin then no antennuation is
        # considered.

        # Properties for the ensemble of slabs.
        j = self.trans[i]
        temps = np.linspace(-theta[3], theta[3], self.nslabs) + theta[0]

        # Model the emission from each slab, combine them and include
        # flux calibration rescaling if appropriate.

        spectra = [self.spectrum(j, t, theta[2], theta[1],  theta[4+(i*2)],
                                 theta[5+(i*2)], self.obs[j][0])
                   for t in temps]
        spectrum = np.sum(spectra, axis=0)
        return spectrum * (1. + np.random.randn() * fluxcal)

    def slab_column(self, logSigma):
        '''Column density of each slab.'''
        return np.log10(np.power(10, logSigma) / self.nslabs)

    def total_width(self, temperature, nonthermal):
        '''Total line width in [km/s].'''
        thermal = np.sqrt(2. * sc.k * temperature / sc.m_p / self.grid.mu)
        return np.hypot(nonthermal, thermal) * 1e-3

    def spectrum(self, j, temp, sigma, rho, centre, ntwidth, velax):
        '''Gaussian spectrum from intensity.'''
        width = self.total_width(temp, ntwidth)
        inten = self.grid.interpolate_intensity(j, width, temp, rho, sigma)
        return self.gaussian(velax, centre, width, inten / np.sqrt(2. * np.pi))

    @staticmethod
    def gaussian(x, x0, dx, A):
        '''Gaussian function.'''
        return A * np.exp(-0.5 * np.power((x - x0) / dx, 2.))
