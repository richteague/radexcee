import numpy as np
from radexgridclass import radexgrid
from extractspectra import spectrum
import scipy.constants as sc

class radexcee:
    
    def __init__(self, paths, snr, grid_path, **kwargs):
        """
        spectra   : list of spectra classes from `extractspectra.py`.
        snr       : list of the snr ratios to use for each line. If
                    only a single value is given, assume the same for
                    all the lines.
        grid_path : path to the pre-calculated line intensity grid.
        """
        
        #TODO: Clean up the whole reading in thing.
        
        try:
            iter(paths)
        except TypeError:
            self.paths = [paths]
        else:
            self.paths = paths
        
        self.spectra = [spectrum(p) for p in self.paths]
        self.trans = [int(spec.transition) for spec in self.spectra]
        self.ntrans = len(self.trans)
        
        try:
            iter(snr)
        except TypeError:
            self.snr = [snr for s in self.spectra]
        else:
            self.snr = snr 
        
            
        if len(self.snr) != len(self.spectra):
            raise ValueError()
        toiter = zip(self.spectra, self.snr)
        self.lines = [spec.noisySpectra(s) 
                      for spec, s in toiter]
        self.velax = self.spectra[0].velax
        self.dvelax = np.diff(self.velax).mean()
        self.grid = radexgrid(grid_path)
        self.rms = [np.nanmax(spec.noisySpectra(s)) / s for spec, s in toiter]
        
        
        return
        
    def lnpriors(self, theta, gradients):
        """Log-prior function."""
        
        # Consider only the mean values first.
        # The RADEX grid requires the FWHM of the line. The free
        # parameter is the non-thermal width (Doppler-b) of the line.
        # For extra parameters, clip temperatures to the grid values.

        if theta[0] < 0:
            return -np.inf
        else:
            fwhm = self.grid.total_linewidth(theta)
        
        temp = theta[1]
        dens = theta[2]
        colu = theta[3]
        
        for i, p in enumerate([fwhm, temp, dens, colu]):
            if not (self.grid.vals[i*3] <= p <= self.grid.vals[i*3+1]):
                return -np.inf
        
        # If appropriate, check to make sure that all the gradients
        # are both positive and less than the specified limit.
        if len(theta) > 4:
            toiter = zip(self.pgradients(theta, gradients), gradients)
            if not all([0 <= g <= gmax for g, gmax in toiter]):
                return -np.inf
        
        return 0.0


    def lnprob(self, theta, fluxcal=0.0, gradients=[0, 0, 0], nslabs=5):
        """Log-probability function."""  
        
        # Number of slabs.
        self.nslabs = nslabs
        
        # Priors.
        lp = self.lnpriors(theta, gradients)
        if not np.isfinite(lp):
            return -np.inf

        # Flux calibration for each of the lines. This will be inclued in 
        # the uncertainties and as a rescaling of the model spectra.
        fluxcal = self.check_iterable(fluxcal) 
        
        # Likelihood.
        return lp + self.lnlike(theta, fluxcal, gradients)

        
    def lnlike(self, theta, fluxcal, gradients):
        """
        Log-likelihood function.
        """
        
        # Each model is the summation of homogeneous slab models.               
        models = [self.calc_models(j, theta, fc, gradients) 
                  for j, fc in zip(self.trans, fluxcal)]

        # Use Chi-square as the distance metric.
        toiter = zip(models, self.lines, self.rms, fluxcal)
        lnlike = [self.lnchi2(m, o, u, c) for m, o, u, c in toiter]
        return np.nansum(lnlike)

        
    def lnchi2(self, model, obs, rms, fluxcal=0.0):
        """
        Returns the log-chi_squared value. Also models the flux calibration.
        """
        unc = np.hypot(rms, model * fluxcal)        
        lnx2 = ((obs - model) / unc)**2
        lnx2 -= 0.5 * np.log(2. * np.pi * unc**2)
        return -0.5 * np.sum(lnx2)


    def pgradients(self, theta, gradients):
        """Returns the gradients in the physical parameters."""
        if len(theta) == 4:
            return [0. for i in range(len(gradients))]
        g = [int(gg != 0) for gg in gradients]
        return [theta[4+sum(g[:i])] if g[i] else 0. for i in range(len(g))]


    def calc_models(self, j, theta, fluxcal, gradients):
        """Build a spectra from several slab models."""
        
        # Calculate the parameters for each slab model.
        ranges = self.pgradients(theta, gradients)
        params = zip(self.param_gradient(theta[0], ranges[0]), 
                     self.param_gradient(theta[1], ranges[1]),
                     self.param_gradient(theta[2], ranges[2]),
                     self.slab_columns(theta[3]))       
        
        # Generate a spectrum for each slab then combine.
        # Just simply add them, need to update this for thicker lines.
        mods = np.array([self.grid.get_spectra(j, p, self.velax) for p in params])
        mods = np.sum(mods, axis=0)
        
        # Include flux calibration.
        scale = 1. + np.random.randn() * fluxcal
        return abs(scale) * mods


    def param_gradient(self, pmean, prange):
        """Gradient in the parameters."""
        pmin = (1. - prange * 0.5)
        pmax = (1. + prange * 0.5)
        return pmean * np.linspace(pmin, pmax, self.nslabs)


    def slab_columns(self, logcolumn):
        """Returns column density of a slab."""
        c = np.log10(np.power(10, logcolumn) / self.nslabs)
        return [c for n in np.arange(self.nslabs)]


    def check_iterable(self, inp):
        """
        Checks if the input is an iterable of same length as observations,
        or a single number. If the latter, will return an iterable of the
        correct length.
        """
        if hasattr(inp, '__iter__'):
            if len(inp) == self.ntrans:
                return inp
            else:
                raise ValueError('Mismatch in lengths of iterables.')
        return [inp for i in range(self.ntrans)]
        
    '''
    ## Obsolete. ##
    def temperature_step(self, T):
        """Returns the appropriate temperature step."""
        # Note the 1e3 factor is to account for the [km/s] > [m/s] change.
        sigma_fwhm = self.sigma * self.dvelax * 1e3 / self.snr[0] / 0.6
        mean_width = self.thermal_width(T)
        delta_temp = (mean_width - sigma_fwhm)**2
        delta_temp *= self.mu * sc.m_p / 2. / sc.k
        delta_temp = abs(delta_temp - T)
        return delta_temp 
        
    ## Obsolete. ##
    def thermal_width(self, T):
        """Thermal width of a line."""
        return np.sqrt(2. * sc.k * T / self.mu / sc.m_p)
   '''
        
