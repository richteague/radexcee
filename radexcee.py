import numpy as np
from radexgridclass import radexgrid
from extractspectra import spectrum
import scipy.constants as sc

class radexcee:
    
    def __init__(self, obsdict, grid_path):
        """Initialise."""
        self.parse_to_dictionaries(obsdict)     
        self.grid = radexgrid(grid_path)       
        return
    
    def parse_to_dictionaries(self, obsdict):
        """Splits the provided dictionary into several dictionaries."""
        self.trans = sorted(obsdict.keys())
        self.ntrans = len(self.trans)
        self.velaxes = {j : obsdict[j]['velax'] for j in self.trans}
        self.dvelaxes = {j : np.diff(obsdict[j]['velax']).mean() for j in self.trans}
        self.lines = {j : obsdict[j]['intensity'] for j in self.trans}
        self.rms = {j : obsdict[j]['rms'] for j in self.trans}
        self.fluxcal = {j : obsdict[j]['fluxcal'] for j in self.trans}
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


    def lnprob(self, theta, gradients=[0, 0, 0], nslabs=5):
        """Log-probability function."""  
        
        # Number of slabs.
        self.nslabs = nslabs
        
        # Priors.
        lp = self.lnpriors(theta, gradients)
        if not np.isfinite(lp):
            return -np.inf
        
        # Likelihood.
        return lp + self.lnlike(theta, gradients)

        
    def lnlike(self, theta, gradients):
        """Log-likelihood function."""
        
        # Each model is the summation of homogeneous slab models.               
        models = [self.calc_models(j, theta, gradients) for j in self.trans]

        # Use Chi-square as the distance metric.
        ll = [self.lnchi2(model, self.lines[j], self.rms[j], self.fluxcal[j]) 
              for model, j in zip(models, self.trans)]
        return np.nansum(ll)

        
    def lnchi2(self, model, observation, rms, fluxcal):
        """Returns the log-chi_squared value."""
        unc = np.hypot(rms, model * fluxcal)        
        lnx2 = ((observation - model) / unc)**2
        lnx2 -= 0.5 * np.log(2. * np.pi * unc**2)
        return -0.5 * np.sum(lnx2)


    def pgradients(self, theta, gradients):
        """Returns the gradients in the physical parameters."""
        if len(theta) == 4:
            return [0. for i in range(len(gradients))]
        g = [int(gg != 0) for gg in gradients]
        return [theta[4+sum(g[:i])] if g[i] else 0. for i in range(len(g))]


    def calc_models(self, j, theta, gradients):
        """Build a spectra from several slab models."""
        
        # Calculate the parameters for each slab model.
        ranges = self.pgradients(theta, gradients)
        params = zip(self.param_gradient(theta[0], ranges[0]), 
                     self.param_gradient(theta[1], ranges[1]),
                     self.param_gradient(theta[2], ranges[2]),
                     self.slab_columns(theta[3]))       
        
        # Combine slabs.
        mods = [self.grid.get_spectra(j, p, self.velaxes[j]) for p in params]
        mods = np.sum(mods, axis=0)
        
        # Include flux calibration.
        scale = 1. + np.random.randn() * self.fluxcal[j]
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
 
