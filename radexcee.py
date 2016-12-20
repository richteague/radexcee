import emcee
import numpy as np
from radexgridclass import radexgrid
from extractspectra import spectrum
import scipy.constants as sc

class radexcee:
    """
    Class to run emcee sampling of fitting multiple lines.
    """
    
    labels = [r'$\Delta V_{\rm turb}$', 
              r'$T_{\rm kin}$', 
              r'$\log_{10} \, n({\rm H_2})$', 
              r'$\log_{10}\,{\rm N(CO)}$',
              r'$\delta \Delta V_{\rm turb}$',
              r'$\delta T_{\rm kin}$',
              r'$\delta \log_{10} \, n({\rm H_2})$']
    
    def __init__(self, obsdict, grid_path, **kwargs):
        
        # Provided
        self.parse_to_dictionaries(obsdict)     
        self.grid = radexgrid(grid_path)       
        
        # Defaults
        self.maxrange = kwargs.get('maxrange', 2.0)
        self.nslabs = kwargs.get('nslabs', 5)
        return
    
    def parse_to_dictionaries(self, obsdict):
        """
        Splits the provided dictionary into several dictionaries.
        """
        self.trans = sorted(obsdict.keys())
        self.ntrans = len(self.trans)
        self.velaxes = {j : obsdict[j]['velax'] for j in self.trans}
        self.dvelaxes = {j : np.diff(obsdict[j]['velax']).mean() for j in self.trans}
        self.lines = {j : obsdict[j]['intensity'] for j in self.trans}
        self.rms = {j : obsdict[j]['rms'] for j in self.trans}
        self.fluxcal = {j : obsdict[j]['fluxcal'] for j in self.trans}
        return
    
    def lnpriors(self, theta):
        """
        Log-prior function. First four parameters should lie in the RADEX
        grid. The last three should be less than self.maxrange.
        """
        for i in range(4):
            if not (self.grid.vals[i*3] <= theta[i] <= self.grid.vals[i*3+1]):
                return -np.inf
        for i in range(4,7):
            if not (0. <= theta[i] <= self.maxrange):
                return -np.inf
        return 0.0

    def combine_theta(self, theta, g):
        """
        Parses theta using gradients.
        """
        t = [tt for tt in theta[:4]]
        t = t + [theta[4+sum(g[:i])] if g[i] else 0 for i in range(len(g))]
        return t
        
    def lnprob(self, theta, gradients):
        """
        Log-probability function.
        """  
        
        # Update theta. Includes the gradients in the end
        # and includes the first as the total FWHM of the line.
        theta = self.combine_theta(theta, gradients)
        if theta[0] >= 0:
            theta[0] = self.grid.total_linewidth(theta)
        else:
            return -np.inf 
                
        # Log-Priors.
        lp = self.lnpriors(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        # Log-Likelihood.
        return lp + self.lnlike(theta)

        
    def lnlike(self, theta):
        """
        Log-likelihood function.
        Each model is the summation of homogeneous slab models.
        Use Chi-square as the distance metric.
        """         
        models = {j : self.calc_models(j, theta) for j in self.trans}
        ll = [self.lnchi2(models[j], self.lines[j], self.rms[j], 
                          self.fluxcal[j]) for j in self.trans]
        return np.nansum(ll)

        
    def lnchi2(self, model, observation, rms, fluxcal):
        """
        Returns the log-chi_squared value.
        """
        unc = np.hypot(rms, model.max() * fluxcal)
        unc *= 5.  
        lnx2 = ((observation - model) / unc)**2
        lnx2 -= 0.5 * np.log(2. * np.pi * unc**2)
        return -0.5 * np.sum(lnx2)

    def calc_models(self, j, theta):
        """
        Build a spectra from several slab models.
        """
        
        # Gradients in properties.
        params = zip(self.param_gradient(theta[0], theta[4]), 
                     self.param_gradient(theta[1], theta[5]),
                     self.param_gradient(theta[2], theta[6]),
                     self.slab_columns(theta[3]))       
        
        # Combine slabs.
        mods = [self.grid.get_spectra(j, p, self.velaxes[j]) for p in params]
        mods = np.sum(mods, axis=0)
        
        # Flux calibration.
        scale = abs(1. + np.random.randn() * self.fluxcal[j])
        return scale * mods


    def param_gradient(self, pmean, prange):
        """
        Gradient in the parameters.
        Limit the minimum value to zero.
        """
        pmin = (1. - prange)
        pmin = max(0, pmin)
        pmax = (1. + prange)
        return pmean * np.linspace(pmin, pmax, self.nslabs)


    def slab_columns(self, logcolumn):
        """
        Returns column density of a slab.
        """
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
        
        
    def run_sampler(self, nwalkers, nburnin, nsamples, 
                          gradients=[False, False, False]):
        """
        Call emcee to run on the models with specified parameters.
        gradients specifies if there's a gradient in that parameter or not.
        """
        
        # Number of dimensions.
        ndim = int(4 + sum(gradients))
        
        # Starting positions.
        pos = np.array([np.random.random_sample(ndim) 
                        for w in range(nwalkers)])
        pos[:,0] *= 1e3
        for d in range(4):
            pos[:, d] *= self.grid.gridranges[d]
            pos[:, d] += self.grid.vals[3*d]
        for d in range(ndim-4, ndim):
            pos[:, d] = abs(pos[:, d])
            
        # Run the sampler.
        sampler = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        self.lnprob, 
                                        args=[gradients])
        
        # Use a double burn-in setup. First burn-in uses all
        # parameters available for the sampler. The second uses
        # the median values from the former. This should remove
        # cases of walkers getting stuck in poor regions of parameter
        # space.
        _, _, _ = sampler.run_mcmc(pos, int(nburnin*0.5))
        initial = sampler.chain.reshape((-1, ndim))
        sampler.reset()
        
        # Resample the points and second burn-in.
        p0 = self.samples_median(initial)
        pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]     
        _, _, _ = sampler.run_mcmc(pos, int(nburnin+nsamples))   
        samples = sampler.chain[:,nburnin:,:].reshape((-1, ndim))
        return sampler, samples
        
        
    def samples_median(self, samples):
        """
        Returns the median of each dimension.
        """
        return [np.median(s) for s in samples.T]



        
 
