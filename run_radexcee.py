# Functions to ease the running of radexcee.
# Assumes that analyseLIME is available.

import time
import numpy as np
from radexcee import radexcee
from analyseLIME.analysecube import cubeclass
from scipy.optimize import curve_fit

def multi_radexcee(observations, gridpath, **kwargs):
    """
    Run radexcee for a list of dictionaries.
    """
    print 'Running %d instances.' % len(observations)
    return [run_radexcee(obs, gridpath, **kwargs) for obs in observations]


def run_radexcee(obs, gridpath, **kwargs):
    """
    Run radexcee for the observation and return the samples or percentiles.
    """
    
    # Defaults.
    nwalkers = kwargs.get('nwalkers', 100) 
    nburnin = kwargs.get('nburnin', 100) 
    nsamples = kwargs.get('nsamples', 100) 
    gradients = kwargs.get('gradients', [False, False, False])
    nslabs = kwargs.get('nslabs', 1)
    toreturn = kwargs.get('toreturn', 'sampler')
    
    if toreturn not in ['percentiles', 'sampler', 'samples']:
        raise ValueError('toreturn must be percentiles, sampler or samples.')
    
    # Run the model.
    models = radexcee(obs, gridpath, nslabs=nslabs)
    sampler, samples = models.run_sampler(nwalkers, nburnin, nsamples, gradients=gradients)
    
    # Return the percentiles, samples or sampler.
    if toreturn == 'percentiles':
        return models.sample_percentiles(sampler, nburnin)
    elif toreturn == 'samples':
        return samples
    else:
        return sampler


def sampler_to_samples(sampler, nburnin=0):
    """
    Converts the sampler to samples for plotting figures.
    """
    ndim = sampler.chain.shape[-1]
    return sampler.chain[:, nburnin:, :].reshape((-1, ndim))


def samples_to_percentiles(samples):
    """
    Returns the 16th, 50th and 84th percentiles for each dimension.
    """
    pcnts = [np.percentile(s, [16,50,84]) for s in samples.T]
    return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts])


def parse_percentiles(samples, param, avg=True):
    """
    Parses the percentiles array.
    """
    if param > samples.shape[0] - 1:
        raise ValueError("'param' too big for samples.")
    else:
        param = int(param)
    y = samples[param][0]
    dy = samples[param][1:]
    if avg:
        dy = np.average(dy)
    return y, dy

def samples_median(samples):
    """
    Returns the median of each dimension.
    """
    return [np.median(s) for s in samples.T] 


def pix_to_rvals(cube, pixel):
    """
    Returns the (deprojected) radial value for the pixels.
    """
    return cube.rvals[pixel[1], pixel[0]] * cube.dist
    
    
def random_pixels(cube, npix, **kwargs):
    """
    Returns npix random pixels from the cube which satisfy the 
    position criteria (assumed to be the unprojected offset and 
    polar angle). The kwarg 'space' will limit too find spacing.
    TODO: How do we sample the centre more thoroughly?
    """
    
    # Minimum and maximum radial values.
    r_min = kwargs.get('r_min', 0.)
    r_max = kwargs.get('r_max', np.sqrt(2) * cube.posax.max())
    if r_min >= r_max:
        raise ValueError('r_min >= r_max')
        
    # Minimum and maxium theta values.
    t_min = kwargs.get('t_min', -np.pi)
    t_max = kwargs.get('t_max', np.pi)
    if t_min >= t_max:
        raise ValueError('t_min >= t_max')
        
    # Minimum spacing distance.
    space = kwargs.get('space', None)
    maxtime = kwargs.get('maxtime', 10.)
    
    # Start with [0, 0] in order to not have this position used.
    check = False        
    pxls = np.array([[0, 0]])
    t0 = time.time()
    
    while len(pxls) <= npix:
        
        # Break if time runs over maximum time.
        if time.time() - t0 > maxtime:
            print 'Maximum time reached.'
            break
        
        # On-sky coordinates. Use a normal distribution around 
        # the center in order to increase the sampling of the 
        # inner region.
        
        x = -1
        y = -1

        while x not in np.arange(cube.npix):
            x = np.random.randn() * cube.npix / 2.
            x += cube.x0
            x = int(x)

        while y not in np.arange(cube.npix):
            y = np.random.randn() * cube.npix / 2.
            y += cube.y0
            y = int(y)

        r = np.hypot(cube.posax[y], cube.posax[x])
        t = np.arctan2(cube.posax[y], cube.posax[x])
        
        # Radial offsets.
        if r < r_min:
            continue
        if r > r_max:
            continue
                
        # Polar offsets.
        if t < t_min:
                continue
        if t > t_max:
                continue 
        
        # Minimum spacing.
        if space is not None:
            yy, xx = cube.posax[y], cube.posax[x]
            s = 0
            for p in pxls:
                dy = yy - cube.posax[p[0]]
                dx = xx - cube.posax[p[1]]
                if np.hypot(dy, dx) < space:
                    s += 1
            if s > 0:
                continue
        
        # Check that it is not already in the list.
        if check:
            if not any(np.sum(abs(pxls - [y, x]), 1) == 0):
                pxls = np.vstack([pxls, [y, x]])
        else:
            pxls = np.vstack([pxls, [y, x]])
            check = True
            
    return pxls[1:]

def samples_median(samples):
    """
    Returns the median of each dimension.
    """
    return [np.median(s) for s in samples.T]   
    
def makedict(cube, pixels, **kwargs):
    """
    Returns a dictionary suitable to send to radexcee. 
    The pixel chosen is specified by (xidx, yidx).
    The keys must include:
    rms - the RMS noise value.
    velax - the velocity axis on which to compare spectra.
    fluxcal - the flux calibration uncertainty (percentage of max value)
    intensity - the spectrum.
    """

    dic = {}
    
    # Defaults
    yidx, xidx = pixels
    nFWHM = kwargs.get('nFWHM', None)
    dic['fluxcal'] = kwargs.get('fluxcal', 0.0)
    snr = kwargs.get('snr', 10.)
    
    # Include noise specified by the SNR provided.
    # and then convert to an RMS noise value.
    inten = cube.onlyline()[:, yidx, xidx].copy()
    dic['rms'] = inten.max() / snr
    
    # Fit a gaussian and then mask spectra to given nFWHM.
    velax = cube.velax
    if nFWHM is not None:
        width = curve_fit(gaussian, velax, inten)[0][1]
        inten = inten[abs(velax) < nFWHM * 2.355 * width]
        velax = velax[abs(velax) < nFWHM * 2.355 * width]
        
    dic['velax'] = velax
    inten += np.random.randn(inten.size) * dic['rms']
    dic['intensity'] = inten    
    return dic
 
def gaussian(x, x0, dx, A):
    """
    Simple Gaussian function.
    """
    return A * np.exp(-0.5 * np.power((x-x0)/dx, 2.))
