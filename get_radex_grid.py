""" 
Script to generate a grid of brightness temperatures from RADEX for a
specified slab model. Can either be called from the command line or 
externally as a function.

Timing seems to scale linearly.
1e2 entries: 00:01 
1e3 entries: 00:19 
1e4 entries: 03:23
1e5 entries: 32:43
"""

import os
import sys
import time
import pyradex
import numpy as np
from makeLime import lamdaclass

# Suppress all the numerical warnings from pyradex.
import warnings
warnings.filterwarnings("ignore")

def runRadex(species, widths, temperatures, densities, columns, **kwargs):
    """Calls pyradex iteratively to produce a table of intensities."""

    # Make sure all the provided values are iterable.
    
    temperatures, tnum, tmin, tmax = formatInput(temperatures, log=False)
    densities, dnum, dmin, dmax = formatInput(densities, log=True)
    columns, cnum, cmin, cmax = formatInput(columns, log=True)
    widths, wnum, wmin, wmax = formatInput(widths, log=False)
    
    # Check that the collisional rate file exists.
    # This is hardwired to where the collisional rates are.
    
    rates_path = os.getenv('RADEX_DATAPATH')
    if not os.path.isfile('{}{}.dat'.format(rates_path, species)):
        raise ValueError('Not found collisional rates.')
    rates = lamdaclass.ratefile('{}{}.dat'.format(rates_path, species))
        
    # We assume that the density is n(H2) and the ortho/para ratio is 3.
    # Check if oH2 and pH2 are valid colliders in the collisional rate file.
    # If they are, recalculate the densities.
    
    opr = kwargs.get('opr', 3.)
    if ('oH2' in rates.partners and 'pH2' in rates.partners):
        opr_flag = True
        print 'Assuming an ortho / para ratio of {}.'.format(opr)
    else:
        opr_flag = False
    
    # Dummy array to hold the results.
    # Hold up the 'jmax' transition, default is 10.
    # Saves both the brightness temperature and optical depth.

    jmax = kwargs.get('jmax', 9) + 1
    Tb = np.zeros((jmax, 2, wnum, tnum, dnum, cnum))    

    # First initialise pyradex, then iterate through all the permutations.

    radex = pyradex.Radex(species=species, temperature=temperatures[0],
                          density=densityDict(densities[0], opr, opr_flag),
                          column=columns[0], deltav=widths[0], **kwargs)

    t0 = time.time()
    for l, width in enumerate(widths):
        radex.deltav = width
        for t, temp in enumerate(temperatures):
            radex.temperature = temp
            for d, dens in enumerate(densities):
                radex.density = densityDict(dens, opr, opr_flag)
                for c, col in enumerate(columns):
                    radex.column = col
                    with np.errstate(divide='ignore'):
                        radex.run_radex(reuse_last=False, reload_molfile=True)
                    Tb[:, 0, l, t, d, c] = radex.T_B[:jmax]
                    Tb[:, 1, l, t, d, c] = radex.tau[:jmax]
    Tb = np.nan_to_num(Tb)
    t1 = time.time()

    print 'Generated table in {}.'.format(seconds2hms(t1-t0))

    # Use the filename convention, 
    #       species_widths_temperatures_densities_columns.npy
    # where each variable contains the three values,
    #       minval_maxval_nvals,
    # with number in the '%.2f' format. This will help with reading 
    # in the file.
    
    fn = '{}_'.format(species)
    fn += '{:.2f}_{:.2f}_{:d}_'.format(wmin, wmax, wnum)
    fn += '{:.2f}_{:.2f}_{:d}_'.format(tmin, tmax, tnum)
    fn += '{:.2f}_{:.2f}_{:d}_'.format(dmin, dmax, dnum)
    fn += '{:.2f}_{:.2f}_{:d}.npy'.format(cmin, cmax, cnum)
    np.save(kwargs.get('path', './') + fn, Tb)
    
    return


def formatInput(arr, log=True):
    """Make sure the input is iterable and find the min and max values."""
    if type(arr) is not np.ndarray:
        amin = arr
        amax = arr
        arr = [arr]
    else:
        amin = arr.min()
        amax = arr.max()
    anum = len(arr)
    if log:
        amin = np.log10(amin)
        amax = np.log10(amax)
    return arr, anum, amin, amax

def densityDict(density, opr, opr_flag):
    """Return a dictionary of the densities"""
    if opr_flag:
        return {'oH2': density * opr / (opr + 1.),
                'pH2': density * 1. / (opr + 1.)}
    else:
        return {'H2': density}
       
def seconds2hms(seconds):
    """Convert seconds to hours, minutes, seconds."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)

# Allow to be called from the command line.
if __name__ == '__main__':
    species = str(sys.argv[1])
    vals = [float(argv) for argv in sys.argv[2:]]
    widths = np.linspace(vals[0], vals[1], vals[2])
    temperatures = np.linspace(vals[3], vals[4], vals[5])
    densities = np.logspace(vals[6], vals[7], vals[8])
    columns = np.logspace(vals[9], vals[10], vals[11])
    runRadex(species, widths, temperatures, densities, columns)

