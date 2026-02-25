#!/usr/bin/env python

import copy
import numpy as np
import scipy.interpolate
from astropy.table import Table
import matplotlib.pyplot as plt

import rosesim

#------------------------------------------------------------------------------

class Isochrone():
    """
    A class to load and manipulate theoretical PARSEC isochrones, converting 
    them to the appropriate Roman Space Telescope filter system (AB magnitudes).
    """
    def __init__(self, logage=10.1, feh=-2.0, distance_modulus=16.0, 
                 band_1='F106', band_2='F158', 
                 filename='/scratch/gpfs/JENNYG/jiaxuanl/Data/SBF/Rosesim/PARSEC/PARSEC_v1.2S_Roman_vega_nTP20.dat',
                 verbose=True):
        self.logage = logage
        self.feh = feh
        self.distance_modulus = distance_modulus

        self.band_1 = band_1
        self.band_1_detection = True
        self.band_2 = band_2
        self.imf_type = 'Kroupa'
        self.hb_stage = 4
        self.hb_spread = 0.1
        self.filename = filename
        self.verbose = verbose
        
        self._parse(self.filename)

    def _parse(self, filename):
        """
        Reads the PARSEC data file, filters by age and metallicity, 
        and converts Vega magnitudes to AB magnitudes.
        """
        if self.verbose:
            print(f"Loading isochrone data from {filename}...")

        columns = [l.split() for l in open(filename) if 'Zini' in l][0][1:]  # strip leading '#'

        self.data = np.genfromtxt(filename, comments='#', names=columns)
        self.data = self.data[np.abs(self.data['logAge'] - self.logage) < 0.01]
        self.data = self.data[np.abs(self.data['MH'] - self.feh) < 0.01]
        self.data = Table(self.data)
        
        filters = [column for column in columns if 'mag' in column]
        self.data.rename_columns(filters, [item.replace('mag', '') for item in filters])

        self.mass_act = self.data['Mass']
        self.luminosity = 10**self.data['logL']
        self.mag_1 = self.data[self.band_1]
        self.mag_2 = self.data[self.band_2]
        self.stage = self.data['label']

        # Convert Vega to AB using the external rosesim mapping
        self.mag_1 += rosesim.Roman_zp_AB_Vega_parsec[self.band_1.upper()]
        self.mag_2 += rosesim.Roman_zp_AB_Vega_parsec[self.band_2.upper()]

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2
        
        # if self.verbose:
            # print(f"Successfully loaded isochrone: logAge={self.logage}, [Fe/H]={self.feh}")

    def separation(self, mag_1, mag_2):
        """
        Calculates the minimum separation between test points and the isochrone interpolation.
        Adapted from DES/ugali.

        Args:
            mag_1 (array-like): Magnitude of test points in the first band.
            mag_2 (array-like): Magnitude of test points in the second band.

        Returns:
            array-like: Minimum Euclidean distance in color-magnitude space.
        """
        iso_mag_1 = self.mag_1 + self.distance_modulus
        iso_mag_2 = self.mag_2 + self.distance_modulus

        def interp_iso(iso_mag_1, iso_mag_2, mag_1, mag_2):
            interp_1 = scipy.interpolate.interp1d(iso_mag_1, iso_mag_2, bounds_error=False)
            interp_2 = scipy.interpolate.interp1d(iso_mag_2, iso_mag_1, bounds_error=False)

            dy = interp_1(mag_1) - mag_2
            dx = interp_2(mag_2) - mag_1

            dmag_1 = np.fabs(dx * dy) / (dx**2 + dy**2) * dy
            dmag_2 = np.fabs(dx * dy) / (dx**2 + dy**2) * dx

            return dmag_1, dmag_2

        # Separate the various stellar evolution stages
        if np.issubdtype(self.stage.dtype, np.number):
            sel = (self.stage < self.hb_stage)
        else:
            sel = (self.stage != self.hb_stage)

        # First do the MS/RGB
        rgb_mag_1 = iso_mag_1[sel]
        rgb_mag_2 = iso_mag_2[sel]
        dmag_1, dmag_2 = interp_iso(rgb_mag_1, rgb_mag_2, mag_1, mag_2)

        # Then do the HB (if it exists)
        if not np.all(sel):
            hb_mag_1 = iso_mag_1[~sel]
            hb_mag_2 = iso_mag_2[~sel]

            hb_dmag_1, hb_dmag_2 = interp_iso(hb_mag_1, hb_mag_2, mag_1, mag_2)

            dmag_1 = np.nanmin([dmag_1, hb_dmag_1], axis=0)
            dmag_2 = np.nanmin([dmag_2, hb_dmag_2], axis=0)

        return np.sqrt(dmag_1**2 + dmag_2**2)

#------------------------------------------------------------------------------

def cut_isochrone_path(g, r, g_err, r_err, isochrone, mag_max, radius=0.1, return_all=False, verbose=True):
    """
    Creates a boolean mask identifying objects within a specific radius of the isochrone path.
    Modified by Will Cerny.

    Args:
        g, r (array-like): Stellar magnitudes.
        g_err, r_err (array-like): Magnitude errors.
        isochrone (Isochrone): The initialized Isochrone object.
        mag_max (float): Faint-end magnitude limit for the evaluation bins.
        radius (float): Selection radius in color-magnitude space.
        return_all (bool): If True, returns the mask and the evaluation bin arrays.
        verbose (bool): If True, prints status.

    Returns:
        array-like: Boolean mask of selected stars. (Or tuple of arrays if return_all=True).
    """
    if np.all(isochrone.stage == 'Main'):
        index_transition = len(isochrone.stage)
    else:
        index_transition = np.nonzero(isochrone.stage >= isochrone.hb_stage)[0][0] + 1

    mag_1_rgb = isochrone.mag_1[0:index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0:index_transition] + isochrone.distance_modulus

    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]

    # Cut one way..
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < np.sqrt(radius**2 + r_err**2 + g_err**2))

    # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < np.sqrt(radius**2 + r_err**2 + g_err**2))

    cut = np.logical_or(cut_1, cut_2)

    ## Cut for horizontal branch
    #mag_1_hb = isochrone.mag_1[isochrone.stage == isochrone.hb_stage][1:] + isochrone.distance_modulus
    #mag_2_hb = isochrone.mag_2[isochrone.stage == isochrone.hb_stage][1:] + isochrone.distance_modulus
    #f_isochrone = scipy.interpolate.interp1d(mag_2_hb, mag_1_hb - mag_2_hb, bounds_error=False, fill_value = 999.)
    #color_diff = np.fabs((g - r) - f_isochrone(r))
    #cut_4 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
    #f_isochrone = scipy.interpolate.interp1d(mag_1_hb, mag_1_hb - mag_2_hb, bounds_error=False, fill_value = 999.)
    #color_diff = np.fabs((g - r) - f_isochrone(g))
    #cut_3 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
    #
    #cut_hb = np.logical_or(cut_3, cut_4)
    #cut = np.logical_or(cut, cut_hb)

    mag_bins = np.arange(17., mag_max + 0.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = np.tile(0., len(mag_centers))
    
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = np.median(np.sqrt(radius**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

#------------------------------------------------------------------------------

def drawIsochrone(isochrone, ax=None, **kwargs):
    """
    Plots the theoretical isochrone track on a given Matplotlib axis, 
    handling discontinuities in the color-magnitude distribution. 
    From ugali.utils.plotting.

    Args:
        isochrone (Isochrone): The initialized Isochrone object to plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to plot on.
        **kwargs: Additional plotting arguments (e.g., 'cookie=True' for broad path).
    """
    if ax is None:
        ax = plt.gca()
        
    if kwargs.pop('cookie', None):
        # Broad cookie cutter
        defaults = dict(alpha=0.5, color='0.5', zorder=0, linewidth=15, linestyle='-')
    else:
        # Thin lines
        defaults = dict(color='k', linestyle='-')
        
    # Combine default plotting styles with any user-provided kwargs
    kwargs = {**defaults, **kwargs}

    isos = isochrone.isochrones if hasattr(isochrone, 'isochrones') else [isochrone]
    
    for iso in isos:
        iso = copy.deepcopy(iso)
        mag = iso.mag_1 + iso.distance_modulus
        color = iso.mag_1 - iso.mag_2

        # Find discontinuities in the color magnitude distributions
        dmag = np.fabs(mag[1:] - mag[:-1])
        dcolor = np.fabs(color[1:] - color[:-1])
        idx = np.where((dmag > 1.0) | (dcolor > 0.25))[0]
        
        # +1 to map from difference array to original array
        mags = np.split(mag, idx + 1)
        colors = np.split(color, idx + 1)

        for i, (c, m) in enumerate(zip(colors, mags)):
            if i > 0:
                kwargs['label'] = None
            ax.plot(c, m, **kwargs)
            
    return ax