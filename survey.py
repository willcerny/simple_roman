#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Sidney Mau, Will Cerny, Jiaxuan Li"

import glob
import os, sys
import pandas as pd
import numpy as np
import healpy as hp
import scipy.stats
import scipy.interpolate
import scipy.ndimage
import astropy.units as u
from . import projector

#-------------------------------------------------------------------------------

# From https://github.com/DarkEnergySurvey/ugali/blob/master/ugali/utils/healpix.py

def superpixel(subpix, nside_subpix, nside_superpix):
    """
    Return the indices of the super-pixels which contain each of the sub-pixels.
    """
    if nside_subpix==nside_superpix: return subpix
    theta, phi = hp.pix2ang(nside_subpix, subpix)
    return(hp.ang2pix(nside_superpix, theta, phi))


def subpixel(superpix, nside_superpix, nside_subpix):
    """
    Return the indices of sub-pixels (resolution nside_subpix) within
    the super-pixel with (resolution nside_superpix).

    ADW: It would be better to convert to next and do this explicitly
    """
    if nside_superpix==nside_subpix: return superpix
    vec = hp.pix2vec(nside_superpix, superpix)
    radius = np.degrees(2. * hp.max_pixrad(nside_superpix))
    subpix = hp.query_disc(nside_subpix, vec, np.radians(radius))
    pix_for_subpix = superpixel(subpix,nside_subpix,nside_superpix)
    # Might be able to speed up array indexing...
    return(subpix[pix_for_subpix == superpix])

#-------------------------------------------------------------------------------

class Survey():
    """
    Class to handle survey-specific parameters.
    """
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

        self.mag_1 = self.catalog['mag'].format(self.band_1)
        self.mag_2 = self.catalog['mag'].format(self.band_2)
        self.mag_dered_1 = self.catalog['mag_dered'].format(self.band_1)
        self.mag_dered_2 = self.catalog['mag_dered'].format(self.band_2)
        self.mag_err_1 = self.catalog['mag_err'].format(self.band_1)
        self.mag_err_2 = self.catalog['mag_err'].format(self.band_2)

        self.load_fracdet

    @property
    def load_fracdet(self):
        """
        Load-in the fracdet map if it exists.
        """
        #if self.survey['fracdet']:
        #    print('Reading fracdet map {} ...'.format(self.survey['fracdet']))
        #    fracdet = ugali.utils.healpix.read_map(self.survey['fracdet'])
        #else:
        #    print('No fracdet map specified ...')
        #    fracdet = None
        ##return(fracdet)
        #self.fracdet = fracdet
        # SM: Commenting this out until I have a fracdet map to debug with
        self.fracdet = None

#-------------------------------------------------------------------------------

class Region():
    """
    Class to handle regions.
    """
    def __init__(self, survey, ra, dec):
        self.survey = survey
        self.nside = self.survey.catalog['nside']
        self.fracdet = self.survey.fracdet

        self.ra = ra
        self.dec = dec
        self.proj = projector.Projector(self.ra, self.dec)
        self.pix_center = hp.ang2pix(self.nside, self.ra, self.dec, lonlat=True)

    def _split_catalog(self):
        """
        Internal helper to standardize how we split stars from galaxies.
        """
        self.data = self.all_data[self.all_data['star_flag']]
        self.galaxies = self.all_data[~self.all_data['star_flag']]

    def load_data_roman(self, data_dir):
        """
        Loads the catalog from disk. This serves as the 'base' catalog, 
        which could be purely background, or could be a real science field.
        """
        data = pd.read_csv(data_dir)
        processed_data = data[data['quality_flag']]

        # 1. Store as the immutable pristine base catalog
        self.base_data = processed_data
        
        # 2. Initialize the active data to this base catalog
        self.all_data = self.base_data.copy()
        
        # 3. Finalize the arrays
        self._split_catalog()

    def reset_data(self):
        """
        Wipes the active catalog and resets it to the pristine base data.
        Perfect for batch injection-recovery tests.
        """
        self.all_data = self.base_data.copy()
        self._split_catalog()

    def characteristic_density(self, iso_sel, delta_x=0.01, smoothing=2./60., bin_extent=8., delta_x_coverage=0.1, bin_extent_coverage=5., verbose=True):
        """
        Computes the global characteristic density of the selected region.
        
        This method projects the isochrone-filtered stars onto a 2D spatial grid 
        and calculates the median background density. It incorporates fractional 
        detection (fracdet) maps to correct for survey coverage gaps if available.

        Args:
            iso_sel (array-like): Boolean mask array for stars passing the isochrone filter.
            delta_x (float): Spatial bin size for the fine-grained histogram in degrees.
            smoothing (float): Gaussian smoothing kernel width in degrees.
            bin_extent (float): Maximum spatial extent for the fine histogram in degrees.
            delta_x_coverage (float): Spatial bin size for the coarse background histogram in degrees.
            bin_extent_coverage (float): Maximum spatial extent for the coarse histogram in degrees.
            verbose (bool): If True, prints diagnostic output to the console.

        Returns:
            float: The global characteristic density in units of stars per square degree.
        """
        x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']][iso_sel], 
                                       self.data[self.survey.catalog['basis_2']][iso_sel])
        
        area = delta_x**2
        bins = np.arange(-bin_extent, bin_extent + 1.e-10, delta_x)
        centers = 0.5 * (bins[0: -1] + bins[1:])
        yy, xx = np.meshgrid(centers, centers)
    
        h = np.histogram2d(x, y, bins=[bins, bins])[0]
        h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)
    
        # Coarse grid for background coverage estimation
        area_coverage = (delta_x_coverage)**2
        bins_coverage = np.arange(-bin_extent_coverage, bin_extent_coverage + 1.e-10, delta_x_coverage)
        h_coverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
        h_goodcoverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
    
        n_goodcoverage = h_coverage[h_goodcoverage > 0].flatten()
        characteristic_density = np.median(n_goodcoverage) / area_coverage # per square degree
        
        if verbose:
            print('Characteristic density = {:0.1f} deg^-2'.format(characteristic_density))
    
        # Fractional detection (fracdet) correction
        if self.fracdet is not None:
            fracdet_zero = np.tile(0., len(self.fracdet))
            cut = (self.fracdet != hp.UNSEEN)
            fracdet_zero[cut] = self.fracdet[cut]
    
            nside_fracdet = hp.npix2nside(len(self.fracdet))
            
            subpix_region_array = []
            for pix in np.unique(hp.ang2pix(self.nside,
                                            self.data[self.survey.catalog['basis_1']][iso_sel],
                                            self.data[self.survey.catalog['basis_2']][iso_sel],
                                            lonlat=True)):
                subpix_region_array.append(subpixel(self.pix_center, self.nside, nside_fracdet))
            subpix_region_array = np.concatenate(subpix_region_array)
    
            cut = (self.fracdet[subpix_region_array] != hp.UNSEEN)
            mean_fracdet = np.mean(self.fracdet[subpix_region_array[cut]])
    
            characteristic_density_raw = 1. * characteristic_density
            characteristic_density /= mean_fracdet 
            
            if verbose:
                print('Characteristic density (fracdet corrected) = {:0.1f} deg^-2'.format(characteristic_density))
    
        return(characteristic_density)
    
    def characteristic_density_local(self, iso_sel, x_peak, y_peak, angsep_peak, annulus=[0.3, 0.5], verbose=True):
        """
        Computes the local characteristic density (background) around a specific candidate peak. 
        
        Evaluates the stellar density within an annulus surrounding the candidate. 
        If local coverage is spotty or azimuthal distribution is too uneven (e.g., near an edge), 
        it safely falls back to the global characteristic density.

        Args:
            iso_sel (array-like): Boolean mask array for stars passing the isochrone filter.
            x_peak (float): X-coordinate of the candidate peak in projected image space.
            y_peak (float): Y-coordinate of the candidate peak in projected image space.
            angsep_peak (array-like): Angular separation of all stars from the candidate peak.
            annulus (list of floats): The [inner, outer] radii of the background annulus in degrees.
            verbose (bool): If True, prints diagnostic output to the console.

        Returns:
            float: The localized background density in units of stars per square degree.
        """
        characteristic_density = self.density
        x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']][iso_sel], 
                                       self.data[self.survey.catalog['basis_2']][iso_sel])
        # If fracdet map is available, use that information to either compute local density,
        # or in regions of spotty coverage, use the typical density of the region
        if self.fracdet is not None:
            fracdet_zero = np.tile(0., len(self.fracdet))
            cut = (self.fracdet != hp.UNSEEN)
            fracdet_zero[cut] = self.fracdet[cut]
    
            nside_fracdet = hp.npix2nside(len(self.fracdet))
            
            subpix_region_array = []
            for pix in np.unique(hp.ang2pix(self.nside,
                                            self.data[self.survey.catalog['basis_1']][iso_sel],
                                            self.data[self.survey.catalog['basis_2']][iso_sel],
                                            lonlat=True)):
                subpix_region_array.append(subpixel(self.pix_center, self.nside, nside_fracdet))
            subpix_region_array = np.concatenate(subpix_region_array)
    
            cut = (self.fracdet[subpix_region_array] != hp.UNSEEN)
            mean_fracdet = np.mean(self.fracdet[subpix_region_array[cut]])
    
            subpix_region_array = subpix_region_array[self.fracdet[subpix_region_array] > 0.99]
            
            # NOTE: cut_magnitude_threshold variable must be defined or passed into the class
            subpix = hp.ang2pix(nside_fracdet, 
                                self.data[self.survey.catalog['basis_1']][cut_magnitude_threshold][iso_sel], 
                                self.data[self.survey.catalog['basis_2']][cut_magnitude_threshold][iso_sel],
                                lonlat=True)
    
            ra_peak, dec_peak = self.proj.imageToSphere(x_peak, y_peak)
            subpix_all = hp.query_disc(nside_fracdet, hp.ang2vec(ra_peak, dec_peak, lonlat=True), np.radians(annulus[1]))
            subpix_inner = hp.query_disc(nside_fracdet, hp.ang2vec(ra_peak, dec_peak, lonlat=True), np.radians(annulus[0]))
            subpix_annulus = subpix_all[~np.in1d(subpix_all, subpix_inner)]
            mean_fracdet = np.mean(fracdet_zero[subpix_annulus])
            
            if verbose:
                print('mean_fracdet {}'.format(mean_fracdet))
                
            if mean_fracdet < 0.5:
                characteristic_density_local = characteristic_density
                if verbose:
                    print('characteristic_density_local baseline {}'.format(characteristic_density_local))
            else:
                subpix_annulus_region = np.intersect1d(subpix_region_array, subpix_annulus)
                if verbose:
                    print('{} percent pixels with complete coverage'.format(float(len(subpix_annulus_region)) / len(subpix_annulus)))
                
                if (float(len(subpix_annulus_region)) / len(subpix_annulus)) < 0.25:
                    characteristic_density_local = characteristic_density
                    if verbose:
                        print('characteristic_density_local spotty {}'.format(characteristic_density_local))
                else:
                    characteristic_density_local = float(np.sum(np.in1d(subpix, subpix_annulus_region))) / (hp.nside2pixarea(nside_fracdet, degrees=True) * len(subpix_annulus_region))
                    if verbose:
                        print('characteristic_density_local cleaned up {}'.format(characteristic_density_local))
        else:
            area_field = np.pi * (annulus[1]**2 - annulus[0]**2)
            n_field = np.sum((angsep_peak > annulus[0]) & (angsep_peak < annulus[1]))
            characteristic_density_local = n_field / area_field
    
            cut_annulus = (angsep_peak > annulus[0]) & (angsep_peak < annulus[1]) 
            phi = np.degrees(np.arctan2(y[cut_annulus] - y_peak, x[cut_annulus] - x_peak)) 
            h = np.histogram(phi, bins=np.linspace(-180., 180., 13))[0]
            
            if np.sum(h > 0) < 10 or np.sum(h > 0.5 * np.median(h)) < 10:
                characteristic_density_local = characteristic_density
    
        if verbose:
            print('\tCharacteristic density local = {:0.1f} deg^-2 = {:0.3f} arcmin^-2'.format(
                characteristic_density_local, characteristic_density_local / 60.**2))
    
        return characteristic_density_local

    def find_peaks(self, iso_sel, delta_x=0.01, smoothing=2./60., bin_extent=8., verbose=True):
        """
        Projects the filtered stars into a 2D spatial grid, smooths the field, 
        and identifies isolated overdensity peaks. 

        The algorithm iteratively adjusts a threshold density factor to isolate 
        distinct regions (fewer than 10 disconnected islands) to define peak candidates.

        Args:
            iso_sel (array-like): Boolean mask array for stars passing the isochrone filter.
            delta_x (float): Spatial bin size for the histogram in degrees.
            smoothing (float): Gaussian smoothing kernel width in degrees.
            bin_extent (float): Maximum spatial extent for the histogram in degrees.
            verbose (bool): If True, prints diagnostic output to the console.

        Returns:
            tuple: Three lists containing the x-coordinates of the peaks, the 
                   y-coordinates of the peaks, and an array of angular separations 
                   for all filtered stars relative to each peak.
        """
        characteristic_density = self.density
        x, y = self.proj.sphereToImage(self.data[self.survey.catalog['basis_1']][iso_sel], 
                                       self.data[self.survey.catalog['basis_2']][iso_sel])
        
        area = delta_x**2
        bins = np.arange(-bin_extent, bin_extent + 1.e-10, delta_x)
        centers = 0.5 * (bins[0: -1] + bins[1:])
        yy, xx = np.meshgrid(centers, centers)
    
        h = np.histogram2d(x, y, bins=[bins, bins])[0]
        h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)
    
        # Iteratively reduce the contrast against the background 
        # until there are fewer than 10 disconnected peaks
        factor_array = np.arange(1., 5., 0.05)
        rara, decdec = self.proj.imageToSphere(xx.flatten(), yy.flatten())
        cutcut = (hp.ang2pix(self.nside, rara, decdec, lonlat=True) == self.pix_center).reshape(xx.shape)
        threshold_density = 5 * characteristic_density * area
        
        for factor in factor_array:
            h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > (area * characteristic_density * factor))
            if n_region < 10:
                threshold_density = area * characteristic_density * factor
                break
    
        h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > threshold_density)
    
        x_peak_array = []
        y_peak_array = []
        angsep_peak_array = []
    
        for index in range(1, n_region + 1):
            index_peak = np.ravel_multi_index(scipy.ndimage.maximum_position(input=h_g, labels=h_region, index=index), h_g.shape)
            x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]
            
            angsep_peak = np.sqrt((x - x_peak)**2 + (y - y_peak)**2)
    
            x_peak_array.append(x_peak)
            y_peak_array.append(y_peak)
            angsep_peak_array.append(angsep_peak)
            
        if verbose:
            print('Found {} peaks within the selected region'.format(n_region))
        
        return x_peak_array, y_peak_array, angsep_peak_array

    def fit_aperture(self, iso_sel, x_peak, y_peak, angsep_peak, aperture_min=0.01, aperture_max=0.3, aperture_step=0.01, verbose=True):
        """
        Determines the optimal size of the candidate dwarf galaxy by growing a circular 
        aperture and maximizing the Poisson significance against the background. 

        Args:
            iso_sel (array-like): Boolean mask array for stars passing the isochrone filter.
            x_peak (float): X-coordinate of the candidate peak in projected image space.
            y_peak (float): Y-coordinate of the candidate peak in projected image space.
            angsep_peak (array-like): Angular separation of all stars from the candidate peak.
            aperture_min (float): Minimum aperture radius to test in degrees.
            aperture_max (float): Maximum aperture radius to test in degrees.
            aperture_step (float): Step size for testing aperture radii in degrees.
            verbose (bool): If True, prints diagnostic output to the console.

        Returns:
            tuple: Lists containing the optimal ra/dec, radius, max significance, 
                   observed star counts, half-radius star counts, and expected 
                   background model counts.
        """
        characteristic_density_local = self.characteristic_density_local(iso_sel, x_peak, y_peak, angsep_peak, verbose=verbose)
    
        ra_peak_array = []
        dec_peak_array = []
        r_peak_array = []
        sig_peak_array = []
        n_obs_peak_array = []
        n_obs_half_peak_array = []
        n_model_peak_array = []
    
        size_array = np.arange(aperture_min, aperture_max, aperture_step)
        
        # Vectorized array math instead of a slow for-loop block
        n_obs_array = np.array([np.sum(angsep_peak < size) for size in size_array])
        n_model_array = characteristic_density_local * (np.pi * size_array**2)
        sig_array = np.clip(scipy.stats.norm.isf(scipy.stats.poisson.sf(n_obs_array, n_model_array)), 0., 37.5)
    
        ra_peak, dec_peak = self.proj.imageToSphere(x_peak, y_peak)
    
        # Find the aperture size that yields the highest significance
        index_peak = np.argmax(sig_array)
        r_peak = size_array[index_peak]
        
        n_obs_peak = n_obs_array[index_peak]
        n_model_peak = n_model_array[index_peak]
        n_obs_half_peak = np.sum(angsep_peak < (0.5 * r_peak))
    
        if verbose:
            print('\t\t Candidate: x_peak: {:8.3f}, y_peak: {:8.3f}, r_peak: {:8.3f}, sig: {:8.3f}, ra_peak: {:8.3f}, dec_peak: {:8.3f}'.format(
                x_peak, y_peak, r_peak, np.max(sig_array), ra_peak, dec_peak))
            
        # Append the best-fit parameters to the tracking arrays
        ra_peak_array.append(ra_peak)
        dec_peak_array.append(dec_peak)
        r_peak_array.append(r_peak)
        sig_peak_array.append(sig_array[index_peak])
        n_obs_peak_array.append(n_obs_peak)
        n_obs_half_peak_array.append(n_obs_half_peak)
        n_model_peak_array.append(n_model_peak)
    
        return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, n_obs_peak_array, n_obs_half_peak_array, n_model_peak_array