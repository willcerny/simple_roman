import numpy as np
import yaml
import os
import healpy as hp
import pandas as pd
from . import survey
from . import isochrone
from . import projector
from . import coordinate_tools
from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage
from matplotlib.colors import ListedColormap

cmap_gray = plt.get_cmap('gray')
cmap_gray_mask = ListedColormap(cmap_gray(np.linspace(0.1, 1.0, 100)))
cmap_gray_mask.set_bad('white')

class Search:
    """
    A pipeline for identifying resolved stellar overdensities using spatial and photometric matched-filters.
    """
    def __init__(self, config_file='config.yaml', band1='F106', band2='F158', ra=None, dec=None, nside=None, ipix=None):
        """
        Initializes the Search pipeline, parsing target coordinates, loading the 
        baseline background catalog, and pre-computing theoretical isochrones.
        """
        # Coordinate parsing
        if ra is not None and dec is not None:
            if nside is not None or ipix is not None:
                raise ValueError('Please specify either (ra, dec) or (nside, ipix).')
        elif nside is not None and ipix is not None:
            ra, dec = hp.pix2ang(nside=nside, ipix=ipix, lonlat=True)
        else:
            raise ValueError('Please specify either (ra, dec) or (nside, ipix).')

        self.config_file = config_file
        self.band1 = band1
        self.band2 = band2

        # 1. Initial config load (mutates self.cfg, self.verbose, and self.surveyobj)
        self.reload_config(reload_isochrones=False, _initial_load=True)
        
        self.region = survey.Region(self.surveyobj, ra, dec)

        if self.verbose:
            print(f"Search coordinates: (RA, Dec) = ({self.region.ra:.2f}, {self.region.dec:.2f})")
            print(f"Search healpixel: {self.region.pix_center} (nside = {self.region.nside})")

        # 2. Load the background catalog ONLY ONCE
        print('Loading background catalog...')
        self.region.load_data_roman(self.cfg['catalog']['bkg_file'])

        if 'halo_star_file' in self.cfg['catalog']:
            print('Loading halo star catalog...')
            self.region.load_halo_stars(self.cfg['catalog']['halo_star_file'])
        
        if self.verbose:
            print(f"Loaded {len(self.region.data)} point sources and {len(self.region.galaxies)} galaxies in baseline background catalog.")

        # 3. Pre-compute Isochrones ONLY ONCE
        self._compute_isochrones()

    def reload_config(self, config_file=None, reload_isochrones=False, _initial_load=False):
        """
        Dynamically reloads the YAML configuration file and updates survey parameters 
        in-place without needing to re-initialize the entire Search object or reload data.

        Args:
            config_file (str, optional): A new path to a YAML config. If None, uses the 
                                         previously stored path.
            reload_isochrones (bool): If True, forces a re-computation of the isochrone 
                                      grids. Set to False to skip this expensive step.
            _initial_load (bool): Internal flag to suppress print statements during __init__.
        """
        if config_file is not None:
            self.config_file = config_file
            
        with open(self.config_file, 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        
        self.verbose = self.cfg.get('verbose', True)
        
        # Update the core survey object with the new config parameters
        self.surveyobj = survey.Survey(self.cfg)
        
        # If the region already exists in memory, we must safely sync its survey reference
        if hasattr(self, 'region'):
            self.region.survey = self.surveyobj
            self.region.nside = self.surveyobj.catalog['nside']
            self.region.fracdet = getattr(self.surveyobj, 'fracdet', None)
            
        if self.verbose and not _initial_load:
            print(f"Configuration dynamically reloaded from '{self.config_file}'")
            
        if reload_isochrones:
            self._compute_isochrones()

    def _compute_isochrones(self):
        """
        Internal helper method to pre-compute the theoretical isochrone grids.
        """
        if self.verbose:
            print('Pre-computing isochrones given the distance modulus range...')
            
        self.distance_modulus_search_array = np.arange(
            self.cfg['search']['distance_modulus_min'], 
            self.cfg['search']['distance_modulus_max'], 
            self.cfg['search']['distance_modulus_step']
        )
        self.iso_search_array = [
            isochrone.Isochrone(logage=self.cfg['isochrone']['logage'], feh=self.cfg['isochrone']['feh'],
                                band_1=self.band1, band_2=self.band2, 
                                distance_modulus=dm,
                                filename=self.cfg['isochrone']['filename'], 
                                verbose=self.verbose) 
            for dm in self.distance_modulus_search_array
        ]

    def _search_by_distance(self, distance_modulus, iso_sel):
        """
        Internal method that runs the spatial matched-filter algorithm for a 
        single distance slice.

        Calculates the local background density, identifies spatial peaks, and 
        fits an aperture to determine the Poisson significance of the overdensity.

        Args:
            distance_modulus (float): The distance modulus currently being tested.
            iso_sel (array-like): Boolean mask of stars that pass the isochrone filter 
                for this specific distance.

        Returns:
            tuple: Eight flattened arrays containing the parameters for all recovered 
                candidates at this distance slice (RA, Dec, optimal radius, significance, 
                distance modulus, observed counts, half-radius counts, expected background).
        """
        if len(self.region.data[iso_sel]) == 0:
            return [], [], [], [], [], [], [], []

        cfg_s = self.cfg['search']
        
        # Calculate local field density
        self.region.density = self.region.characteristic_density(
            iso_sel, 
            delta_x=cfg_s['delta_x'] / 3600, 
            smoothing=cfg_s['smoothing'] / 3600, 
            bin_extent=cfg_s['bin_extent'] / 3600, 
            bin_extent_coverage=cfg_s['bin_extent_coverage'] / 3600, 
            delta_x_coverage=cfg_s['delta_x_coverage'] / 3600,
            verbose=self.verbose
        )

        # Find spatial peaks
        x_peaks, y_peaks, angsep_peaks = self.region.find_peaks(
            iso_sel, 
            delta_x=cfg_s['delta_x'] / 3600, 
            smoothing=cfg_s['smoothing'] / 3600, 
            bin_extent=cfg_s['bin_extent'] / 3600,
            verbose=self.verbose
        )

        ra_arr, dec_arr, r_arr, sig_arr = [], [], [], []
        n_obs_arr, n_obs_half_arr, n_model_arr = [], [], []

        for x_peak, y_peak, angsep_peak in zip(x_peaks, y_peaks, angsep_peaks):
            ra, dec, r, sig, n_obs, n_obs_half, n_model = self.region.fit_aperture(
                iso_sel, x_peak, y_peak, angsep_peak, 
                aperture_min=cfg_s['aperture_min'] / 3600, 
                aperture_max=cfg_s['aperture_max'] / 3600, 
                aperture_step=cfg_s['aperture_step'] / 3600,
                verbose=self.verbose
            )

            ra_arr.append(ra)
            dec_arr.append(dec)
            r_arr.append(r)
            sig_arr.append(sig)
            n_obs_arr.append(n_obs)
            n_obs_half_arr.append(n_obs_half)
            n_model_arr.append(n_model)

        dist_mod_arr = distance_modulus * np.ones(len(np.concatenate(ra_arr))) if ra_arr else []

        return (np.concatenate(ra_arr) if ra_arr else [],
                np.concatenate(dec_arr) if dec_arr else [],
                np.concatenate(r_arr) if r_arr else [],
                np.concatenate(sig_arr) if sig_arr else [],
                dist_mod_arr,
                np.concatenate(n_obs_arr) if n_obs_arr else [],
                np.concatenate(n_obs_half_arr) if n_obs_half_arr else [],
                np.concatenate(n_model_arr) if n_model_arr else [])

    def run_injection_trial(self, mock_ra, mock_dec, mock_distance, mock_log_m_star, seed=1, mc_source_id=1):
        """
        Executes a complete injection-recovery trial by resetting the catalog, 
        injecting a synthetic mock galaxy, and searching the updated data.

        This method automatically sorts recovered candidates by their statistical 
        significance and prunes overlapping peaks.

        Args:
            mock_ra (float): Right Ascension of the injected mock galaxy in degrees.
            mock_dec (float): Declination of the injected mock galaxy in degrees.
            mock_distance (float): Distance to the mock galaxy in Mpc.
            mock_log_m_star (float): Log10 of the mock galaxy's stellar mass.
            mc_source_id (int, optional): Unique trial identifier. Defaults to 1.

        Returns:
            dict or None: A dictionary mapping column names to 1D arrays of the 
                surviving candidates' properties. Returns None if no significant 
                peaks are detected.
        """
        # 1. Reset data to the baseline background (Consistency with Region class)
        self.region.reset_data()

        # 2. Inject the mock galaxy
        from . import mock
        gal = mock.MockGalaxy(mock_ra, mock_dec, mock_distance, mock_log_m_star, seed=seed)
        gal.make_star_catalog()
        print('reff', gal.reff)

        # Note: If self.base_data is a Pandas DataFrame, you should use pd.concat. 
        # If setup_dolphot_cat returns a numpy recarray, keep np.concatenate.
        if isinstance(self.region.base_data, pd.DataFrame):
            self.region.all_data = pd.concat([self.region.base_data, gal.star_cat], ignore_index=True)
        else:
            self.region.all_data = np.concatenate([self.region.base_data, gal.star_cat])
        # Re-split the newly combined data
        self.region._split_catalog()

        if len(self.region.data) == 0:
            return None

        # if 'MC_SOURCE_ID' in self.region.data.columns:
        #     true_flag = self.region.data['MC_SOURCE_ID'] == 1
        #     print('Number of stars above F106<27.4 and F158<27.4:', np.sum((mag_1[true_flag] < 27.4) & (mag_2[true_flag] < 27.4)))

        # 3. Dynamically compute the isochrone selection masks for the new combined data
        iso_selection_array = [
            isochrone.cut_isochrone_path(
                self.region.data[self.cfg['catalog']['mag'].format(self.band1)],
                self.region.data[self.cfg['catalog']['mag'].format(self.band2)],
                self.region.data[self.cfg['catalog']['mag_err'].format(self.band1)],
                self.region.data[self.cfg['catalog']['mag_err'].format(self.band2)],
                iso,
                mag_max=self.cfg['catalog']['mag_max'],
                radius=self.cfg['isochrone']['radius'],
                verbose=self.verbose
            )
            for iso in self.iso_search_array
        ]
        self.iso_selection_array = iso_selection_array

        # 4. Search over all distance moduli
        results = [self._search_by_distance(dm, iso_sel) 
                   for dm, iso_sel in zip(self.distance_modulus_search_array, iso_selection_array)]
        
        ra_peak, dec_peak, r_peak, sig_peak, dist_mod, n_obs, n_obs_half, n_model = map(np.concatenate, zip(*results))
        mc_id_arr = np.ones(len(dist_mod)) * mc_source_id

        if len(sig_peak) == 0:
            return None

        # 5. Sort and consolidate overlapping peaks
        idx_sort = np.argsort(sig_peak)[::-1]
        ra_peak, dec_peak, r_peak, sig_peak = ra_peak[idx_sort], dec_peak[idx_sort], r_peak[idx_sort], sig_peak[idx_sort]
        dist_mod, n_obs, n_obs_half, n_model, mc_id_arr = dist_mod[idx_sort], n_obs[idx_sort], n_obs_half[idx_sort], n_model[idx_sort], mc_id_arr[idx_sort]

        for ii in range(len(sig_peak)):
            if sig_peak[ii] < 0:
                continue
            sep = coordinate_tools.angsep(ra_peak[ii], dec_peak[ii], ra_peak, dec_peak)
            sig_peak[(sep < r_peak[ii]) & (np.arange(len(sig_peak)) > ii)] = -1.

        keep = sig_peak > 0.
        
        return {
            'RA': ra_peak[keep],
            'DEC': dec_peak[keep],
            'R': r_peak[keep],
            'SIG': sig_peak[keep],
            'MODULUS': dist_mod[keep],
            'N_OBS': n_obs[keep],
            'N_OBS_HALF': n_obs_half[keep],
            'N_MODEL': n_model[keep],
            'MC_SOURCE_ID': mc_id_arr[keep]
        }

    def write_output(self, results_dict, outfile):
        """
        Appends the recovered candidate parameters to a CSV file.

        Args:
            results_dict (dict): The dictionary of candidate arrays returned 
                by `run_injection_trial`.
            outfile (str): Name of the output CSV file to write or append to.
        """
        if results_dict is None or len(results_dict['SIG']) == 0:
            return
            
        data = [tuple(row) for row in np.stack([
            results_dict['SIG'], results_dict['RA'], results_dict['DEC'], 
            results_dict['MODULUS'], results_dict['R'], results_dict['N_OBS'], 
            results_dict['N_OBS_HALF'], results_dict['N_MODEL'], results_dict['MC_SOURCE_ID']
        ], axis=-1)]
        
        arr = np.array(data, dtype=[('SIG', float), ('RA', float), ('DEC', float), ('MODULUS', float), 
                                    ('R', float), ('N_OBS', float), ('N_OBS_HALF', float), 
                                    ('N_MODEL', float), ('MC_SOURCE_ID', int)])
        
        arr = Table(arr)
        arr.write(os.path.join(self.cfg.get('results_dir', './'), outfile), format='csv', overwrite=True)

    def print_results(self, results_dict, mock_id):
        """
        Outputs a human-readable summary of the recovered candidates to the terminal.
        This output is automatically suppressed if the configuration `verbose` flag is False.

        Args:
            results_dict (dict): The dictionary of candidate arrays returned 
                by `run_injection_trial`.
            mock_id (int): The unique identifier for the current injection trial.
        """
        print(f"\n--- Results for Mock ID {mock_id} ---")
        
        if results_dict is None or len(results_dict['SIG']) == 0:
            print("  -> No significant peaks recovered.")
            return

        for ii in range(len(results_dict['SIG'])):
            sig = results_dict['SIG'][ii]
            ra = results_dict['RA'][ii]
            dec = results_dict['DEC'][ii]
            r_arcsec = results_dict['R'][ii] * 3600.0
            dist_mod = results_dict['MODULUS'][ii]
            dist_kpc = coordinate_tools.distanceModulusToDistance(dist_mod)
            mc_id = results_dict['MC_SOURCE_ID'][ii]

            print(f"  -> Recovered: {sig:0.2f} sigma | "
                  f"(RA, Dec) = ({ra:0.4f}, {dec:0.4f}) | "
                  f"r = {r_arcsec:0.2f} arcsec | "
                  f"d = {dist_kpc:0.1f} kpc (mu = {dist_mod:0.2f} mag) | "
                  f"MC_ID: {mc_id}")

    def plot_candidate(self, ra, dec, distance_modulus, r_peak, sig, outfile='candidate_plot.png'):
        """
        Generates and saves a 3-panel validation plot for visual vetting of a 
        candidate using in-memory data. 

        The panels include:
        1. Smoothed spatial map of stars surviving the isochrone filter.
        2. Smoothed spatial map of background galaxies (for false-positive checking).
        3. Background-subtracted Hess (color-magnitude) diagram with theoretical 
           isochrone overlaid.

        Args:
            ra (float): Right Ascension of the candidate center.
            dec (float): Declination of the candidate center.
            distance_modulus (float): Best-fit distance modulus for the candidate.
            r_peak (float): Optimal candidate radius in degrees.
            sig (float): Poisson significance of the detection.
            outfile (str, optional): Filename for the saved plot. Set to None to skip saving. 
                Defaults to 'candidate_plot.png'.
        """
        idx = np.argmin(np.abs(self.distance_modulus_search_array - distance_modulus))
        iso = self.iso_search_array[idx]
        
        mag_1 = self.region.data[self.cfg['catalog']['mag'].format(self.band1)]
        mag_2 = self.region.data[self.cfg['catalog']['mag'].format(self.band2)]
        mag_err_1 = self.region.data[self.cfg['catalog']['mag_err'].format(self.band1)]
        mag_err_2 = self.region.data[self.cfg['catalog']['mag_err'].format(self.band2)]

        iso_filter = isochrone.cut_isochrone_path(
            mag_1, mag_2, mag_err_1, mag_err_2, iso,
            mag_max=self.cfg['catalog']['mag_max'], 
            radius=self.cfg['isochrone']['radius'],
            verbose=self.verbose
        )

        bound = max(r_peak * 10.0, 3 / 60.0) 
        delta_x = self.cfg['search']['delta_x'] / 3600 # deg
        bin_extent = self.cfg['search']['bin_extent'] / 3600 # deg
        smoothing = self.cfg['search']['smoothing'] / 3600 # deg
        bins = np.arange(-bin_extent, bin_extent + 1.e-10, delta_x)

        proj = projector.Projector(ra, dec)
        b1_key = self.cfg['catalog']['basis_1']
        b2_key = self.cfg['catalog']['basis_2']

        x_stars, y_stars = proj.sphereToImage(self.region.data[b1_key], self.region.data[b2_key])
        if hasattr(self.region, 'galaxies') and len(self.region.galaxies) > 0:
            x_gals, y_gals = proj.sphereToImage(self.region.galaxies[b1_key], self.region.galaxies[b2_key])
        else:
            x_gals, y_gals = np.array([]), np.array([])

        color = mag_1 - mag_2
        r0 = 3.0 * r_peak
        r1 = 5.0 * r_peak
        r2 = np.sqrt(r0**2 + r1**2)
        angsep = coordinate_tools.angsep(ra, dec, self.region.data[b1_key], self.region.data[b2_key])
        
        inner = (angsep < r0)
        outer = ((angsep > r1) & (angsep < r2))

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        fig.subplots_adjust(wspace=0.5)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)

        # PANEL 1: Stellar histogram
        ax = axs[0]
        signal = np.histogram2d(x_stars[iso_filter], y_stars[iso_filter], bins=[bins, bins])[0]
        sigma_smooth = smoothing * (0.25 * np.arctan(0.25 * r0 * 60. - 1.5) + 1.3)
        
        if self.verbose:
            print('smoothing size:', sigma_smooth)
            
        convolution = scipy.ndimage.filters.gaussian_filter(signal, sigma_smooth/delta_x).T
        pc = ax.pcolormesh(bins, bins, convolution, cmap='Greys', rasterized=True)
        # how to get vmin and vmax from pc?
        # vmin = pc.get_clim()[0]
        # vmax = pc.get_clim()[1]

        ax.text(0.07, 0.92, 'Stars (Isochrone Filtered)', transform=ax.transAxes, verticalalignment='top', bbox=props)
        ax.set_xlim(bound, -bound)
        ax.set_ylim(-bound, bound)
        ax.set_xlabel(r'$\Delta {\rm RA}$ (deg)')
        ax.set_ylabel(r'$\Delta {\rm Dec}$ (deg)')
        # fig.colorbar(pc, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0))

        circle = plt.Circle((0, 0), r0, color='r', fill=False, linestyle='--')
        ax.add_artist(circle)

        ax.text(0.95, 0.05, 'Radius: {:.1f} arcsec'.format(r_peak*3600), transform=ax.transAxes, verticalalignment='bottom', color='red', ha='right')
        ax.text(0.95, 0.15, '{:.1f} sigma'.format(sig), transform=ax.transAxes, verticalalignment='bottom', color='red', ha='right')

        # PANEL 2: Galactic histogram
        ax = axs[1]
        if len(x_gals) > 0:
            signal_gals = np.histogram2d(x_gals, y_gals, bins=[bins, bins])[0]
            convolution_gals = scipy.ndimage.filters.gaussian_filter(signal_gals, sigma_smooth/delta_x).T
            pc2 = ax.pcolormesh(bins, bins, convolution_gals, cmap='Greys', rasterized=True)
            # set vmin and vmax to be the same as pc
            vmin, vmax = np.percentile(convolution_gals, [1, 99.999])
            pc2.set_clim(vmin, vmax+0.2)
            # fig.colorbar(pc2, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0))
        
        ax.text(0.07, 0.92, 'Galaxies', transform=ax.transAxes, verticalalignment='top', bbox=props)
        ax.set_xlim(bound, -bound)
        ax.set_ylim(-bound, bound)
        ax.set_xlabel(r'$\Delta {\rm RA}$ (deg)')
        ax.set_ylabel(r'$\Delta {\rm Dec}$ (deg)')

        # PANEL 3: Hess Diagram
        ax = axs[2]
        xbins = np.arange(-0.75, 1.5, 0.1)
        mag_limit = self.cfg['catalog']['mag_max']
        ybins = np.arange(mag_limit - 8.0, mag_limit + 0.5, 0.25) 
        
        fg = np.histogram2d(color[inner], mag_1[inner], bins=[xbins, ybins])[0].T
        bg = np.histogram2d(color[outer], mag_1[outer], bins=[xbins, ybins])[0].T
        
        mask_abs = np.absolute(fg) + np.absolute(bg)
        mask_abs[mask_abs == 0.] = np.nan
        signal_hess = np.ma.array((fg - bg), mask=np.isnan(mask_abs))
        
        pc3 = ax.pcolormesh(xbins, ybins, signal_hess, cmap='coolwarm', rasterized=True, vmin=-5, vmax=5)

        isochrone.drawIsochrone(iso, ax=ax, color='royalblue', lw=2, linestyle='-', zorder=10)

        ax.set_xlim(-0.75, 1.5)
        ax.set_ylim(mag_limit, mag_limit - 8.0) 
        ax.set_xlabel(rf'{self.band1} $-$ {self.band2} (AB mag)')
        ax.set_ylabel(rf'{self.band1} (AB mag)')
        # fig.colorbar(pc3, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0))

        if 'MC_SOURCE_ID' in self.region.data.columns:
            true_flag = (self.region.data['MC_SOURCE_ID'] == 1) & inner
            ax.scatter(color[true_flag], mag_1[true_flag], s=30, ec='k', fc='lime', zorder=10, alpha=0.7, label='UFD stars')
            print('Number of stars above F106<27.4 and F158<27.4:', np.sum((mag_1[true_flag] < 27.4) & (mag_2[true_flag] < 27.4)))
            try:
                halo_flag = (self.region.data['MC_SOURCE_ID'] == 2) & inner
                if np.sum(halo_flag) > 0:
                    ax.scatter(color[halo_flag], mag_1[halo_flag], s=30, ec='k', fc='darkviolet', zorder=10, alpha=0.7, label='Halo stars')
            except:
                pass
            # ax.scatter(color[iso_filter], mag_1[iso_filter], s=20, ec='none', fc='darkgreen', alpha=0.7, label='Isochrone stars')
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(0, 1.0), bbox_transform=ax.transAxes, edgecolor='black', handletextpad=0, labelspacing=0, borderpad=0.3)
        # Save
        if outfile is not None:
            save_dir = self.cfg.get('output', {}).get('save_dir', './')
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, outfile), bbox_inches='tight')
            plt.close(fig)