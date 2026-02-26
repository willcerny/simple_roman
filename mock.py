## Handles mock galaxy injection, for Roman Cycle 1 proposal

import os
import sys
import numpy as np
from astropy import units as u
from astropy.table import Table

import artpop
import rosesim
from rosesim.rose import RomanGalaxy
sys.path.append('/home/jiaxuanl/Research/Roman_Cycle1/')
from ripples.utils import setup_dolphot_cat, apply_obs_model_two_band
from ripples import ripples_completeness_dict, ripples_mag_uncertainty_dict, mass_size_LVDB

class MockGalaxy():
    def __init__(self, ra, dec, distance, log_m_star, reff=None, seed=42):
        """
        Initialize a mock galaxy.

        Args:
            ra, dec: RA and Dec of the galaxy in degrees
            distance: distance to the galaxy in Mpc
            log_m_star: log of the stellar mass in M_sun
            reff: effective radius in kpc. If None, use the LVDB mass-size relation.
            seed: random seed for reproducibility
        """
        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.log_m_star = log_m_star
        dmod = 5 * np.log10(distance) + 25
        log_age = 10.1
        feh = -2.0

        if reff is None:
            reff = 10**mass_size_LVDB(log_m_star) / 1000 * u.kpc
        else:
            reff = reff * u.kpc
        self.reff = reff

        gal_kwargs = {'age': (10**log_age) * u.yr,
                  'feh': feh,
                  'total_mass': 10**log_m_star,
                  'r_eff': reff,
                  'distance': distance * u.Mpc}
        gal = RomanGalaxy(prefix='mock', 
                          data_dir="/scratch/gpfs/JENNYG/jiaxuanl/Data/Roman/UFD_detection")
        reff = (gal_kwargs['r_eff'] / gal_kwargs['distance']).cgs.value * 206265
        s = int(reff / rosesim.pixel_scale * 15)
        s = s + (s+1) % 2

        dmod = 5 * np.log10(gal_kwargs['distance'].value) + 25
        abs_mag_lim = 1
        mag_lim = dmod + abs_mag_lim

        iso = artpop.Isochrone.from_parsec(os.path.join(os.environ['ROSESIM_DATA_PATH'], 
                                                        'PARSEC/PARSEC_v1.2S_Roman_vega_nTP20.dat'),
                                        log_age=log_age, MH=feh)
        iso.mag_table.rename_columns(iso.filters, [item.replace('mag', '') for item in iso.filters])

        sp = artpop.SSP(iso, total_mass=gal_kwargs['total_mass'],
                        distance=gal_kwargs['distance'],
                        mag_limit=mag_lim, mag_limit_band='F158',
                        random_state=np.random.RandomState(seed))
        src = artpop.SersicSP(sp, n=0.8, theta=0 * u.deg, ellip=0,
                            r_eff=gal_kwargs['r_eff'], xy_dim=s, pixel_scale=rosesim.pixel_scale)
        # convert Vega to AB
        for filt in src.mags.colnames:
            src.mags[filt] += rosesim.Roman_zp_AB_Vega_parsec[filt]
        gal.load_src(src, ra=ra, dec=dec)
        gal.gen_catalog()
        gal.obj_cat['label'] = 1

        self.gal = gal
        self.src = src

    def make_star_catalog(self, nexp=5, MA_table='IM_600_16'):
        rng = np.random.default_rng(123)
        if MA_table != "IM_600_16":
            raise ValueError("MA_table must be 'IM_600_16'")
        if nexp not in [5]:
            raise ValueError("nexp must be 5 (for hosts at 3.5 Mpc)")

        ufd = self.gal.obj_cat

        # Ultra-faint dwarf sources
        ufd = apply_obs_model_two_band(
            -2.5 * np.log10(ufd["F106"]), 
            -2.5 * np.log10(ufd["F158"]),
            bands=['F106', 'F158'],
            completeness_dict=ripples_completeness_dict['3.5Mpc'],
            mag_uncertainty_dict=ripples_mag_uncertainty_dict['3.5Mpc'],
            rng=rng,
            detection="both",     # since you care about colors
            depth_factor=1
        )
        ufd['MC_SOURCE_ID'] = self.gal.obj_cat['label']
        ufd['RA'] = self.gal.obj_cat['ra']
        ufd['DEC'] = self.gal.obj_cat['dec']

        self.star_cat = Table(ufd).to_pandas()