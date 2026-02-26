import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd  # <--- Added this
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import time
import os, sys

# Import your custom modules
sys.path.append("/home/jiaxuanl/Research/Roman_Cycle1/ripples/")
from simple_roman import search

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

CHECKPOINT_FILE = 'mass_completeness_checkpoint.csv'

# ==========================================
# 1. Global Simulation Parameters
# ==========================================
central_ra = 201.3704160
central_dec = -43.0166667
box_size_deg = 200.0 / 3600.0
fixed_distance = 3.5 # Mpc

mass_array = np.arange(3.8, 5.5, 0.1)
seeds_per_mass = 60

def compute_bootstrap_errors(detections, n_bootstraps=1000):
    n_trials = len(detections)
    if n_trials == 0:
        return 0.0, 0.0
        
    # Resample and calculate rates
    bootstrapped_samples = np.random.choice(detections, size=(n_bootstraps, n_trials), replace=True)
    bootstrapped_rates = np.mean(bootstrapped_samples, axis=1)
    
    # 16th and 84th percentiles for 1-sigma
    lower_bound = np.percentile(bootstrapped_rates, 16)
    upper_bound = np.percentile(bootstrapped_rates, 84)
    
    actual_rate = np.mean(detections)
    err_low = actual_rate - lower_bound
    err_high = upper_bound - actual_rate
    
    return max(0, err_low), max(0, err_high) # Ensure non-negative for plotting

# ==========================================
# 2. Worker Function
# ==========================================
def run_single_simulation(params):
    log_m, seed, mock_id = params
    
    np.random.seed(seed)
    delta_dec = np.random.uniform(-box_size_deg / 2.0, box_size_deg / 2.0)
    delta_ra = np.random.uniform(-box_size_deg / 2.0, box_size_deg / 2.0) / np.cos(np.radians(central_dec))
    mock_ra = central_ra + delta_ra
    mock_dec = central_dec + delta_dec
    
    # ufd_search is accessed via global scope in the worker
    results = ufd_search.run_injection_trial(
        mock_ra=mock_ra, 
        mock_dec=mock_dec, 
        mock_distance=fixed_distance, 
        mock_log_m_star=log_m,
        mc_source_id=mock_id,
        seed=seed
    )
    
    output = {
        'MC_SOURCE_ID': mock_id, 'log_m': log_m, 'seed': seed,
        'true_ra': mock_ra, 'true_dec': mock_dec,
        'RA': np.nan, 'DEC': np.nan, 'SIG': 0.0, 'sep': np.nan,
        'detected': False
    }

    if results is None or len(results['SIG']) == 0:
        return output

    true_coord = SkyCoord(mock_ra, mock_dec, unit='deg')
    cat = Table(results)
    coords = SkyCoord(cat['RA'], cat['DEC'], unit='deg')
    cat['sep'] = coords.separation(true_coord).to(u.arcsec).value
    
    # Match criteria: separation < 10" and Significance > 4.5
    temp = cat[cat['sep'] < 10.0]
    if len(temp) > 0:
        temp.sort('sep')
        best_match = temp[0]
        output.update({
            'RA': best_match['RA'], 'DEC': best_match['DEC'],
            'SIG': best_match['SIG'], 'sep': best_match['sep'],
            'detected': best_match['SIG'] > 4.5
        })

    return output

# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == '__main__':
    print("Initializing global Search object...")
    global ufd_search
    ufd_search = search.Search(
        config_file='/home/jiaxuanl/Research/Roman_Cycle1/ripples/simple_roman/config.yaml', 
        ra=central_ra, dec=central_dec
    )
    
    tasks = [(m, s, i) for i, m in enumerate(mass_array) for s in range(seeds_per_mass)]
            
    num_cores = 12 # leave 1 core for the OS
    print(f"\nDispatching {len(tasks)} simulations across {num_cores} cores...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_list = pool.map(run_single_simulation, tasks)
        
    print(f"Simulations completed in {(time.time() - start_time) / 60:.2f} minutes.")
    
    # Create DataFrame directly from list of dicts
    df = pd.DataFrame(results_list)
    df.to_csv('mass_completeness_results.csv', index=False)

    # ==========================================
    # 4. Aggregation and Plotting
    # ==========================================
    rates, yerr_low, yerr_high = [], [], []
    unique_masses = np.sort(df['log_m'].unique())

    print("Calculating Bootstrap Errors...")
    for log_m in unique_masses:
        detections = df[df['log_m'] == log_m]['detected'].astype(int).values
        rate = np.mean(detections)
        e_low, e_high = compute_bootstrap_errors(detections, n_bootstraps=5000)
        
        rates.append(rate)
        yerr_low.append(e_low)
        yerr_high.append(e_high)
        print(f"Log M* = {log_m:.2f} | Rate: {rate:.2f} (+{e_high:.2f} / -{e_low:.2f})")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(unique_masses, rates, yerr=[yerr_low, yerr_high], 
                 fmt='o-', color='royalblue', lw=2, markersize=8, capsize=5, ecolor='slategray')

    plt.axhline(0.5, color='gray', linestyle='--', label='50% Completeness')
    plt.title(f'UFD Detection Completeness at {fixed_distance} Mpc', fontsize=14)
    plt.xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=12)
    plt.ylabel('Detection Rate (SIG > 4.5)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend()

    plt.savefig('completeness_curve_with_errors.png', dpi=300, bbox_inches='tight')
    # plt.show()