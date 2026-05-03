import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
# Users: Point this directly to the specific CSV file you want to process
FILE_PATH = "sample_lightcurve.csv" 
OUTPUT_DIR = "Plots"
# Preprocessing raw data in bins
BIN_SIZE = 3

# Flare detection parameters
PEAK_THRESHOLD = 2                 # Peak significance threshold (sigma above median)  
SADDLE_RATIO     = 0.2             # Saddle-point merge depth ratio
MIN_CLUSTER_SIZE = 5               # Minimum points in a valid cluster
SMOOTH_WINDOW    = 7               # Window size for the gradient (descent logic)
REGION_THRESHOLD = 1             # The median of the entire flare must be at least 1 sigma above global median
MAX_GAP = 60                      # Maximum temporal gap (days) before stopping expansion

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_light_curve(filepath):
    df = pd.read_csv(filepath)
    df = df[df['catflags'] == 0].copy()
    return df


def get_uneven_gradient(mjd, flux, window):
    """
    O(N) gradient estimation for unevenly spaced data.
    Provides a smoothed 'trend' used for the expansion/descent logic.
    """
    n = len(flux)
    grad = np.zeros(n)
    half_w = window // 2
    mjd_ref = mjd - mjd[0]

    cs_x  = np.cumsum(np.concatenate(([0], mjd_ref)))
    cs_y  = np.cumsum(np.concatenate(([0], flux)))
    cs_xx = np.cumsum(np.concatenate(([0], mjd_ref**2)))
    cs_xy = np.cumsum(np.concatenate(([0], mjd_ref * flux)))

    for i in range(n):
        s = max(0, i - half_w)
        e = min(n, i + half_w + 1)
        w = e - s
        if w < 2:
            continue

        sum_x  = cs_x[e]  - cs_x[s]
        sum_y  = cs_y[e]  - cs_y[s]
        sum_xx = cs_xx[e] - cs_xx[s]
        sum_xy = cs_xy[e] - cs_xy[s]

        denom = w * sum_xx - sum_x**2
        grad[i] = (w * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0

    return grad


def bin_light_curve(mjd, flux, fluxerr, mag, magerr, bin_size=BIN_SIZE):
    """O(N) Vectorized light curve binning using np.bincount."""
    bin_edges = np.arange(mjd.min(), mjd.max() + bin_size, bin_size)
    bin_indices = np.digitize(mjd, bin_edges) - 1
    num_bins = len(bin_edges)

    wf, wm = 1.0 / fluxerr**2, 1.0 / magerr**2
    counts = np.bincount(bin_indices, minlength=num_bins)
    sum_wf = np.bincount(bin_indices, weights=wf, minlength=num_bins)
    sum_wm = np.bincount(bin_indices, weights=wm, minlength=num_bins)

    valid = (counts > 0) & (sum_wf > 0) & (sum_wm > 0)

    b_t  = np.bincount(bin_indices, weights=mjd,      minlength=num_bins)[valid] / counts[valid]
    b_f  = np.bincount(bin_indices, weights=flux * wf, minlength=num_bins)[valid] / sum_wf[valid]
    b_fe = np.sqrt(1.0 / sum_wf[valid])
    b_m  = np.bincount(bin_indices, weights=mag * wm,  minlength=num_bins)[valid] / sum_wm[valid]
    b_me = np.sqrt(1.0 / sum_wm[valid])

    return b_t, b_f, b_fe, b_m, b_me


# ============================================================
# PEAK-FIRST FLARE DETECTION
# ============================================================
def detect_flares(mjd, flux, mag, threshold_factor):
    n_points = len(flux)
    if n_points == 0:
        return [], np.array([]), np.array([]), np.array([])


    median_f = np.median(flux)
    std_f    = np.std(flux) 
    
    peak_threshold = median_f + threshold_factor * std_f

    # Calculate smoothed gradient for expansion boundaries
    grad = get_uneven_gradient(mjd, flux, window=SMOOTH_WINDOW)

    # Find Local Maxima on RAW flux (to preserve peak sensitivity)
    peaks = []
    if n_points >= 2 and flux[0] > flux[1]:
        peaks.append(0)
    for i in range(1, n_points - 1):
        if flux[i] > flux[i - 1] and flux[i] > flux[i + 1]:
            peaks.append(i)
    if n_points >= 2 and flux[-1] > flux[-2]:
        peaks.append(n_points - 1)

    peaks = [p for p in peaks if flux[p] > peak_threshold]
    if not peaks:
        return [], np.array([]), np.array([]), np.array([])

    # BFS Expansion using the Smoothed Gradient
    assignments = np.full(n_points, -1)
    frontiers = {p: {'left': p, 'right': p, 'active': True} for p in peaks}
    for p in peaks:
        assignments[p] = p

    any_active = True
    while any_active:
        any_active = False
        for p in peaks:
            if not frontiers[p]['active']:
                continue
            expanded = False

            # Expand LEFT: Rise phase (moving backward in time)
            l = frontiers[p]['left']
            # STOP EXPANSION if the point dips below the global median or is farther than maximum gap
            if l > 0 and assignments[l - 1] == -1 and flux[l - 1] >= median_f and (mjd[l] - mjd[l - 1]) <= MAX_GAP:
                if flux[l - 1] < flux[l]:
                    assignments[l - 1] = p
                    frontiers[p]['left'] = l - 1
                    expanded = True
                elif grad[l] >= 0:
                    assignments[l - 1] = p
                    frontiers[p]['left'] = l - 1
                    expanded = True

            # Expand RIGHT: Decay phase (moving forward in time)
            r = frontiers[p]['right']
            # STOP EXPANSION if the point dips below the global median or is farther than maximum gap
            if r < n_points - 1 and assignments[r + 1] == -1 and flux[r + 1] >= median_f and (mjd[r + 1] - mjd[r]) <= MAX_GAP:
                if flux[r + 1] < flux[r]:
                    assignments[r + 1] = p
                    frontiers[p]['right'] = r + 1
                    expanded = True
                elif grad[r] <= 0:
                    assignments[r + 1] = p
                    frontiers[p]['right'] = r + 1
                    expanded = True

            if not expanded:
                frontiers[p]['active'] = False
            else:
                any_active = True

    # Build raw clusters
    raw_clusters = []
    for p in peaks:
        l, r = frontiers[p]['left'], frontiers[p]['right']
        if (r - l + 1) >= MIN_CLUSTER_SIZE:
            raw_clusters.append({'start_idx': l, 'end_idx': r, 'peak_flux': flux[p]})

    if not raw_clusters:
        return [], np.array([]), np.array([]), np.array([])

    # Saddle-Point Merging
    merged_indices = [[raw_clusters[0]['start_idx'], raw_clusters[0]['end_idx']]]
    curr_peak_f = raw_clusters[0]['peak_flux']
    for i in range(1, len(raw_clusters)):
        prev_s, prev_e = merged_indices[-1]
        next_s, next_e = raw_clusters[i]['start_idx'], raw_clusters[i]['end_idx']
        next_peak_f = raw_clusters[i]['peak_flux']

        temporal_gap = mjd[next_s] - mjd[prev_e]
        no_data_between = next_s <= prev_e + 1

        if temporal_gap > MAX_GAP:
            # Large data void — always separate
            merged_indices.append([next_s, next_e])
            curr_peak_f = next_peak_f
        elif no_data_between or next_s <= prev_e + 2:
            # Adjacent or overlapping — always merge
            merged_indices[-1][1] = next_e
            curr_peak_f = max(curr_peak_f, next_peak_f)
        else:
            # True saddle exists between clusters
            saddle_f = np.min(flux[prev_e + 1:next_s])
            is_shallow = (saddle_f - median_f) > (SADDLE_RATIO * (min(curr_peak_f, next_peak_f) - median_f))
            if is_shallow:
                merged_indices[-1][1] = next_e
                curr_peak_f = max(curr_peak_f, next_peak_f)
            else:
                merged_indices.append([next_s, next_e])
                curr_peak_f = next_peak_f

    # ============================================================
    # THE LAST GATE: FILTER OUT REGIONS WITH LOW MEDIAN
    # ============================================================
    valid_indices = []
    support_threshold = median_f + REGION_THRESHOLD * std_f

    for s, e in merged_indices:
        region_flux = flux[s:e + 1]
        
        region_median = np.median(region_flux)
        if region_median >= support_threshold:
            valid_indices.append([s, e])
            
    merged_indices = valid_indices

    # Build regions
    flares, edges, b_flux, b_mag = [], [mjd[0]], [], []
    last_idx = 0
    for s, e in merged_indices:
        if s > last_idx:
            edges.append(mjd[s])
            b_flux.append(np.mean(flux[last_idx:s]))
            b_mag.append(np.mean(mag[last_idx:s]))
        edges.append(mjd[e] + 0.001)
        b_flux.append(np.mean(flux[s:e + 1]))
        b_mag.append(np.mean(mag[s:e + 1]))
        flares.append({
            'start':        mjd[s],
            'end':          mjd[e],
            'peak_flux':    np.max(flux[s:e + 1]),
            'significance': (np.max(flux[s:e + 1]) - median_f) / std_f
        })
        last_idx = e + 1
    if last_idx < n_points:
        edges.append(mjd[-1])
        b_flux.append(np.mean(flux[last_idx:]))
        b_mag.append(np.mean(mag[last_idx:]))
    return flares, np.array(edges), np.array(b_flux), np.array(b_mag)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_name = os.path.basename(FILE_PATH).replace('.csv', '')

    plt.rcParams.update({
        'font.family':         'serif',
        'font.size':           11,
        'axes.linewidth':      0.8,
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.top':           True,
        'ytick.right':         True,
        'xtick.major.size':    5,
        'ytick.major.size':    5,
        'xtick.minor.size':    3,
        'ytick.minor.size':    3,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    df = load_light_curve(FILE_PATH)
    
    if df is None:
        print(f"No data found at {FILE_PATH}")
        return

    b_t, b_f, b_fe, b_m, b_me = bin_light_curve(
        df['mjd'].values, df['flux'].values, df['fluxerr'].values,
        df['mag'].values, df['magerr'].values
    )

    flares, edges, fluxes, mags = detect_flares(b_t, b_f, b_m, PEAK_THRESHOLD)

    # Raw photometry
    ax.plot(df['mjd'], df['mag'], '.', color='silver', alpha=0.3, ms=2, zorder=1)
    
    # Binned photometry 
    ax.errorbar(b_t, b_m, yerr=b_me, fmt='o', color='r', ms=3, alpha=0.7,
                elinewidth=0.5, capsize=0, zorder=2)
    
    # PDRS blocks (black line)
    if len(edges) > 1:
        ax.step(edges[:-1], mags, where='post', color='black', lw=1.2, zorder=3)
    
    # Shaded flare regions (Orange)
    for f in flares:
        ax.axvspan(f['start'], f['end'], color='orange', alpha=0.3, zorder=0)
        
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_xlabel('Relative MJD (days)', fontsize=11)
    ax.invert_yaxis()
    
    n = len(flares)
    ax.text(0.98, 0.05,
            f'$\\sigma_{{\\mathrm{{thresh}}}} = {PEAK_THRESHOLD}$  |  {n} flare{"s" if n != 1 else ""}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))

    # Legend moved completely outside the plot area
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker='.', color='w', markerfacecolor='silver', ms=6, label='Raw photometry'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', ms=5, label=f'{BIN_SIZE}-day binned'),
        Line2D([0], [0], color='black', lw=1.2, label='PDRS blocks'),
        Patch(facecolor='orange', alpha=0.3, label='Detected flares'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9, edgecolor='0.7',
              fancybox=False, handletextpad=0.4)

    fig.suptitle("2597579 SDSS Stripe 82", fontsize=13, fontweight='bold', y=0.95)

    # Clean generic filename output
    out_path = os.path.join(OUTPUT_DIR, f"{source_name}_detected_flares.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')  
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
