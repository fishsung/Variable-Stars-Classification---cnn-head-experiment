import os
import gc
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from celerite2 import GaussianProcess, terms

if len(sys.argv) != 2:
    print("Usage: python build_dataset.py <TYPE>")
    sys.exit(1)

TARGET_TYPE = sys.argv[1]

# -----------------------------
# Settings
# -----------------------------
phot_dir = "Phot_I_data"          # original OGLE photometry
output_dir = "lightcurve_plots"   # save images here
os.makedirs(output_dir, exist_ok=True)

train_count = 4500
test_count  = 500
img_size = 512
random_seed = 42
np.random.seed(random_seed)

# -----------------------------
# 1. Load merged CSV
# -----------------------------
data = pd.read_csv("ogle3_full_with_windex.csv")
numeric_cols = ['i', 'v', 'period', 'amplitude', 'windex']
data[numeric_cols] = data[numeric_cols].astype(float)

# -----------------------------
# 2. Helper functions
# -----------------------------
def phase_fold(time, period):
    return (time % period) / period

def normalize_mag(mag):
    mag = np.array(mag)
    return (mag - mag.min()) / (mag.max() - mag.min())

def smooth_curve(mag, window=7, poly=2):
    if len(mag) < window:
        window = len(mag) // 2 * 2 + 1
    return savgol_filter(mag, window, poly)

def remove_outliers(mag, smoothed):
    residuals = mag - smoothed
    sigma = np.std(residuals)
    mask = np.abs(residuals) <= 3*sigma
    return mag[mask], mask

def plot_lc_image(phase, mag, size=512, save_path=None):
    phase_double = np.concatenate([phase, phase + 1])
    mag_double = np.concatenate([mag, mag])
    
    fig, ax = plt.subplots(figsize=(1,1), dpi=size)
    ax.scatter(phase_double, mag_double, s=1, color='black')
    ax.set_facecolor('white')
    ax.set_axis_off()
    ax.set_xlim(0,2)  # two phases
    ax.set_ylim(0,1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=size)
    plt.close(fig)
    del fig, ax

def augment_lightcurve(phase, mag):
    # Sort
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    mag_sorted = mag[sort_idx]

    # -----------------------------
    # REMOVE duplicate phase points
    # -----------------------------
    unique_mask = np.diff(phase_sorted, prepend=-1) > 1e-6
    phase_sorted = phase_sorted[unique_mask]
    mag_sorted = mag_sorted[unique_mask]

    # If too few points → fallback (important!)
    if len(phase_sorted) < 10:
        return phase_sorted, mag_sorted

    # -----------------------------
    # GP kernel (more stable)
    # -----------------------------
    kernel = terms.SHOTerm(
        S0=np.var(mag_sorted) + 1e-6,
        w0=2*np.pi,
        Q=5.0
    )

    gp = GaussianProcess(kernel, mean=np.mean(mag_sorted))

    # -----------------------------
    # ADD NOISE (critical!)
    # -----------------------------
    yerr = 1e-3 * np.ones_like(phase_sorted)

    try:
        gp.compute(phase_sorted, yerr=yerr)
    except Exception:
        # fallback if GP fails
        return phase_sorted, mag_sorted

    # -----------------------------
    # Sample new phase points
    # -----------------------------
    n_points = len(phase_sorted)
    phase_new = np.sort(np.random.rand(n_points))

    # Predict
    mu, var = gp.predict(mag_sorted, phase_new, return_var=True)

    # Numerical safety
    var = np.maximum(var, 1e-6)

    # Sample
    mag_new = np.random.normal(mu, np.sqrt(var))

    return phase_new, mag_new

train_csv_path = f"datasets/train_{TARGET_TYPE}.csv"
test_csv_path = f"datasets/test_{TARGET_TYPE}.csv"

# initialize CSVs
pd.DataFrame(columns=data.columns).to_csv(train_csv_path, index=False)
pd.DataFrame(columns=data.columns).to_csv(test_csv_path, index=False)


def process_and_save(row, suffix="", csv_path=None):
    star_id = row['id']
    phot_file = os.path.join(phot_dir, f"{star_id}.dat")

    if not os.path.exists(phot_file):
        return None

    try:
        t, m = np.loadtxt(phot_file, usecols=(0,1), unpack=True)
    except:
        return None

    # preprocess
    phase = phase_fold(t, row['period'])
    mag_norm = normalize_mag(m)
    mag_smooth = smooth_curve(mag_norm)
    mag_clean, mask = remove_outliers(mag_norm, mag_smooth)
    phase_clean = phase[mask]

    # sort
    sort_idx = np.argsort(phase_clean)
    phase_clean = phase_clean[sort_idx]
    mag_clean = mag_clean[sort_idx]

    # new ID if needed
    new_id = star_id if suffix == "" else f"{star_id}_{suffix}"

    # save image
    save_path = os.path.join(output_dir, f"{new_id}.png")
    plot_lc_image(phase_clean, mag_clean, size=img_size, save_path=save_path)

    # write CSV immediately
    row_out = row.copy()
    row_out['id'] = new_id
    pd.DataFrame([row_out]).to_csv(csv_path, mode='a', header=False, index=False)

    # free memory
    del t, m, phase, mag_norm, mag_smooth, mag_clean, phase_clean

    return True


def process_and_augment(row, suffix, csv_path):
    star_id = row['id']
    phot_file = os.path.join(phot_dir, f"{star_id}.dat")

    if not os.path.exists(phot_file):
        return None

    try:
        t, m = np.loadtxt(phot_file, usecols=(0,1), unpack=True)
    except:
        return None

    # preprocess
    phase = phase_fold(t, row['period'])
    mag_norm = normalize_mag(m)
    mag_smooth = smooth_curve(mag_norm)
    mag_clean, mask = remove_outliers(mag_norm, mag_smooth)
    phase_clean = phase[mask]

    # sort
    sort_idx = np.argsort(phase_clean)
    phase_clean = phase_clean[sort_idx]
    mag_clean = mag_clean[sort_idx]

    # augment
    phase_synth, mag_synth = augment_lightcurve(phase_clean, mag_clean)

    new_id = f"{star_id}_{suffix}"

    # save image
    save_path = os.path.join(output_dir, f"{new_id}.png")
    plot_lc_image(phase_synth, mag_synth, size=img_size, save_path=save_path)

    # write CSV
    row_out = row.copy()
    row_out['id'] = new_id
    pd.DataFrame([row_out]).to_csv(csv_path, mode='a', header=False, index=False)

    # free memory
    del t, m, phase, mag_norm, mag_smooth
    del mag_clean, phase_clean, phase_synth, mag_synth

    return True

# -----------------------------
# MAIN LOOP
# -----------------------------

type_data = data[data['type'] == TARGET_TYPE]

for sub in type_data['mode'].unique():
    sub_data = type_data[type_data['mode'] == sub]

    n_total = train_count + test_count

    # sampling
    if len(sub_data) >= n_total:
        sampled = sub_data.sample(n=n_total, random_state=random_seed)
    else:
        sampled = sub_data

    sampled_train, sampled_test = train_test_split(
        sampled,
        test_size=test_count / n_total,
        random_state=random_seed
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    valid_train_rows = []

    for _, row in tqdm(sampled_train.iterrows(), total=len(sampled_train), desc=f"Train {TARGET_TYPE}-{sub}"):
        if process_and_save(row, suffix="", csv_path=train_csv_path):
            valid_train_rows.append(row)

    current_n = len(valid_train_rows)

    if current_n < train_count:
        needed = train_count - current_n

        for i in range(needed):
            row = valid_train_rows[np.random.randint(0, current_n)]
            process_and_augment(row, f"trainsyn_{i}", train_csv_path)

    # -----------------------------
    # TEST
    # -----------------------------
    valid_test_rows = []

    for _, row in tqdm(sampled_test.iterrows(), total=len(sampled_test), desc=f"Test {TARGET_TYPE}-{sub}"):
        if process_and_save(row, suffix="", csv_path=test_csv_path):
            valid_test_rows.append(row)

    current_n = len(valid_test_rows)

    if current_n < test_count:
        needed = test_count - current_n

        for i in range(needed):
            row = valid_test_rows[np.random.randint(0, current_n)]
            process_and_augment(row, f"testsyn_{i}", test_csv_path)

    # cleanup per subtype
    gc.collect()