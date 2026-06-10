import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
from utils.preprocessing import load_and_preprocess
from utils.pipeline import get_run_pair
from utils.csp import CSP

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <subject> <run>")
        sys.exit(1)
        
    subject = int(sys.argv[1])
    run = int(sys.argv[2])
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Map run to the experiment runs pair
    runs = get_run_pair(run)
    
    print(f"Loading raw data for visualization (subject {subject}, runs {runs})...")
    # Fetch .edf files and load raw
    edf_files = mne.datasets.eegbci.load_data(subject, runs, verbose='WARNING')
    raws = [mne.io.read_raw_edf(f, preload=True, verbose='WARNING') for f in edf_files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    
    # Store pre-filtered copy
    raw_pre = raw.copy()
    
    # Apply bandpass filter to mu + beta bands (8-30 Hz)
    print("Applying bandpass filter (8-30 Hz)...")
    raw_filtered = raw.copy().filter(8., 30., fir_design='firwin', verbose='WARNING')
    
    # 1. Raw vs Filtered Signal Plot (first 10s, 5 channels)
    print("Plotting raw and filtered signals...")
    target_channels = [ch for ch in ['C3', 'Cz', 'C4', 'FC3', 'FC4'] if ch in raw.ch_names]
    if len(target_channels) < 5:
        target_channels = raw.ch_names[:5]
        
    raw_plot_pre = raw_pre.copy().crop(0.0, 10.0)
    raw_plot_post = raw_filtered.copy().crop(0.0, 10.0)
    
    data_pre, times_pre = raw_plot_pre.get_data(picks=target_channels, return_times=True)
    data_post, times_post = raw_plot_post.get_data(picks=target_channels, return_times=True)
    
    # Save Raw Signal Plot
    fig, axes = plt.subplots(len(target_channels), 1, figsize=(10, 8), sharex=True)
    for i, ch_name in enumerate(target_channels):
        axes[i].plot(times_pre, data_pre[i] * 1e6, color='#2c3e50')
        axes[i].set_ylabel(f"{ch_name} (μV)")
        axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Raw EEG Signal (First 10s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/raw_signal_plot.png', dpi=150)
    plt.close()
    print("Saved 'assets/raw_signal_plot.png'")
    
    # Save Filtered Signal Plot
    fig, axes = plt.subplots(len(target_channels), 1, figsize=(10, 8), sharex=True)
    for i, ch_name in enumerate(target_channels):
        axes[i].plot(times_post, data_post[i] * 1e6, color='#16a085')
        axes[i].set_ylabel(f"{ch_name} (μV)")
        axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Filtered EEG Signal (8-30 Hz, First 10s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/filtered_signal_plot.png', dpi=150)
    plt.close()
    print("Saved 'assets/filtered_signal_plot.png'")
    
    # 2. Power Spectral Density Plot (pre vs post filter overlay)
    print("Computing PSD overlay...")
    sfreq = raw_pre.info['sfreq']
    data_pre_all = raw_pre.get_data()
    data_post_all = raw_filtered.get_data()
    
    freqs, psd_pre_all = welch(data_pre_all, fs=sfreq, nperseg=int(2*sfreq))
    _, psd_post_all = welch(data_post_all, fs=sfreq, nperseg=int(2*sfreq))
    
    mean_psd_pre = 10 * np.log10(np.mean(psd_pre_all, axis=0))
    mean_psd_post = 10 * np.log10(np.mean(psd_post_all, axis=0))
    
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, mean_psd_pre, label='Raw (Pre-filter)', color='#e74c3c', alpha=0.7, linewidth=1.5)
    plt.plot(freqs, mean_psd_post, label='Filtered (8-30 Hz)', color='#2ecc71', alpha=0.9, linewidth=1.8)
    plt.xlim(1, 50)
    plt.xlabel('Frequency (Hz)', fontsize=11)
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=11)
    plt.title('Power Spectral Density (PSD) Pre vs Post Filtering', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('assets/psd_overlay_plot.png', dpi=150)
    plt.close()
    print("Saved 'assets/psd_overlay_plot.png'")
    
    # 3. Topographic map of CSP filters
    print("Fitting CSP to plot spatial filters topomaps...")
    X, y = load_and_preprocess(subject, runs)
    csp = CSP(n_components=4)
    csp.fit(X, y)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for i in range(4):
        # Plot topomap for each component filter
        mne.viz.plot_topomap(csp.filters_[i], raw.info, axes=axes[i], show=False)
        axes[i].set_title(f"Component {i+1}", fontsize=11, fontweight='bold')
    fig.suptitle('Custom CSP Spatial Filters Topographic Maps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/csp_filters_topomap.png', dpi=150)
    plt.close()
    print("Saved 'assets/csp_filters_topomap.png'")
    
if __name__ == '__main__':
    main()
