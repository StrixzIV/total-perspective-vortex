import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

def compute_erd(subject, runs, active_event_name):
    """
    Compute Event-Related Desynchronization (ERD) percentage:
    ERD% = ((Power_Active - Power_Baseline) / Power_Baseline) * 100
    """
    # Fetch EDF files and load raw
    edf_files = mne.datasets.eegbci.load_data(subject, runs, verbose='WARNING')
    raws = [mne.io.read_raw_edf(f, preload=True, verbose='WARNING') for f in edf_files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    
    # Bandpass filter to mu + beta bands (8-30 Hz)
    raw.filter(8., 30., fir_design='firwin', verbose='WARNING')
    
    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose='WARNING')
    
    # Map active event and baseline (T0)
    mapping_active = {}
    mapping_base = {}
    for key, val in event_id.items():
        if active_event_name in key:
            mapping_active[active_event_name] = val
        if 'T0' in key:
            mapping_base['T0'] = val
            
    if active_event_name not in mapping_active:
        raise ValueError(f"Event '{active_event_name}' not found in runs {runs}. Found: {event_id}")
    if 'T0' not in mapping_base:
        raise ValueError(f"Baseline event 'T0' not found in runs {runs}. Found: {event_id}")
        
    # Epoch data (using 0.5s to 3.5s to avoid filter edge artifacts at trial boundary)
    epochs_active = mne.Epochs(
        raw, events, event_id=mapping_active,
        tmin=0.5, tmax=3.5, baseline=None,
        preload=True, verbose='WARNING'
    )
    
    epochs_base = mne.Epochs(
        raw, events, event_id=mapping_base,
        tmin=0.5, tmax=3.5, baseline=None,
        preload=True, verbose='WARNING'
    )
    
    # Compute channel-specific power (variance along time)
    power_active = np.mean(np.var(epochs_active.get_data(copy=True), axis=-1), axis=0)
    power_base = np.mean(np.var(epochs_base.get_data(copy=True), axis=-1), axis=0)
    
    # Calculate ERD% (negative value = desynchronization / power decrease)
    erd_pct = ((power_active - power_base) / power_base) * 100
    
    return erd_pct, raw.info

def main():
    subject = 1
    if len(sys.argv) > 1:
        try:
            subject = int(sys.argv[1])
        except ValueError:
            pass
            
    print(f"Computing ERD topographic maps for subject {subject}...")
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    try:
        # 1. Left Hand Imagery (Runs 4, 8, 12 - Event T1)
        erd_left, info = compute_erd(subject, [4, 8, 12], 'T1')
        
        # 2. Right Hand Imagery (Runs 4, 8, 12 - Event T2)
        erd_right, _ = compute_erd(subject, [4, 8, 12], 'T2')
        
        # 3. Feet Imagery (Runs 6, 10, 14 - Event T2)
        erd_feet, _ = compute_erd(subject, [6, 10, 14], 'T2')
        
    except Exception as e:
        print(f"Error computing ERD: {e}")
        print("Please ensure subject runs are cached. Training subject first caches the runs.")
        sys.exit(1)
        
    # Plotting topomaps side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # We want a common symmetric colormap centered at 0 to show desynchronization (blue) vs synchronization (red)
    # Finding the min and max limits for symmetric colormap
    max_val = max(np.max(np.abs(erd_left)), np.max(np.abs(erd_right)), np.max(np.abs(erd_feet)))
    # Clip max_val to a reasonable limit for visualization (e.g. 50% change)
    vmin, vmax = -max_val, max_val
    # If values are very small or large, set standard limits
    if max_val < 10:
        vmin, vmax = -15, 15
    elif max_val > 60:
        vmin, vmax = -60, 60
        
    # 1. Left Hand topomap
    im_left, _ = mne.viz.plot_topomap(erd_left, info, axes=axes[0], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
    axes[0].set_title("Imagining Left Hand\n(Desynchronization at Right C4)", fontsize=11, fontweight='bold')
    
    # 2. Right Hand topomap
    im_right, _ = mne.viz.plot_topomap(erd_right, info, axes=axes[1], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
    axes[1].set_title("Imagining Right Hand\n(Desynchronization at Left C3)", fontsize=11, fontweight='bold')
    
    # 3. Feet topomap
    im_feet, _ = mne.viz.plot_topomap(erd_feet, info, axes=axes[2], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
    axes[2].set_title("Imagining Both Feet\n(Desynchronization at Central Cz)", fontsize=11, fontweight='bold')
    
    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im_left, cax=cbar_ax)
    cbar.set_label('ERD / ERS (% change in mu/beta power)', fontsize=10, fontweight='bold')
    
    fig.suptitle(f"Event-Related Desynchronization (ERD) Topographic Maps\nSubject {subject} - Mu/Beta Band (8-30 Hz)", fontsize=13, fontweight='bold', y=1.10)
    
    plt.savefig('assets/erd_topomaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved 'assets/erd_topomaps.png' successfully!")

if __name__ == '__main__':
    main()
