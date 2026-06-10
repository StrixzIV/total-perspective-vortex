#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import mne

# Try importing tqdm for a nice progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def get_eegbci_base_path():
    """Dynamically resolve the EEGBCI local storage path using MNE."""
    paths = mne.datasets.eegbci.load_data(1, [3], verbose='WARNING')
    if paths:
        return Path(paths[0]).parent.parent
    raise RuntimeError("Could not resolve EEGBCI base path from MNE.")

def parse_range(range_str, max_val):
    """
    Parse a range string like '1-109' or '3,4,5' or '3-6,8' into a list of integers.
    """
    result = set()
    for part in range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    
    # Validate
    valid = [x for x in sorted(result) if 1 <= x <= max_val]
    return valid

def download_subject(subject, runs, force_update):
    """Download the specified runs for a single subject."""
    try:
        # load_data downloads or updates the cache
        mne.datasets.eegbci.load_data(subject, runs, force_update=force_update, verbose='WARNING')
        return subject, True, None
    except Exception as e:
        return subject, False, str(e)

def main():
    parser = argparse.ArgumentParser(
        description="Download or verify the PhysioNet EEG Motor Movement/Imagery dataset for Total Perspective Vortex."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="1-109",
        help="Subjects to download (e.g., '1-10', '1,2,5', or '1-109'). Default is 1-109."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="3-14",
        help="Runs to download (e.g., '3,4', '3-6', or '3-14' for all experimental runs). Default is 3-14."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads. Default is 4."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check which files are missing/present without downloading."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownloading of all files even if they already exist."
    )
    
    args = parser.parse_args()
    
    # 1. Parse subjects and runs
    try:
        subjects = parse_range(args.subjects, 109)
    except Exception as e:
        print(f"Error parsing subjects: {e}")
        sys.exit(1)
        
    try:
        runs = parse_range(args.runs, 14)
    except Exception as e:
        print(f"Error parsing runs: {e}")
        sys.exit(1)
        
    if not subjects:
        print("No valid subjects specified.")
        sys.exit(1)
    if not runs:
        print("No valid runs specified.")
        sys.exit(1)
        
    print(f"Subjects: {len(subjects)} (ranging from {min(subjects)} to {max(subjects)})")
    print(f"Runs: {len(runs)} ({runs})")
    
    # 2. Get base directory
    try:
        print("Resolving dataset base path...")
        base_path = get_eegbci_base_path()
        print(f"Local storage path: {base_path}")
    except Exception as e:
        print(f"Failed to resolve local storage path: {e}")
        sys.exit(1)
        
    # 3. Check what exists locally
    present = []
    missing = []
    
    for subject in subjects:
        for run in runs:
            file_path = base_path / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
            if file_path.exists():
                present.append((subject, run, file_path))
            else:
                missing.append((subject, run, file_path))
                
    total_files = len(subjects) * len(runs)
    print(f"\nStatus summary:")
    print(f"  Total expected files: {total_files}")
    print(f"  Already downloaded:   {len(present)} ({len(present)/total_files*100:.1f}%)")
    print(f"  Missing:              {len(missing)} ({len(missing)/total_files*100:.1f}%)")
    
    if args.check:
        print("\nCheck complete. Use without '--check' to download missing files.")
        sys.exit(0)
        
    if len(missing) == 0 and not args.force:
        print("\nAll files are already present. Nothing to download.")
        sys.exit(0)
        
    # Determine which subjects actually need downloading
    subjects_to_download = []
    if args.force:
        subjects_to_download = subjects
    else:
        # Find subjects that have at least one missing file
        missing_subjects = set(subj for subj, run, path in missing)
        subjects_to_download = [subj for subj in subjects if subj in missing_subjects]
        
    print(f"\nNeed to download data for {len(subjects_to_download)} subjects.")
    print(f"Starting download using {args.workers} workers...")
    
    failed_subjects = []
    
    if HAS_TQDM:
        pbar = tqdm(total=len(subjects_to_download), desc="Downloading subjects")
    else:
        pbar = None
        
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_subject, subj, runs, args.force): subj
            for subj in subjects_to_download
        }
        
        for future in as_completed(futures):
            subj = futures[future]
            subj, success, err_msg = future.result()
            
            if success:
                if pbar:
                    pbar.update(1)
                else:
                    print(f"  [SUCCESS] Subject {subj:03d}")
            else:
                failed_subjects.append((subj, err_msg))
                if pbar:
                    pbar.update(1)
                    tqdm.write(f"  [FAILED] Subject {subj:03d}: {err_msg}")
                else:
                    print(f"  [FAILED] Subject {subj:03d}: {err_msg}")
                    
    if pbar:
        pbar.close()
        
    print("\nDownload finished.")
    if failed_subjects:
        print(f"Errors occurred during download of {len(failed_subjects)} subjects:")
        for subj, err in failed_subjects:
            print(f"  Subject {subj:03d}: {err}")
        sys.exit(1)
    else:
        print("All requested files downloaded successfully!")

if __name__ == "__main__":
    main()
