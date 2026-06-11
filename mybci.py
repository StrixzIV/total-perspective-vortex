import sys
import numpy as np
from utils.preprocessing import load_and_preprocess
from utils.pipeline import train_and_evaluate, build_pipeline
from utils.predict import predict_stream
from sklearn.model_selection import train_test_split

EXPERIMENTS = {
    0: [3, 4],    # motor execution: hand vs feet
    1: [5, 6],    # motor imagery: hand vs feet
    2: [7, 8],    # motor imagery: left vs right hand
    3: [9, 10],   # motor imagery: hands vs feet
    4: [11, 12],  # motor execution: left vs right hand
    5: [13, 14],  # motor execution: hands vs feet
}

def run_batch_evaluation(wavelet_bonus=False, lda_bonus=False, riemannian_bonus=False, subjects=None):
    """
    Run the batch evaluation across the specified subjects and 6 experiment types.
    """
    if subjects is None:
        subjects = list(range(1, 110))
        
    print(f"Starting batch evaluation (wavelets={'enabled' if wavelet_bonus else 'disabled'}, lda={'custom' if lda_bonus else 'scikit-learn'}, riemannian={'enabled' if riemannian_bonus else 'disabled'})...")
    print("Tip: Run `python download_data.py` first to pre-download all dataset files in parallel.")
    print(f"Evaluating subjects: {subjects}")
    
    # Store accuracies: {exp_id: [accuracies]}
    exp_accuracies = {i: [] for i in range(6)}
    
    for exp_id in range(6):
        runs = EXPERIMENTS[exp_id]
        for subj in subjects:
            try:
                # Load and preprocess
                X, y = load_and_preprocess(subj, runs)
                
                # Check that we have enough trials
                if len(y) < 10:
                    print(f"experiment {exp_id}: subject {subj:03d}: skipped (too few trials: {len(y)})")
                    continue
                
                # Stratified 80/20 train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, shuffle=True
                )
                
                # Build and train pipeline
                pipe = build_pipeline(
                    n_components=6,
                    wavelet_bonus=wavelet_bonus,
                    lda_bonus=lda_bonus,
                    riemannian_bonus=riemannian_bonus
                )
                pipe.fit(X_train, y_train)
                
                # Score on test set
                acc = pipe.score(X_test, y_test)
                exp_accuracies[exp_id].append(acc)
                print(f"experiment {exp_id}: subject {subj:03d}: accuracy = {acc:.4f}")
                
            except Exception as e:
                print(f"experiment {exp_id}: subject {subj:03d}: failed with error: {e}")
                
    print("\nMean accuracy of the six different experiments for all 109 subjects:")
    overall_means = []
    for exp_id in range(6):
        accs = exp_accuracies[exp_id]
        if accs:
            mean_acc = np.mean(accs)
            overall_means.append(mean_acc)
            print(f"experiment {exp_id}: accuracy = {mean_acc:.4f}")
        else:
            print(f"experiment {exp_id}: accuracy = N/A (no subjects processed)")
            
    if overall_means:
        print(f"\nMean accuracy of 6 experiments: {np.mean(overall_means):.4f}")
    else:
        print("\nMean accuracy of 6 experiments: N/A")

def parse_subjects(args):
    """Parse --subjects <range> or <list> if present in command arguments."""
    subjects = list(range(1, 110))
    if "--subjects" in args:
        try:
            idx = args.index("--subjects")
            if idx + 1 < len(args):
                subj_str = args[idx + 1]
                if "-" in subj_str:
                    start, end = map(int, subj_str.split("-"))
                    subjects = list(range(start, end + 1))
                else:
                    subjects = list(map(int, subj_str.split(",")))
        except Exception as e:
            print(f"Warning: Failed to parse --subjects argument, defaulting to all subjects. Error: {e}")
    return subjects

def print_usage():
    print("Usage:")
    print("  python mybci.py                          # Run batch evaluation on all 109 subjects")
    print("  python mybci.py --subjects 1-5           # Run batch evaluation on subjects 1 to 5")
    print("  python mybci.py --wavelet                # Run batch evaluation with Wavelet bonus")
    print("  python mybci.py --lda                    # Run batch evaluation with Custom LDA classifier")
    print("  python mybci.py --riemann                # Run batch evaluation with Riemannian MDM classifier")
    print("  python mybci.py <S> <R> train            # Train on subject S, run R")
    print("  python mybci.py <S> <R> train --wavelet  # Train with Wavelet bonus")
    print("  python mybci.py <S> <R> train --lda      # Train with Custom LDA classifier")
    print("  python mybci.py <S> <R> train --riemann  # Train with Riemannian MDM classifier")
    print("  python mybci.py <S> <R> predict          # Predict on subject S, run R in streaming mode")

def main():
    args = sys.argv[1:]
    
    # Check options
    wavelet_bonus = "--wavelet" in args
    lda_bonus = "--lda" in args
    riemannian_bonus = "--riemann" in args
    
    # Parse subjects if present
    subjects = parse_subjects(args)
    
    # Filter out options and --subjects from positional args
    clean_args = []
    skip = False
    for i, arg in enumerate(args):
        if skip:
            skip = False
            continue
        if arg in ["--wavelet", "--lda", "--riemann"]:
            continue
        if arg == "--subjects":
            skip = True
            continue
        clean_args.append(arg)
        
    if len(clean_args) == 0:
        run_batch_evaluation(
            wavelet_bonus=wavelet_bonus,
            lda_bonus=lda_bonus,
            riemannian_bonus=riemannian_bonus,
            subjects=subjects
        )
        return
        
    if len(clean_args) >= 3:
        try:
            subject = int(clean_args[0])
            run = int(clean_args[1])
            action = clean_args[2].lower()
        except ValueError:
            print_usage()
            sys.exit(1)
            
        if action == "train":
            train_and_evaluate(
                subject, run,
                wavelet_bonus=wavelet_bonus,
                lda_bonus=lda_bonus,
                riemannian_bonus=riemannian_bonus
            )
        elif action == "predict":
            predict_stream(subject, run)
        else:
            print_usage()
            sys.exit(1)
    else:
        print_usage()
        sys.exit(1)

if __name__ == '__main__':
    main()
