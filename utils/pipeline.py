from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from utils.csp import CSP
from utils.model import save_model, get_model_path
from utils.preprocessing import load_and_preprocess, ChannelScaler, DWTFeatureExtractor
from utils.lda import CustomLDA
from utils.riemann import RiemannianMDM

def get_run_pair(run):
    """
    Map an input run number to the pair of runs corresponding to its experiment.
    
    Runs 3,4   -> Exp 0
    Runs 5,6   -> Exp 1
    Runs 7,8   -> Exp 2
    Runs 9,10  -> Exp 3
    Runs 11,12 -> Exp 4
    Runs 13,14 -> Exp 5
    """
    for pair in [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]:
        if run in pair:
            return pair
    raise ValueError(f"Run {run} is not a valid motor imagery/execution run (3-14).")

def build_pipeline(n_components=4, wavelet_bonus=False, lda_bonus=False, riemannian_bonus=False):
    if riemannian_bonus:
        return Pipeline([
            ('scaler', ChannelScaler()),
            ('mdm', RiemannianMDM()),
        ])

    clf = CustomLDA() if lda_bonus else LinearDiscriminantAnalysis()

    if wavelet_bonus:
        return Pipeline([
            ('scaler', ChannelScaler()),
            ('csp', CSP(n_components=n_components, transform_into='signals')),
            ('dwt', DWTFeatureExtractor(wavelet='db4', level=4)),
            ('clf', clf),
        ])

    return Pipeline([
        ('scaler', ChannelScaler()),
        ('csp', CSP(n_components=n_components, transform_into='features')),
        ('clf', clf),
    ])

def train_and_evaluate(subject, run, wavelet_bonus=False, lda_bonus=False, riemannian_bonus=False):
    """
    Load data, split into train/test, cross-validate on train, fit final,
    save the model, and report test accuracy.
    """
    # Map run to the experiment runs pair
    runs = get_run_pair(run)
    
    # Load data
    print(f"Loading data for subject {subject}, runs {runs}...")
    X, y = load_and_preprocess(subject, runs)
    
    # Stratified split: 80% train, 20% test (strict anti-overfitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True
    )
    
    # Build pipeline
    pipe = build_pipeline(
        n_components=4,
        wavelet_bonus=wavelet_bonus,
        lda_bonus=lda_bonus,
        riemannian_bonus=riemannian_bonus
    )
    
    # 10-fold cross validation on training set
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv)
    
    # Output scores in specified format
    scores_str = " ".join([f"{s:.4f}" for s in scores])
    print(scores_str)
    print(f"cross_val_score: {scores.mean():.4f}")
    
    # Fit on the full training portion
    pipe.fit(X_train, y_train)
    
    # Save the fitted pipeline
    save_model(pipe, subject, run)
    
    # Print held-out test score for verification
    test_score = pipe.score(X_test, y_test)
    print(f"Held-out test split accuracy: {test_score:.4f}")
    
    return test_score
