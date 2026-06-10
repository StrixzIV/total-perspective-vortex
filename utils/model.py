import os
import pickle

def get_model_path(subject, run):
    os.makedirs('models', exist_ok=True)
    return f'models/subj{subject:03d}_run{run:02d}.pkl'

def save_model(model, subject, run):
    path = get_model_path(subject, run)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {path}')

def load_model(subject, run):
    path = get_model_path(subject, run)
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}. Please train a model first.')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
