import time
import numpy as np
from utils.model import load_model
from utils.preprocessing import load_and_preprocess
from utils.pipeline import get_run_pair

def predict_stream(subject, run):
    """
    Simulate real-time streaming prediction epoch by epoch.
    Pads predictions to ensure a <=2s real-time latency per epoch.
    """
    # Load model
    pipe = load_model(subject, run)
    
    # Map run to the experiment runs pair
    runs = get_run_pair(run)
    
    # Load and preprocess data
    print(f"Loading data for streaming prediction (subject {subject}, run {run})...")
    X, y = load_and_preprocess(subject, runs)
    
    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    
    for i, (epoch, label) in enumerate(zip(X, y)):
        t0 = time.time()
        
        # Add epoch batch dimension for sklearn predict
        pred = pipe.predict(epoch[np.newaxis, ...])[0]
        
        elapsed = time.time() - t0
        # Pad to simulate real-time gap
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
            
        match = (pred == label)
        correct += int(match)
        print(f"epoch {i:02d}: [{pred}] [{label}] {'True' if match else 'False'}")
        
    accuracy = correct / len(y) if len(y) > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f}")
