import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_samples=10000, fs=100, num_subcarriers=52):
    """
    Generate synthetic CSI data for 4 classes: Static, Walking, Sitting Down, Falling.
    Returns a DataFrame with columns: timestamp, label, I_0, Q_0, ..., I_51, Q_51
    """
    labels = ['Static', 'Walking', 'Sitting Down', 'Falling']
    
    data = []
    
    # Generate segments for each class
    segment_length = 200  # 2 seconds at 100Hz
    num_segments = num_samples // segment_length
    
    time = np.arange(num_samples) / fs
    
    current_idx = 0
    while current_idx < num_samples:
        label = np.random.choice(labels)
        end_idx = min(current_idx + segment_length, num_samples)
        seg_len = end_idx - current_idx
        
        t = time[current_idx:end_idx]
        
        # Base CSI (static channel)
        base_I = np.random.randn(num_subcarriers)
        base_Q = np.random.randn(num_subcarriers)
        
        I_data = np.tile(base_I, (seg_len, 1))
        Q_data = np.tile(base_Q, (seg_len, 1))
        
        # Add noise
        I_data += np.random.randn(seg_len, num_subcarriers) * 0.1
        Q_data += np.random.randn(seg_len, num_subcarriers) * 0.1
        
        # Add motion perturbation based on class
        if label == 'Static':
            pass # Just noise
        elif label == 'Walking':
            # Doppler shift around 10-20 Hz
            doppler = np.sin(2 * np.pi * 15 * t)[:, None]
            I_data += doppler * np.random.randn(num_subcarriers) * 0.5
            Q_data += doppler * np.random.randn(num_subcarriers) * 0.5
        elif label == 'Sitting Down':
            # Transient low frequency movement
            envelope = np.exp(-((t - t[seg_len//2])**2) / 0.1)[:, None]
            doppler = np.sin(2 * np.pi * 5 * t)[:, None]
            I_data += envelope * doppler * np.random.randn(num_subcarriers) * 1.0
            Q_data += envelope * doppler * np.random.randn(num_subcarriers) * 1.0
        elif label == 'Falling':
            # Sharp high frequency transient
            envelope = np.exp(-((t - t[seg_len//2])**2) / 0.05)[:, None]
            doppler = np.sin(2 * np.pi * 30 * t)[:, None]
            I_data += envelope * doppler * np.random.randn(num_subcarriers) * 2.0
            Q_data += envelope * doppler * np.random.randn(num_subcarriers) * 2.0
            
        for i in range(seg_len):
            row = [time[current_idx + i], label]
            for sc in range(num_subcarriers):
                row.append(I_data[i, sc])
                row.append(Q_data[i, sc])
            data.append(row)
            
        current_idx = end_idx
        
    cols = ['timestamp', 'label']
    for sc in range(num_subcarriers):
        cols.extend([f'I_{sc}', f'Q_{sc}'])
        
    df = pd.DataFrame(data, columns=cols)
    return df

if __name__ == '__main__':
    print("Generating synthetic CSI data...")
    df = generate_synthetic_data(num_samples=20000)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = os.path.join(models_dir, 'synthetic_csi.csv')
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
