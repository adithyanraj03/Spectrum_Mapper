import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
import argparse
import serial
import torch
import torch.nn.functional as F
from collections import deque

from processing import CSIProcessor
from ml_model import CSINet

# Configuration
WINDOW_SIZE = 200
NUM_SUBCARRIERS = 52
UPDATE_INTERVAL = 500 # ms

# Global Data Structures
data_lock = threading.Lock()
raw_buffer_I = deque(maxlen=WINDOW_SIZE * 2)
raw_buffer_Q = deque(maxlen=WINDOW_SIZE * 2)
timestamps = deque(maxlen=WINDOW_SIZE * 2)

# Global Data Structures
data_lock = threading.Lock()
raw_buffer_I = deque(maxlen=WINDOW_SIZE * 2)
raw_buffer_Q = deque(maxlen=WINDOW_SIZE * 2)
timestamps = deque(maxlen=WINDOW_SIZE * 2)
history_predictions = deque(maxlen=100) # Store last 100 predictions
history_confidences = deque(maxlen=100)

latest_results = {
    'amplitude': np.zeros((WINDOW_SIZE, NUM_SUBCARRIERS)),
    'Zxx': np.zeros((33, 5)), # STFT approx shape
    'pca_features': np.zeros((WINDOW_SIZE, 10)),
    'prediction': 'Unknown',
    'confidence': 0.0,
    'class_probs': np.zeros(4),
    'raw_I_mid': np.zeros(WINDOW_SIZE),
    'raw_Q_mid': np.zeros(WINDOW_SIZE)
}

classes = ['Falling', 'Sitting Down', 'Static', 'Walking']

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Try to load model
try:
    classes_path = os.path.join(parent_dir, 'models', 'classes.npy')
    model_path = os.path.join(parent_dir, 'models', 'csi_model.pth')
    
    classes = np.load(classes_path, allow_pickle=True)
    # The actual input dim from processing is PCA flattened (2000) + STFT features (33*5=165)
    # But because our test data generated has length 200 and using `64 nperseg` for stft causes dim changes 
    # The true output from np.concatenate([pca_features.flatten(), spec_features]) in process_window
    input_dim = 2165 # Assumed baseline, replace dynamically if needed but we trained dynamically
    # Since train_model used X.shape[1], let's load it dynamically if we saved state dict, state dict keys hint size:
    state_dict = torch.load(model_path, weights_only=True)
    input_dim = state_dict['fc1.weight'].shape[1]
    
    model = CSINet(input_dim=input_dim, num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not load model, using random predictions. Error: {e}")
    model = None

processor = CSIProcessor(window_size=WINDOW_SIZE, num_subcarriers=NUM_SUBCARRIERS)

def ingestion_worker(replay_file=None, port=None, baudrate=115200):
    global latest_results
    
    if replay_file:
        print(f"Replaying from {replay_file}")
        df = pd.read_csv(replay_file)
        
        while True:
            for i in range(len(df)):
                row = df.iloc[i]
                
                with data_lock:
                    I_data = [row[f'I_{sc}'] for sc in range(NUM_SUBCARRIERS)]
                    Q_data = [row[f'Q_{sc}'] for sc in range(NUM_SUBCARRIERS)]
                    timestamps.append(row['timestamp'])
                    raw_buffer_I.append(I_data)
                    raw_buffer_Q.append(Q_data)
                    
                    if len(raw_buffer_I) >= WINDOW_SIZE:
                        # Process window
                        np_I = np.array(list(raw_buffer_I)[-WINDOW_SIZE:])
                        np_Q = np.array(list(raw_buffer_Q)[-WINDOW_SIZE:])
                        
                        try:
                            features, amp, Zxx, pca_feat = processor.process_window(np_I, np_Q)
                            
                            # Inference
                            if model:
                                with torch.no_grad():
                                    t_features = torch.FloatTensor(features).unsqueeze(0)
                                    output = model(t_features)
                                    probs = F.softmax(output, dim=1).numpy()[0]
                                    pred_idx = np.argmax(probs)
                                    latest_results['prediction'] = classes[pred_idx]
                                    latest_results['confidence'] = probs[pred_idx]
                                    latest_results['class_probs'] = probs
                                    
                                    history_predictions.append(classes[pred_idx])
                                    history_confidences.append(probs[pred_idx])
                            
                            latest_results['amplitude'] = amp
                            latest_results['Zxx'] = np.abs(Zxx)
                            latest_results['pca_features'] = pca_feat
                            latest_results['raw_I_mid'] = np_I[:, NUM_SUBCARRIERS//2]
                            latest_results['raw_Q_mid'] = np_Q[:, NUM_SUBCARRIERS//2]
                            
                        except Exception as e:
                            print(f"Processing error: {e}")
                            
                time.sleep(0.01) # Simulate 100Hz
                
    elif port:
        print(f"Listening on serial port {port} at {baudrate}")
        try:
            ser = serial.Serial(port, baudrate)
            while True:
                line = ser.readline().decode('utf-8').strip()
                # Parse logic for actual ESP32 CSI packets goes here
                # Expected format: CSV of I/Q pairs or similar
        except Exception as e:
            print(f"Serial error: {e}")

# Dash App
app = dash.Dash(__name__, title="Wi-Fi CSI Human Sensing")

app.layout = html.Div([
    html.Div([
        html.H3(id='prediction-text', style={'textAlign': 'center'}),
        dcc.Graph(id='confidence-gauge', style={'height': '250px'})
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    html.Div([
        dcc.Graph(id='amplitude-heatmap')
    ], style={'width': '65%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='doppler-spectrogram')
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='pca-scatter')
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='iq-trace')
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='history-distribution')
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL, n_intervals=0)
])

@app.callback(
    [Output('prediction-text', 'children'),
     Output('confidence-gauge', 'figure'),
     Output('amplitude-heatmap', 'figure'),
     Output('doppler-spectrogram', 'figure'),
     Output('pca-scatter', 'figure'),
     Output('iq-trace', 'figure'),
     Output('history-distribution', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    with data_lock:
        amp = latest_results['amplitude'].copy()
        Zxx = latest_results['Zxx'].copy()
        pca = latest_results['pca_features'].copy()
        pred = latest_results['prediction']
        conf = latest_results['confidence']
        raw_I = latest_results['raw_I_mid'].copy()
        raw_Q = latest_results['raw_Q_mid'].copy()
        hist_preds = list(history_predictions)
        
    pred_text = f"Activity: {pred}"
    
    gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = conf * 100,
        title = {'text': "Confidence (%)"},
        gauge = {'axis': {'range': [0, 100]}}
    ))
    gauge.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    
    heatmap = go.Figure(data=go.Heatmap(
        z=amp.T,
        colorscale='Viridis'
    ))
    heatmap.update_layout(title="Amplitude Heatmap (Subcarriers vs Time)", margin=dict(l=40, r=20, t=40, b=30))
    
    spectrogram = go.Figure(data=go.Heatmap(
        z=Zxx,
        colorscale='Jet'
    ))
    spectrogram.update_layout(title="Doppler Spectrogram (STFT of PCA 1)", margin=dict(l=40, r=20, t=40, b=30))
    
    scatter = go.Figure(data=go.Scatter3d(
        x=pca[:, 0],
        y=pca[:, 1],
        z=pca[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color=np.arange(len(pca)), colorscale='Viridis', opacity=0.8)
    ))
    scatter.update_layout(title="3D PCA Trajectory", margin=dict(l=0, r=0, t=40, b=0))
    
    # I/Q Trace
    iq_fig = go.Figure()
    iq_fig.add_trace(go.Scatter(y=raw_I, mode='lines', name='I (In-phase)'))
    iq_fig.add_trace(go.Scatter(y=raw_Q, mode='lines', name='Q (Quadrature)'))
    iq_fig.update_layout(title="Raw I/Q Data (Middle Subcarrier)", margin=dict(l=40, r=20, t=40, b=30))
    
    # History Distribution
    hist_counts = {c: hist_preds.count(c) for c in classes} if hist_preds else {c: 0 for c in classes}
    hist_fig = go.Figure([go.Bar(x=list(hist_counts.keys()), y=list(hist_counts.values()))])
    hist_fig.update_layout(title="Recent Predictions Distribution (Last 100)", margin=dict(l=40, r=20, t=40, b=30))
    
    return pred_text, gauge, heatmap, spectrogram, scatter, iq_fig, hist_fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=str, help='Path to synthetic_csi.csv for replay mode')
    parser.add_argument('--port', type=str, help='Serial port for ESP32')
    args = parser.parse_args()
    
    # Start ingestion thread
    t = threading.Thread(target=ingestion_worker, args=(args.replay, args.port), daemon=True)
    t.start()
    
    app.run(debug=False, port=8050)
