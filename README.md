# Wi-Fi CSI Human Sensing Pipeline

[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![Electron: Native](https://img.shields.io/badge/Electron-Desktop-cyan.svg)]()
[![Backend: PyTorch](https://img.shields.io/badge/Backend-PyTorch-ee4c2c.svg)]()
[![Dashboard: Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-blueviolet.svg)]()

A high-performance, real-time Human Sensing interface leveraging Wi-Fi Channel State Information (CSI). Utilizing an ESP32-S3 microcontroller as a dedicated ingestion node streaming 52-subcarrier complex per-packet IQ datasets, this Python and Electron hybrid desktop application natively infers activities across deterministic spatial timelines.

<img src="assets/demo.gif" alt="Real-Time Wi-Fi CSI Activity Recognition Demo"/><br>


---

## 🚀 Key Features

*   **Offline Data Generator**: Construct synthetic data sequences locally to replicate deterministic motion behaviors for purely offline development environments. 
*   **Dimensionality Processing**: Automated outlier removals (Hampel Filtering), sanitization, extraction processing, dimensionality deduction (Principal Components Analysis), and frequency analysis (STFT) converting a massive 52-carrier tensor vector sequentially down to a readable 10-component tracking scope across continuous 2-second windows.
*   **Real-Time Dashboard Rendering**: Electron-embedded interface updating Plotly traces for Spectrograms, Amplitude Heatmaps, and 3D PCA Tracking.

---

## 📐 Pipeline Architecture Layers

<img src="assets/arch_diagram.png" alt="Wi-Fi CSI Sensing Architecture Graphic"/>

| Layer | Component Action | Functionality / Detail |
| :--- | :--- | :--- |
| **Ingestion** | Extractor (`app.py`, `pyserial`) | Live deserialization from hardware Serial (ESP32) vs offline mock-data interpolation via Playback. |
| **Processing** | Sanitization & Aggregation (`processing.py`) | Iterative operations: applies Hampel Filters, handles matrix conjugations, and STFT computations over 200 frequency segments. |
| **Inference Inference** | Long-Short Term Memory Analysis (`ml_model.py`) | Employs PyTorch's backend parameter tensors `csi_model.pth` matched strictly into 4 classes mapping probabilities: `[Unknown, Walking, Static, Falling]`. |
| **Front-End Visualizer** | App Wrapper (`index.js`, `Dash`) | Desktop instance containing interactive feedback variables such as live component tracking via `Electron`. |

---

## 📋 Requirements

*   **ESP32-S3 Firmware**: Operable with `esp32-csi-tool`.
*   **Python**: Version 3.8+ context dependencies (PyTorch, Dash, scikit-learn).
*   **NPM Frameworks**: Electron.JS engine wrapper for Chromium.

---

## 📥 Building & Installation

Ensure you have your global NPM package manager and Python configured correctly contextually.

### 1. Repository Setup

```bash
# 1. Install Node modules
npm install

# 2. Install Python core utilities
pip install -r requirements.txt
```

### 2. Mocking & Tracing Model Files (Optional Offline Generative Workloads)

Generating mock variables will place resulting parameter values internally inside `models/`. For more details view its [internal structure documentation](models/README.md).

```bash
# 1. Create a 20,000 instance Mock CSV
python src/generate_synthetic_csi.py

# 2. Compile Model Layer Logic & Save Data Dict parameters
python src/ml_model.py
```

### 3. Executable Run Command

```bash
# Deploys Python backend daemon iteratively passing over local :8050 to UI
npm start
```
---

## 📖 License & Legal

Distributed under the MIT License. See `LICENSE` for more information.
