# Core Model Files

This directory is an output artifact repository for the mock dataset files (`synthetic_csi.csv`) and the serialized state dictionaries & mapping classes required to initiate the ML model (`csi_model.pth` and `classes.npy`).

## Generating Local Resources
To re-generate the mock testing sets, switch to your root folder or standard working environment and run:
`python src/generate_synthetic_csi.py`

This will programmatically save a 20,000 instance CSV containing `Static`, `Walking`, `Sitting Down`, and `Falling` states and raw IQ complex representations within this `models/` folder.  

## Training an Artifact
With the data accessible inside this `models/` directory, simply run:
`python src/ml_model.py`

This computes dimensionality reduction preprocessing (PCA & STFT inputs) on Windows containing slices of 200 components matching roughly ~100Hz frequency per 2 seconds. The script will dynamically infer length layers and save `classes.npy` parameter vectors onto `csi_model.pth`.

The `app.py` script accesses this path statically when booting via Electron. Ensure everything executes in root contexts.