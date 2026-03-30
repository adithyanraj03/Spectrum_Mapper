import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA

class CSIProcessor:
    def __init__(self, num_subcarriers=52, window_size=200):
        self.num_subcarriers = num_subcarriers
        self.window_size = window_size
        self.pca = PCA(n_components=10)
        self.pca_fitted = False
        
    def hampel_filter(self, data, window_size=5, n_sigmas=3):
        n = len(data)
        new_data = data.copy()
        k = 1.4826 
        
        for i in range((window_size),(n - window_size)):
            x0 = np.median(data[(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(data[(i - window_size):(i + window_size)] - x0))
            if (np.abs(data[i] - x0) > n_sigmas * S0):
                new_data[i] = x0
        return new_data

    def process_window(self, i_data, q_data):
        """
        Processes a window of CSI data (shape: window_size x num_subcarriers)
        Returns flattened feature vector
        """
        # Complex representation
        csi_complex = i_data + 1j * q_data
        
        # Phase sanitization via conjugate multiplication
        sanitized = np.zeros_like(csi_complex)
        sanitized[:, 0] = csi_complex[:, 0]
        for i in range(1, self.num_subcarriers):
            sanitized[:, i] = csi_complex[:, i] * np.conj(csi_complex[:, i-1])
            
        amplitude = np.abs(sanitized)
        
        # Outlier removal per subcarrier
        for i in range(self.num_subcarriers):
            amplitude[:, i] = self.hampel_filter(amplitude[:, i])
            
        # PCA
        if not self.pca_fitted:
            self.pca.fit(amplitude)
            self.pca_fitted = True
            
        pca_features = self.pca.transform(amplitude)
        
        # Spectrogram features (STFT on first PCA component)
        f, t, Zxx = signal.stft(pca_features[:, 0], fs=100, nperseg=64)
        spec_features = np.abs(Zxx).flatten()
        
        # Combine features
        features = np.concatenate([
            pca_features.flatten(),
            spec_features
        ])
        
        return features, amplitude, Zxx, pca_features