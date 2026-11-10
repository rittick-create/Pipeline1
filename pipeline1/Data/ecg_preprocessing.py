"""Filtering methods copied from the original ECG extraction notebook."""
import numpy as np
from scipy.signal import medfilt, iirnotch, filtfilt, butter

class ECGPreprocessor:
    def z_score_normalization(self,signal):
            return (signal - np.mean(signal)) / (np.std(signal) +1e-8)

    def filter_bandpass(self, signal, fs):
            """
            Bandpass filter
            :param signal: 2D numpy array of shape (channels, time)
            :param fs: sampling frequency
            :return: filtered signal
            """
            # Remove power-line interference
            b, a = iirnotch(50, 30, fs)
            filtered_signal = np.zeros_like(signal)
            for c in range(signal.shape[0]):
                filtered_signal[c] = filtfilt(b, a, signal[c])
    
            # Simple bandpass filter
            b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
            for c in range(signal.shape[0]):
                filtered_signal[c] = filtfilt(b, a, filtered_signal[c])
    
            # Remove baseline wander
            baseline = np.zeros_like(filtered_signal)
            for c in range(filtered_signal.shape[0]):
                kernel_size = int(0.4 * fs) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure kernel size is odd
                baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
            filter_ecg = filtered_signal - baseline
    
            return filter_ecg
