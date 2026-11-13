"""Adapt SYMILE's stored ECGs to ECGFounder without another min-max scaling.

Filters follow ECGFounder/util.py (MIT; see Models/ECGFounder_LICENSE).
The lead order below was checked against the source headers for 21 records.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


SOURCE_LEADS = ("I", "II", "III", "aVR", "aVF", "aVL",
                "V1", "V2", "V3", "V4", "V5", "V6")
TARGET_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6")
LEAD_ORDER = [SOURCE_LEADS.index(lead) for lead in TARGET_LEADS]


def preprocess_ecg(ecg):
    """Return float32 (12, 5000) from one SYMILE (1, 5000, 12) record.

    These arrays already contain 10 seconds at 500 Hz. Work in float64 for
    filtering, then normalize across the entire record, as ECGFounder does.
    """
    ecg = np.asarray(ecg)
    if ecg.shape != (1, 5000, 12):
        raise ValueError("Expected one SYMILE ECG shaped (1, 5000, 12).")
    signal = ecg[0][:, LEAD_ORDER].T.astype(np.float64, copy=True)
    if not np.isfinite(signal).all():
        raise ValueError("ECG contains non-finite values.")

    b, a = iirnotch(50, 30, fs=500)
    filtered = np.empty_like(signal)
    for lead in range(12):
        filtered[lead] = filtfilt(b, a, signal[lead])

    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=500)
    for lead in range(12):
        filtered[lead] = filtfilt(b, a, filtered[lead])

    baseline = np.empty_like(filtered)
    for lead in range(12):
        baseline[lead] = medfilt(filtered[lead], kernel_size=201)
    filtered -= baseline

    std = filtered.std()
    if not np.isfinite(filtered).all() or std == 0:
        raise ValueError("ECG has no usable variation after filtering.")
    normalized = (filtered - filtered.mean()) / (std + 1e-8)
    return np.ascontiguousarray(normalized, dtype=np.float32)
