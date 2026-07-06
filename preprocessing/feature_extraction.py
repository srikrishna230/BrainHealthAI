import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# -----------------------------
# Hjorth Parameters
# -----------------------------
def hjorth_parameters(signal):
    signal = np.asarray(signal)

    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    activity = np.var(signal)
    mobility = np.sqrt(np.var(first_deriv) / activity) if activity != 0 else 0

    mobility_deriv = np.sqrt(
        np.var(second_deriv) / np.var(first_deriv)
    ) if np.var(first_deriv) != 0 else 0

    complexity = (
        mobility_deriv / mobility if mobility != 0 else 0
    )

    return activity, mobility, complexity

# -----------------------------
# Band Power
# -----------------------------
def bandpower(freqs, psd, low, high):
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapezoid(psd[idx], freqs[idx])

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(signal, fs=250):

    signal = np.asarray(signal)

    freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))

    delta = bandpower(freqs, psd, 0.5, 4)
    theta = bandpower(freqs, psd, 4, 8)
    alpha = bandpower(freqs, psd, 8, 13)
    beta = bandpower(freqs, psd, 13, 30)
    gamma = bandpower(freqs, psd, 30, 45)

    activity, mobility, complexity = hjorth_parameters(signal)

    mean = np.mean(signal)
    std = np.std(signal)
    variance = np.var(signal)
    rms = np.sqrt(np.mean(signal**2))
    sk = skew(signal)
    kt = kurtosis(signal)

    hist = np.histogram(signal, bins=20)[0]
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))

    return np.array([
        delta,
        theta,
        alpha,
        beta,
        gamma,
        activity,
        mobility,
        complexity,
        mean,
        std,
        variance,
        rms,
        sk,
        kt,
        entropy
    ])