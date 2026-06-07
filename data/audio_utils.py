"""Shared librosa mel + Griffin-Lim (train and synth must match)."""
from __future__ import annotations
import numpy as np

def mel_spectrogram(wav: np.ndarray, cfg: dict) -> np.ndarray:
    import librosa
    a = cfg["audio"]
    mel = librosa.feature.melspectrogram(
        y=wav, sr=a["sample_rate"], n_fft=a["n_fft"],
        hop_length=a["hop_length"], win_length=a["win_length"],
        n_mels=a["n_mels"], fmin=a["fmin"], fmax=a["fmax"], power=1.0,
    )
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)

def mel_to_wav(mel: np.ndarray, cfg: dict, n_iters: int = 64) -> np.ndarray:
    import librosa
    a = cfg["audio"]
    stft = librosa.feature.inverse.mel_to_stft(
        np.exp(mel), sr=a["sample_rate"], n_fft=a["n_fft"],
        fmin=a["fmin"], fmax=a["fmax"], power=1.0,
    )
    return librosa.griffinlim(
        stft, n_iter=n_iters, hop_length=a["hop_length"],
        win_length=a["win_length"], n_fft=a["n_fft"],
    )


