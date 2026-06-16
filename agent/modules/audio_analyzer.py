"""
Audio feature extraction for dog vocalization analysis.
Extracts MFCC, chroma, spectral statistics, and optional wav2vec2 embeddings.
"""

import os
import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MFCC_COEFFICIENTS = 40
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128


@dataclass
class AudioFeatures:
    mfcc: np.ndarray          # shape (40,)
    mfcc_delta: np.ndarray    # shape (40,)
    mfcc_delta2: np.ndarray   # shape (40,)
    chroma: np.ndarray        # shape (12,)
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float
    rms_energy: float
    tempo: float
    wav2vec2_embedding: Optional[np.ndarray]  # shape (768,) if available
    duration_seconds: float
    sample_rate: int
    source: str  # "microphone" | "file"

    def to_vector(self) -> np.ndarray:
        """Concatenate all handcrafted features into a single flat vector (138-dim)."""
        scalar_feats = np.array([
            self.spectral_centroid / 8000.0,   # normalize to ~[0,1]
            self.spectral_bandwidth / 4000.0,
            self.spectral_rolloff / 8000.0,
            self.zero_crossing_rate * 100.0,
            self.rms_energy * 100.0,
            self.tempo / 200.0,
        ])
        return np.concatenate([
            self.mfcc,
            self.mfcc_delta,
            self.mfcc_delta2,
            self.chroma,
            scalar_feats,
        ])  # 40+40+40+12+6 = 138

    def to_full_vector(self) -> np.ndarray:
        """138-dim MFCC vector + optional 768-dim wav2vec2 (906-dim total)."""
        base = self.to_vector()
        if self.wav2vec2_embedding is not None:
            return np.concatenate([base, self.wav2vec2_embedding])
        return base


class AudioAnalyzer:
    """Captures or loads dog audio and extracts acoustic features."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._hf_manager = None

    def _get_hf(self):
        if self._hf_manager is None:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from tools.hf_model_manager import HFModelManager
            self._hf_manager = HFModelManager.instance()
        return self._hf_manager

    def load_from_file(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate. Returns float32 array."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def record_from_microphone(self, duration: float = 3.0) -> np.ndarray:
        """Capture audio from default microphone. Returns float32 array."""
        try:
            import sounddevice as sd
            frames = int(duration * self.sample_rate)
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return audio.flatten()
        except Exception as exc:
            raise RuntimeError(f"Microphone capture failed: {exc}") from exc

    def extract_features(self, audio: np.ndarray, use_wav2vec2: bool = True) -> AudioFeatures:
        """Extract full acoustic feature set from a raw float32 audio array."""
        sr = self.sample_rate

        # MFCC + deltas
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=MFCC_COEFFICIENTS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)
        mfcc_delta2_mean = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)

        # Chroma
        chroma = np.mean(
            librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH),
            axis=1,
        )

        # Spectral statistics
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        spec_rolloff = float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85))
        )
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        rms = float(np.mean(librosa.feature.rms(y=audio)))

        # Tempo
        tempo_raw, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(np.asarray(tempo_raw).flat[0])

        # wav2vec2 embedding (optional, graceful fallback)
        wav2vec2_emb = None
        if use_wav2vec2:
            try:
                wav2vec2_emb = self._get_hf().extract_wav2vec2(audio, sr)
            except Exception as exc:
                logger.debug("wav2vec2 extraction failed, using MFCC fallback: %s", exc)

        return AudioFeatures(
            mfcc=mfcc_mean,
            mfcc_delta=mfcc_delta_mean,
            mfcc_delta2=mfcc_delta2_mean,
            chroma=chroma,
            spectral_centroid=spec_centroid,
            spectral_bandwidth=spec_bandwidth,
            spectral_rolloff=spec_rolloff,
            zero_crossing_rate=zcr,
            rms_energy=rms,
            tempo=tempo,
            wav2vec2_embedding=wav2vec2_emb,
            duration_seconds=len(audio) / sr,
            sample_rate=sr,
            source="unknown",
        )

    def analyze_file(self, file_path: str, use_wav2vec2: bool = True) -> AudioFeatures:
        audio = self.load_from_file(file_path)
        feats = self.extract_features(audio, use_wav2vec2=use_wav2vec2)
        feats.source = "file"
        return feats

    def analyze_microphone(self, duration: float = 3.0, use_wav2vec2: bool = True) -> AudioFeatures:
        audio = self.record_from_microphone(duration)
        feats = self.extract_features(audio, use_wav2vec2=use_wav2vec2)
        feats.source = "microphone"
        return feats

    def validate_audio(self, features: AudioFeatures) -> Tuple[bool, str]:
        """
        Validate audio quality gates.
        Returns (is_valid, reason).
        Gate 1: non-silent (RMS > 0.001)
        Gate 2: minimum duration (> 0.1s)
        Gate 3: plausible vocalization frequency range
        """
        if features.rms_energy < 0.001:
            return False, "Audio appears to be silent (RMS < 0.001)"
        if features.duration_seconds < 0.1:
            return False, f"Audio too short: {features.duration_seconds:.2f}s (minimum 0.1s)"
        if not (50 < features.spectral_centroid < 8000):
            return False, (
                f"Spectral centroid {features.spectral_centroid:.0f}Hz outside "
                "expected range for dog vocalizations (50–8000 Hz)"
            )
        return True, "ok"

    def is_likely_dog_audio(self, features: AudioFeatures) -> Tuple[bool, float]:
        """
        Heuristic check: is the audio likely a dog vocalization?
        Returns (likely_dog, confidence_score 0.0–1.0).
        Based on acoustic signatures of typical dog vocalizations.
        """
        score = 0.0
        # Dog vocalizations: typically 100–4000 Hz fundamental frequency
        if 100 < features.spectral_centroid < 4000:
            score += 0.25
        # Barks have moderate-to-high ZCR
        if features.zero_crossing_rate > 0.02:
            score += 0.20
        # Non-trivial energy
        if features.rms_energy > 0.005:
            score += 0.20
        # Duration > 0.1s
        if features.duration_seconds > 0.1:
            score += 0.20
        # Not extremely high spectral bandwidth (human speech artifact)
        if features.spectral_bandwidth < 3000:
            score += 0.15
        return score >= 0.55, round(score, 2)
