"""
8-class dog behavior intent classifier.
Uses SVM + RandomForest ensemble with heuristic fallback.
Classes: aggression, fear, excitement, pain_distress, attention_seeking, play, alert_warning, greeting
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

BEHAVIOR_CLASSES = [
    "aggression",
    "fear",
    "excitement",
    "pain_distress",
    "attention_seeking",
    "play",
    "alert_warning",
    "greeting",
]

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

logger = logging.getLogger(__name__)
SVM_MODEL_PATH = MODELS_DIR / "svm_classifier.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_classifier.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"


@dataclass
class ClassificationResult:
    behavior_label: str
    confidence: float          # 0.0–1.0
    secondary_label: str
    secondary_confidence: float
    all_probabilities: Dict[str, float]
    feature_vector_dim: int
    model_used: str            # "ensemble" | "svm" | "rf" | "heuristic"
    has_wav2vec2: bool


class HeuristicClassifier:
    """
    Rule-based classifier using ethologically-grounded acoustic thresholds.
    Always available without training data.
    Based on Morton 1977 motivation-structural rules and Pongrácz et al. 2010.
    """

    def classify(self, features) -> Dict[str, float]:
        """Return probability-like scores for each class (not calibrated, just relative)."""
        scores = {c: 0.0 for c in BEHAVIOR_CLASSES}

        centroid = features.spectral_centroid
        zcr = features.zero_crossing_rate
        rms = features.rms_energy
        tempo = features.tempo
        bandwidth = features.spectral_bandwidth
        rolloff = features.spectral_rolloff

        # Aggression: low frequency, high energy, continuous, wide bandwidth
        if centroid < 800 and rms > 0.02 and bandwidth > 1500:
            scores["aggression"] += 0.5
        if zcr > 0.05 and centroid < 1000:
            scores["aggression"] += 0.2

        # Fear: high-pitched, low energy, narrow bandwidth (whimper/whine)
        if centroid > 1500 and rms < 0.02 and bandwidth < 1200:
            scores["fear"] += 0.45
        if rolloff > 4000 and rms < 0.015:
            scores["fear"] += 0.2

        # Excitement: high pitch, high tempo, high energy
        if centroid > 1200 and tempo > 140 and rms > 0.02:
            scores["excitement"] += 0.5
        if centroid > 1500 and rms > 0.03:
            scores["excitement"] += 0.2

        # Pain/distress: sustained high-pitch, mid energy, consistent
        if centroid > 1000 and 0.008 < rms < 0.04 and bandwidth < 1500:
            scores["pain_distress"] += 0.4
        if rolloff > 3000 and zcr < 0.03:
            scores["pain_distress"] += 0.2

        # Attention-seeking: medium pitch, regular/moderate tempo, medium energy
        if 700 < centroid < 1500 and 80 < tempo < 140 and 0.01 < rms < 0.04:
            scores["attention_seeking"] += 0.45
        if zcr < 0.04 and centroid < 1400:
            scores["attention_seeking"] += 0.15

        # Play: higher pitch, short staccato (high tempo), moderate energy
        if centroid > 1000 and tempo > 120 and 0.015 < rms < 0.05 and zcr > 0.03:
            scores["play"] += 0.45
        if bandwidth > 1200 and centroid > 900 and tempo > 100:
            scores["play"] += 0.15

        # Alert/warning: single/few deep barks, low-mid pitch, high initial energy
        if centroid < 1000 and rms > 0.025 and tempo < 100:
            scores["alert_warning"] += 0.5
        if centroid < 800 and zcr < 0.04:
            scores["alert_warning"] += 0.2

        # Greeting: mixed pitch, moderate energy, varied
        if 600 < centroid < 1800 and 0.01 < rms < 0.04 and 80 < tempo < 160:
            scores["greeting"] += 0.35
        # Greeting by residual: if nothing else stands out
        total = sum(scores.values())
        if total < 0.3:
            scores["greeting"] += 0.25

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            # Uniform fallback
            scores = {k: 1.0 / len(BEHAVIOR_CLASSES) for k in BEHAVIOR_CLASSES}

        return scores


class TrainedClassifier:
    """Wrapper for scikit-learn SVM or RandomForest classifier."""

    def __init__(self, model_path: Path, scaler_path: Path):
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._model = None
        self._scaler = None

    def is_available(self) -> bool:
        return self._model_path.exists() and self._scaler_path.exists()

    def _load(self):
        if self._model is None:
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
            with open(self._scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

    def classify(self, feature_vector: np.ndarray) -> Dict[str, float]:
        self._load()
        x = self._scaler.transform(feature_vector.reshape(1, -1))
        probas = self._model.predict_proba(x)[0]
        classes = self._model.classes_
        return {str(c): float(p) for c, p in zip(classes, probas)}

    def train(self, X: np.ndarray, y: List[str], model_type: str = "svm"):
        """Train classifier and save to disk."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "svm":
            clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        clf.fit(X_scaled, y)

        with open(self._model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(self._scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        self._model = clf
        self._scaler = scaler


class BehaviorClassifier:
    """
    Ensemble behavior classifier:
    1. HeuristicClassifier (always available)
    2. SVMClassifier (if trained model exists)
    3. RandomForestClassifier (if trained model exists)
    Votes by averaging probability scores.
    """

    def __init__(self):
        self._heuristic = HeuristicClassifier()
        self._svm = TrainedClassifier(SVM_MODEL_PATH, SCALER_PATH)
        rf_scaler = MODELS_DIR / "rf_scaler.pkl"
        self._rf = TrainedClassifier(RF_MODEL_PATH, rf_scaler)

    def classify(self, audio_features, visual_features=None) -> ClassificationResult:
        """
        Classify behavior from AudioFeatures (+ optional VisualFeatures).
        Returns ClassificationResult with primary label, confidence, and all probabilities.
        """
        # Choose feature vector based on wav2vec2 availability
        has_w2v = audio_features.wav2vec2_embedding is not None
        feature_vec = audio_features.to_full_vector() if has_w2v else audio_features.to_vector()

        # Optional visual feature augmentation
        if visual_features is not None:
            vis_vec = np.array([
                visual_features.posture_score,
                visual_features.motion_intensity,
                visual_features.estimated_head_position,
            ])
            feature_vec = np.concatenate([feature_vec, vis_vec])

        all_scores = {}
        models_used = []

        # Always run heuristic
        heuristic_scores = self._heuristic.classify(audio_features)
        for k, v in heuristic_scores.items():
            all_scores[k] = all_scores.get(k, 0.0) + v
        models_used.append("heuristic")

        # Run SVM if trained
        svm_weight = 0.0
        if self._svm.is_available():
            try:
                svm_scores = self._svm.classify(feature_vec[:138])  # MFCC-only for SVM
                for k, v in svm_scores.items():
                    all_scores[k] = all_scores.get(k, 0.0) + v * 2.0
                svm_weight = 2.0
                models_used.append("svm")
            except Exception as exc:
                logger.debug("SVM classifier inference failed: %s", exc)

        # Run RF if trained
        rf_weight = 0.0
        if self._rf.is_available():
            try:
                rf_scores = self._rf.classify(feature_vec[:138])
                for k, v in rf_scores.items():
                    all_scores[k] = all_scores.get(k, 0.0) + v * 1.5
                rf_weight = 1.5
                models_used.append("rf")
            except Exception as exc:
                logger.debug("RandomForest classifier inference failed: %s", exc)

        # Normalize
        total = sum(all_scores.values()) or 1.0
        normalized = {k: v / total for k, v in all_scores.items()}

        # Sort by score
        sorted_scores = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        primary_label, primary_conf = sorted_scores[0]
        secondary_label, secondary_conf = sorted_scores[1] if len(sorted_scores) > 1 else ("uncertain", 0.0)

        # If confidence too low, return uncertain
        if primary_conf < 0.30:
            primary_label = "uncertain"

        model_str = "+".join(models_used)
        return ClassificationResult(
            behavior_label=primary_label,
            confidence=round(primary_conf, 3),
            secondary_label=secondary_label,
            secondary_confidence=round(secondary_conf, 3),
            all_probabilities={k: round(v, 4) for k, v in normalized.items()},
            feature_vector_dim=len(feature_vec),
            model_used=model_str,
            has_wav2vec2=has_w2v,
        )

    def train_from_directory(self, audio_dir: str, label_map: Dict[str, str], model_type: str = "svm"):
        """
        Train classifier from labeled audio files.
        label_map: {filename_without_ext: behavior_class}
        """
        from agent.modules.audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        X, y = [], []
        audio_dir = Path(audio_dir)

        for fname, label in label_map.items():
            for ext in [".wav", ".mp3", ".ogg", ".flac"]:
                fpath = audio_dir / (fname + ext)
                if fpath.exists():
                    try:
                        feats = analyzer.analyze_file(str(fpath), use_wav2vec2=False)
                        X.append(feats.to_vector())
                        y.append(label)
                    except Exception as exc:
                        logger.debug("Failed to extract features from %s: %s", fpath, exc)
                    break

        if len(X) < 5:
            raise ValueError(f"Need at least 5 labeled samples, found {len(X)}")

        X_arr = np.array(X)
        clf = self._svm if model_type == "svm" else self._rf
        clf.train(X_arr, y, model_type=model_type)
        return len(X)

    def get_behavior_description(self, label: str) -> str:
        """Return a brief ethological description for a behavior class."""
        descriptions = {
            "aggression": "Territorial or resource-guarding behavior. Low-pitched, harsh vocalizations indicating threat perception.",
            "fear": "Appeasement or distress signaling. High-pitched, softer sounds indicating discomfort or threat avoidance.",
            "excitement": "Positive arousal state. Rapid, high-pitched vocalizations associated with anticipation or play.",
            "pain_distress": "Nociceptive or separation-anxiety vocalization. Sustained, high-pitched sounds requiring immediate attention.",
            "attention_seeking": "Learned instrumental behavior. Regular, medium-pitched vocalizations to solicit owner interaction.",
            "play": "Social play invitation. Higher-pitched, staccato vocalizations in a relaxed, affiliative context.",
            "alert_warning": "Sentinel function. Single or few deep barks signaling a novel stimulus in the environment.",
            "greeting": "Affiliative social behavior. Mixed, moderate vocalizations accompanying reunion with familiar individuals.",
            "uncertain": "Insufficient confidence to classify. Multiple behavioral signals may be present simultaneously.",
        }
        return descriptions.get(label, "Unknown behavior class.")
