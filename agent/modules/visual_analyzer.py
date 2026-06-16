"""
Optional MediaPipe-based body language feature extraction for dogs.
MediaPipe Pose is trained on humans; we extract proxy posture features
that are still informative even without dog-specific landmarks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    posture_score: float        # 0.0 (cowering/low) to 1.0 (erect/alert)
    motion_intensity: float     # 0.0–1.0 from frame diff or landmark velocity
    estimated_head_position: float  # 0.0 (low/submissive) to 1.0 (high/alert)
    landmark_count: int
    source: str                 # "video_file" | "webcam_frame" | "image"
    confidence: float           # overall detection confidence 0.0–1.0


class VisualAnalyzer:
    """
    Extracts body posture proxy features from video or image.
    Gracefully returns None if MediaPipe is unavailable or no subject detected.
    """

    def __init__(self):
        self._pose = None
        self._mp_available = None

    def _check_mediapipe(self) -> bool:
        if self._mp_available is None:
            try:
                import mediapipe  # noqa: F401
                import cv2  # noqa: F401
                self._mp_available = True
            except ImportError:
                self._mp_available = False
        return self._mp_available

    def _get_pose(self):
        if self._pose is None:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        return self._pose

    def analyze_image(self, image_path: str) -> Optional[VisualFeatures]:
        """Extract features from a single image file."""
        if not self._check_mediapipe():
            return None
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return self._analyze_frame(frame, source="image")

    def analyze_video_frame(self, frame: np.ndarray) -> Optional[VisualFeatures]:
        """Extract features from a single BGR video frame."""
        if not self._check_mediapipe():
            return None
        return self._analyze_frame(frame, source="webcam_frame")

    def analyze_video_file(self, video_path: str, max_frames: int = 30) -> Optional[VisualFeatures]:
        """
        Analyze a video file by sampling frames and averaging features.
        Returns None if video cannot be opened or no landmarks detected.
        """
        if not self._check_mediapipe():
            return None
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        results = []
        prev_landmarks = None

        for i in range(0, min(total_frames, max_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            feats = self._analyze_frame(frame, source="video_file", prev_landmarks=prev_landmarks)
            if feats is not None:
                results.append(feats)
        cap.release()

        if not results:
            return None
        return self._average_features(results)

    def _analyze_frame(
        self,
        frame: np.ndarray,
        source: str,
        prev_landmarks: Optional[list] = None,
    ) -> Optional[VisualFeatures]:
        """Run MediaPipe Pose on a single frame and extract proxy features."""
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose = self._get_pose()
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark
        lm_count = sum(1 for lm in landmarks if lm.visibility > 0.3)
        if lm_count < 5:
            return None

        # Extract y-coordinates of visible upper-body landmarks (0=nose, 11/12=shoulders, etc.)
        visible_lms = [lm for lm in landmarks if lm.visibility > 0.3]
        y_vals = [lm.y for lm in visible_lms]
        x_vals = [lm.x for lm in visible_lms]

        # Head position proxy: normalized y of nose landmark (0), inverted so high=1
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
        head_pos = 1.0 - nose.y if nose.visibility > 0.3 else 0.5

        # Posture score: spread of y values (tall/erect body = large y range)
        y_range = max(y_vals) - min(y_vals) if y_vals else 0.0
        posture_score = float(np.clip(y_range * 2.0, 0.0, 1.0))

        # Motion intensity: Euclidean distance from previous frame landmarks
        motion = 0.0
        if prev_landmarks is not None:
            try:
                curr_pts = np.array([[lm.x, lm.y] for lm in landmarks])
                prev_pts = np.array(prev_landmarks)
                if curr_pts.shape == prev_pts.shape:
                    motion = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
                    motion = float(np.clip(motion * 10.0, 0.0, 1.0))
            except Exception as exc:
                logger.debug("Motion intensity calculation failed: %s", exc)

        # Overall detection confidence: mean visibility of detected landmarks
        confidence = float(np.mean([lm.visibility for lm in landmarks if lm.visibility > 0.3]))

        return VisualFeatures(
            posture_score=round(posture_score, 3),
            motion_intensity=round(motion, 3),
            estimated_head_position=round(head_pos, 3),
            landmark_count=lm_count,
            source=source,
            confidence=round(confidence, 3),
        )

    def _average_features(self, features: list) -> VisualFeatures:
        """Average a list of VisualFeatures across video frames."""
        return VisualFeatures(
            posture_score=float(np.mean([f.posture_score for f in features])),
            motion_intensity=float(np.mean([f.motion_intensity for f in features])),
            estimated_head_position=float(np.mean([f.estimated_head_position for f in features])),
            landmark_count=int(np.mean([f.landmark_count for f in features])),
            source=features[0].source,
            confidence=float(np.mean([f.confidence for f in features])),
        )

    def is_available(self) -> bool:
        """Check if MediaPipe and OpenCV are available."""
        return self._check_mediapipe()
