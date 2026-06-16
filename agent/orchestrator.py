"""
Core orchestrator for the Dog Behavior Agent.
Wires AudioAnalyzer → BehaviorClassifier → InterpretationEngine → MemoryManager.
"""

import uuid
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BehaviorAnalysisResult:
    session_id: str
    behavior_label: str
    confidence: float
    secondary_label: str
    urgency_level: str
    narrative: str
    breed_context: str
    guidance_steps: List[str]
    confidence_explanation: str
    model_used: str
    has_wav2vec2: bool
    has_visual: bool
    duration_seconds: float
    audio_source: str
    llm_provider: str
    latency_ms: float
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    valid_audio: bool = True
    validation_message: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "behavior_label": self.behavior_label,
            "confidence": self.confidence,
            "secondary_label": self.secondary_label,
            "urgency_level": self.urgency_level,
            "narrative": self.narrative,
            "breed_context": self.breed_context,
            "guidance_steps": self.guidance_steps,
            "confidence_explanation": self.confidence_explanation,
            "model_used": self.model_used,
            "has_wav2vec2": self.has_wav2vec2,
            "has_visual": self.has_visual,
            "duration_seconds": self.duration_seconds,
            "audio_source": self.audio_source,
            "llm_provider": self.llm_provider,
            "latency_ms": self.latency_ms,
            "all_probabilities": self.all_probabilities,
            "valid_audio": self.valid_audio,
            "validation_message": self.validation_message,
        }


class DogBehaviorOrchestrator:
    """
    Main agent orchestrator.
    Lazy-initializes all modules on first use.
    """

    def __init__(self):
        self._audio_analyzer = None
        self._classifier = None
        self._interpreter = None
        self._visual_analyzer = None
        self._memory = None
        self._llm = None
        self._hf = None
        self._knowledge_updater = None

    def _get_audio_analyzer(self):
        if self._audio_analyzer is None:
            from agent.modules.audio_analyzer import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer()
        return self._audio_analyzer

    def _get_classifier(self):
        if self._classifier is None:
            from agent.modules.behavior_classifier import BehaviorClassifier
            self._classifier = BehaviorClassifier()
        return self._classifier

    def _get_interpreter(self):
        if self._interpreter is None:
            from agent.modules.interpretation_engine import InterpretationEngine
            self._interpreter = InterpretationEngine()
        return self._interpreter

    def _get_visual_analyzer(self):
        if self._visual_analyzer is None:
            from agent.modules.visual_analyzer import VisualAnalyzer
            self._visual_analyzer = VisualAnalyzer()
        return self._visual_analyzer

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager()
        return self._memory

    def _get_knowledge_updater(self):
        if self._knowledge_updater is None:
            from tools.knowledge_updater import KnowledgeUpdater
            self._knowledge_updater = KnowledgeUpdater()
        return self._knowledge_updater

    def analyze_file(
        self,
        audio_path: str,
        video_path: Optional[str] = None,
        breed: str = "unknown",
        age: str = "unknown",
        dog_id: Optional[str] = None,
        use_wav2vec2: bool = True,
        show_all_probabilities: bool = False,
    ) -> BehaviorAnalysisResult:
        """Analyze an audio file (+ optional video) and return full behavior analysis."""
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())[:8]

        analyzer = self._get_audio_analyzer()
        classifier = self._get_classifier()
        interpreter = self._get_interpreter()
        memory = self._get_memory()

        # Step 1: Extract audio features
        audio_feats = analyzer.analyze_file(audio_path, use_wav2vec2=use_wav2vec2)

        # Step 2: Audio quality gate
        valid, reason = analyzer.validate_audio(audio_feats)
        if not valid:
            latency = (time.perf_counter() - t0) * 1000
            return BehaviorAnalysisResult(
                session_id=session_id,
                behavior_label="invalid",
                confidence=0.0,
                secondary_label="",
                urgency_level="low",
                narrative=f"Audio validation failed: {reason}",
                breed_context="",
                guidance_steps=["Provide a clear audio recording of dog vocalizations."],
                confidence_explanation="",
                model_used="none",
                has_wav2vec2=False,
                has_visual=False,
                duration_seconds=audio_feats.duration_seconds,
                audio_source=audio_path,
                llm_provider="none",
                latency_ms=round(latency, 1),
                valid_audio=False,
                validation_message=reason,
            )

        # Step 3: Optional visual features
        visual_feats = None
        if video_path:
            try:
                visual_feats = self._get_visual_analyzer().analyze_video_file(video_path)
            except Exception as exc:
                logger.warning("Visual analysis skipped for %s: %s", video_path, exc)

        # Step 4: Classify behavior
        classification = classifier.classify(audio_feats, visual_feats)

        # Step 5: Interpret
        interpretation = interpreter.interpret(classification, audio_feats, breed=breed, age=age)

        latency = (time.perf_counter() - t0) * 1000

        # Step 6: Persist
        try:
            memory.save_session(
                session_id=session_id,
                audio_source=audio_path,
                duration_seconds=audio_feats.duration_seconds,
                behavior_label=classification.behavior_label,
                confidence=classification.confidence,
                secondary_label=classification.secondary_label,
                urgency_level=interpretation.urgency_level,
                model_used=classification.model_used,
                has_wav2vec2=classification.has_wav2vec2,
                has_visual=visual_feats is not None,
                llm_provider=interpretation.llm_provider_used,
                dog_id=dog_id,
                feature_vector_dim=classification.feature_vector_dim,
                spectral_centroid=audio_feats.spectral_centroid,
                rms_energy=audio_feats.rms_energy,
                tempo=audio_feats.tempo,
                zcr=audio_feats.zero_crossing_rate,
            )
        except Exception as exc:
            logger.warning("Failed to persist session %s: %s", session_id, exc)

        return BehaviorAnalysisResult(
            session_id=session_id,
            behavior_label=classification.behavior_label,
            confidence=classification.confidence,
            secondary_label=classification.secondary_label,
            urgency_level=interpretation.urgency_level,
            narrative=interpretation.narrative,
            breed_context=interpretation.breed_context,
            guidance_steps=interpretation.guidance_steps,
            confidence_explanation=interpretation.confidence_explanation,
            model_used=classification.model_used,
            has_wav2vec2=classification.has_wav2vec2,
            has_visual=visual_feats is not None,
            duration_seconds=audio_feats.duration_seconds,
            audio_source=audio_path,
            llm_provider=interpretation.llm_provider_used,
            latency_ms=round(latency, 1),
            all_probabilities=classification.all_probabilities if show_all_probabilities else {},
            valid_audio=True,
            validation_message="ok",
        )

    def analyze_microphone(
        self,
        duration: float = 3.0,
        breed: str = "unknown",
        age: str = "unknown",
        dog_id: Optional[str] = None,
        use_wav2vec2: bool = True,
    ) -> BehaviorAnalysisResult:
        """Record from microphone and analyze."""
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())[:8]

        analyzer = self._get_audio_analyzer()
        classifier = self._get_classifier()
        interpreter = self._get_interpreter()
        memory = self._get_memory()

        audio_feats = analyzer.analyze_microphone(duration, use_wav2vec2=use_wav2vec2)
        valid, reason = analyzer.validate_audio(audio_feats)
        if not valid:
            latency = (time.perf_counter() - t0) * 1000
            return BehaviorAnalysisResult(
                session_id=session_id, behavior_label="invalid", confidence=0.0,
                secondary_label="", urgency_level="low",
                narrative=f"Audio validation failed: {reason}", breed_context="",
                guidance_steps=[], confidence_explanation="", model_used="none",
                has_wav2vec2=False, has_visual=False, duration_seconds=duration,
                audio_source="microphone", llm_provider="none",
                latency_ms=round(latency, 1), valid_audio=False, validation_message=reason,
            )

        classification = classifier.classify(audio_feats)
        interpretation = interpreter.interpret(classification, audio_feats, breed=breed, age=age)
        latency = (time.perf_counter() - t0) * 1000

        try:
            memory.save_session(
                session_id=session_id, audio_source="microphone",
                duration_seconds=audio_feats.duration_seconds,
                behavior_label=classification.behavior_label,
                confidence=classification.confidence,
                secondary_label=classification.secondary_label,
                urgency_level=interpretation.urgency_level,
                model_used=classification.model_used,
                has_wav2vec2=classification.has_wav2vec2,
                has_visual=False,
                llm_provider=interpretation.llm_provider_used,
                dog_id=dog_id,
                feature_vector_dim=classification.feature_vector_dim,
                spectral_centroid=audio_feats.spectral_centroid,
                rms_energy=audio_feats.rms_energy,
                tempo=audio_feats.tempo,
                zcr=audio_feats.zero_crossing_rate,
            )
        except Exception as exc:
            logger.warning("Failed to persist microphone session %s: %s", session_id, exc)

        return BehaviorAnalysisResult(
            session_id=session_id,
            behavior_label=classification.behavior_label,
            confidence=classification.confidence,
            secondary_label=classification.secondary_label,
            urgency_level=interpretation.urgency_level,
            narrative=interpretation.narrative,
            breed_context=interpretation.breed_context,
            guidance_steps=interpretation.guidance_steps,
            confidence_explanation=interpretation.confidence_explanation,
            model_used=classification.model_used,
            has_wav2vec2=classification.has_wav2vec2,
            has_visual=False,
            duration_seconds=audio_feats.duration_seconds,
            audio_source="microphone",
            llm_provider=interpretation.llm_provider_used,
            latency_ms=round(latency, 1),
            valid_audio=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        return self._get_memory().get_stats()

    def get_cost_report(self) -> Dict[str, Any]:
        return self._get_memory().get_cost_summary()

    def update_knowledge(self) -> Dict[str, int]:
        return self._get_knowledge_updater().run()

    def start_scheduler(self):
        """Start APScheduler for weekly knowledge updates."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            self.update_knowledge,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_knowledge_update",
            replace_existing=True,
        )
        scheduler.start()
        return scheduler

    def get_prometheus_metrics(self) -> str:
        """Return Prometheus text format metrics."""
        stats = self.get_stats()
        cost = self.get_cost_report()
        total_cost = sum(v.get("total_usd", 0) for v in cost.values())
        lines = [
            f'dog_behavior_sessions_total {stats["total_sessions"]}',
            f'dog_behavior_dog_profiles_total {stats["dog_profiles"]}',
            f'dog_behavior_known_papers_total {stats["known_papers"]}',
            f'dog_behavior_llm_cost_30d_usd {total_cost:.4f}',
        ]
        for label, count in stats.get("behavior_distribution", {}).items():
            safe_label = label.replace("/", "_")
            lines.append(f'dog_behavior_class_count{{label="{safe_label}"}} {count}')
        return "\n".join(lines) + "\n"
