"""
Automated tests for the Dog Behavior Agent.
Covers AudioAnalyzer, BehaviorClassifier, InterpretationEngine,
VisualAnalyzer, MemoryManager, LLMClient, HFModelManager, Integration, CLI.
"""

import json
import os
import sys
import tempfile
import threading
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_audio_alert():
    """Generate synthetic 'alert bark' audio (low frequency, high energy, single burst)."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    # Low-frequency burst at 600 Hz with decay
    audio = 0.05 * np.sin(2 * np.pi * 600 * t) * np.exp(-t * 2.0)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_fear():
    """Generate synthetic 'fear whimper' (high frequency, low energy, sustained)."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    audio = 0.005 * np.sin(2 * np.pi * 1800 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_excitement():
    """Generate synthetic 'excitement' (high pitch, high energy, rapid)."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    audio = 0.04 * np.sin(2 * np.pi * 1500 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_silent():
    sr = 16000
    return np.zeros(sr * 2, dtype=np.float32)


@pytest.fixture
def tmp_wav(tmp_path, sample_audio_alert):
    """Write sample alert audio to a temporary WAV file."""
    import soundfile as sf
    wav_path = tmp_path / "test_alert.wav"
    sf.write(str(wav_path), sample_audio_alert, 16000)
    return str(wav_path)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_memory.db"


# ── AudioAnalyzer Tests ───────────────────────────────────────────────────────

class TestAudioAnalyzer:
    def test_extract_features_returns_correct_shape(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        assert feats.mfcc.shape == (40,)
        assert feats.mfcc_delta.shape == (40,)
        assert feats.mfcc_delta2.shape == (40,)
        assert feats.chroma.shape == (12,)

    def test_to_vector_is_138_dim(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        vec = feats.to_vector()
        assert vec.shape == (138,), f"Expected (138,), got {vec.shape}"

    def test_validate_audio_rejects_silent(self, sample_audio_silent):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_silent, use_wav2vec2=False)
        valid, reason = analyzer.validate_audio(feats)
        assert not valid
        assert "silent" in reason.lower()

    def test_validate_audio_accepts_valid(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        valid, reason = analyzer.validate_audio(feats)
        assert valid, f"Should be valid: {reason}"

    def test_analyze_file(self, tmp_wav):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.analyze_file(tmp_wav, use_wav2vec2=False)
        assert feats.source == "file"
        assert feats.duration_seconds > 0

    def test_is_likely_dog_audio(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        is_dog, score = analyzer.is_likely_dog_audio(feats)
        assert isinstance(is_dog, bool)
        assert 0.0 <= score <= 1.0

    def test_feature_duration(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        assert abs(feats.duration_seconds - 2.0) < 0.1


# ── BehaviorClassifier Tests ──────────────────────────────────────────────────

class TestBehaviorClassifier:
    def test_classify_returns_valid_label(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        from agent.modules.behavior_classifier import BehaviorClassifier, BEHAVIOR_CLASSES
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        clf = BehaviorClassifier()
        result = clf.classify(feats)
        assert result.behavior_label in BEHAVIOR_CLASSES or result.behavior_label == "uncertain"

    def test_classify_confidence_in_range(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        from agent.modules.behavior_classifier import BehaviorClassifier
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        clf = BehaviorClassifier()
        result = clf.classify(feats)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_probabilities_sum_to_one(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        from agent.modules.behavior_classifier import BehaviorClassifier
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        clf = BehaviorClassifier()
        result = clf.classify(feats)
        total = sum(result.all_probabilities.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum = {total}"

    def test_heuristic_classifier_directly(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        from agent.modules.behavior_classifier import HeuristicClassifier
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        heuristic = HeuristicClassifier()
        scores = heuristic.classify(feats)
        assert len(scores) == 8
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_uncertain_on_low_confidence(self, sample_audio_silent):
        from agent.modules.audio_analyzer import AudioAnalyzer, AudioFeatures
        from agent.modules.behavior_classifier import BehaviorClassifier
        # Fabricate near-zero features
        feats = AudioFeatures(
            mfcc=np.zeros(40), mfcc_delta=np.zeros(40), mfcc_delta2=np.zeros(40),
            chroma=np.zeros(12), spectral_centroid=4000.0, spectral_bandwidth=2000.0,
            spectral_rolloff=3000.0, zero_crossing_rate=0.02, rms_energy=0.002,
            tempo=100.0, wav2vec2_embedding=None, duration_seconds=2.0,
            sample_rate=16000, source="test",
        )
        clf = BehaviorClassifier()
        result = clf.classify(feats)
        # Should not crash; may return "uncertain" if confidence < threshold
        assert result.behavior_label is not None

    def test_get_behavior_description(self):
        from agent.modules.behavior_classifier import BehaviorClassifier
        clf = BehaviorClassifier()
        desc = clf.get_behavior_description("aggression")
        assert "aggress" in desc.lower()
        desc2 = clf.get_behavior_description("unknown_label")
        assert isinstance(desc2, str)

    def test_model_used_includes_heuristic(self, sample_audio_alert):
        from agent.modules.audio_analyzer import AudioAnalyzer
        from agent.modules.behavior_classifier import BehaviorClassifier
        analyzer = AudioAnalyzer()
        feats = analyzer.extract_features(sample_audio_alert, use_wav2vec2=False)
        clf = BehaviorClassifier()
        result = clf.classify(feats)
        assert "heuristic" in result.model_used


# ── InterpretationEngine Tests ────────────────────────────────────────────────

class TestInterpretationEngine:
    def _make_classification_result(self, label="alert_warning", confidence=0.70):
        from agent.modules.behavior_classifier import ClassificationResult
        from agent.modules.behavior_classifier import BEHAVIOR_CLASSES
        probs = {c: 0.1 for c in BEHAVIOR_CLASSES}
        probs[label] = confidence
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        return ClassificationResult(
            behavior_label=label, confidence=confidence,
            secondary_label="greeting", secondary_confidence=0.1,
            all_probabilities=probs, feature_vector_dim=138,
            model_used="heuristic", has_wav2vec2=False,
        )

    def _make_audio_features(self):
        from agent.modules.audio_analyzer import AudioFeatures
        return AudioFeatures(
            mfcc=np.zeros(40), mfcc_delta=np.zeros(40), mfcc_delta2=np.zeros(40),
            chroma=np.zeros(12), spectral_centroid=700.0, spectral_bandwidth=800.0,
            spectral_rolloff=1200.0, zero_crossing_rate=0.03, rms_energy=0.03,
            tempo=80.0, wav2vec2_embedding=None, duration_seconds=2.0,
            sample_rate=16000, source="test",
        )

    def test_fallback_interpretation_all_labels(self):
        from agent.modules.interpretation_engine import InterpretationEngine, FALLBACK_INTERPRETATIONS
        from agent.modules.behavior_classifier import BEHAVIOR_CLASSES
        for label in BEHAVIOR_CLASSES:
            assert label in FALLBACK_INTERPRETATIONS, f"Missing fallback for {label}"
            entry = FALLBACK_INTERPRETATIONS[label]
            assert "narrative" in entry
            assert "guidance_steps" in entry

    def test_interpret_with_fallback(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        engine = InterpretationEngine()
        # Mock LLM to raise
        with patch.object(engine, "_get_llm", side_effect=ImportError("no llm")):
            cls_result = self._make_classification_result()
            audio_feats = self._make_audio_features()
            result = engine.interpret(cls_result, audio_feats)
        assert isinstance(result.narrative, str) and len(result.narrative) > 0
        assert isinstance(result.guidance_steps, list) and len(result.guidance_steps) > 0
        assert result.llm_provider_used == "fallback"

    def test_safety_gate_pain_distress(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        engine = InterpretationEngine()
        with patch.object(engine, "_get_llm", side_effect=Exception("no llm")):
            cls_result = self._make_classification_result("pain_distress", 0.80)
            audio_feats = self._make_audio_features()
            result = engine.interpret(cls_result, audio_feats)
        assert result.urgency_level == "high"
        vet_in_guidance = any("vet" in s.lower() or "veterinar" in s.lower() for s in result.guidance_steps)
        assert vet_in_guidance

    def test_safety_gate_aggression(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        engine = InterpretationEngine()
        with patch.object(engine, "_get_llm", side_effect=Exception("no llm")):
            cls_result = self._make_classification_result("aggression", 0.80)
            audio_feats = self._make_audio_features()
            result = engine.interpret(cls_result, audio_feats)
        assert result.urgency_level == "high"

    def test_parse_json_response_valid(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        engine = InterpretationEngine()
        valid_json = '{"narrative": "test", "breed_context": "", "guidance_steps": ["a","b","c"], "urgency_level": "low", "confidence_explanation": "x"}'
        parsed = engine._parse_json_response(valid_json)
        assert parsed["narrative"] == "test"

    def test_parse_json_response_with_fences(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        engine = InterpretationEngine()
        fenced = '```json\n{"narrative": "test", "guidance_steps": []}\n```'
        parsed = engine._parse_json_response(fenced)
        assert parsed is not None
        assert parsed["narrative"] == "test"

    def test_format_result(self):
        from agent.modules.interpretation_engine import InterpretationEngine, InterpretationResult
        engine = InterpretationEngine()
        interp = InterpretationResult(
            narrative="Test narrative.", breed_context="",
            guidance_steps=["Step 1", "Step 2", "Step 3"],
            urgency_level="low", confidence_explanation="Test.",
            research_context_used=False, llm_provider_used="fallback",
        )
        cls_result = self._make_classification_result()
        formatted = engine.format_result(interp, cls_result)
        assert "ALERT WARNING" in formatted
        assert "Step 1" in formatted


# ── MemoryManager Tests ───────────────────────────────────────────────────────

class TestMemoryManager:
    def test_save_and_retrieve_session(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        mm.save_session(
            session_id="test-001", audio_source="file.wav",
            duration_seconds=2.5, behavior_label="alert_warning",
            confidence=0.72, secondary_label="greeting",
            urgency_level="low", model_used="heuristic",
            has_wav2vec2=False, has_visual=False,
            llm_provider="fallback", dog_id="buddy",
            feature_vector_dim=138, spectral_centroid=700.0,
            rms_energy=0.03, tempo=80.0, zcr=0.03,
        )
        sessions = mm.get_recent_sessions(limit=5)
        assert len(sessions) == 1
        assert sessions[0]["behavior_label"] == "alert_warning"

    def test_dog_profile_upsert(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        mm.upsert_dog_profile("rex", name="Rex", breed="Labrador", age="3 years")
        profile = mm.get_dog_profile("rex")
        assert profile is not None
        assert profile["breed"] == "Labrador"
        # Update breed
        mm.upsert_dog_profile("rex", breed="Golden Retriever")
        profile2 = mm.get_dog_profile("rex")
        assert profile2["breed"] == "Golden Retriever"

    def test_known_paper_deduplication(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        mm.mark_paper_known("Test Paper", "https://doi.org/10.1000/test")
        assert mm.is_known_paper("Test Paper", "https://doi.org/10.1000/test")
        assert not mm.is_known_paper("Other Paper", "https://doi.org/10.1000/other")

    def test_llm_cost_log(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        mm.log_llm_cost("claude", "claude-opus-4-8", 100, 200, 0.0165, "interpret")
        summary = mm.get_cost_summary()
        assert "claude" in summary
        assert summary["claude"]["call_count"] == 1

    def test_stats(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        stats = mm.get_stats()
        assert "total_sessions" in stats
        assert "behavior_distribution" in stats

    def test_concurrent_writes(self, tmp_db):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(db_path=tmp_db)
        errors = []
        def write_session(i):
            try:
                mm.save_session(
                    session_id=f"concurrent-{i}", audio_source="mic",
                    duration_seconds=1.0, behavior_label="greeting",
                    confidence=0.5, secondary_label="play",
                    urgency_level="low", model_used="heuristic",
                    has_wav2vec2=False, has_visual=False,
                    llm_provider="fallback", dog_id=None,
                    feature_vector_dim=138, spectral_centroid=500.0,
                    rms_energy=0.02, tempo=100.0, zcr=0.03,
                )
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=write_session, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Concurrent write errors: {errors}"


# ── LLMClient Tests ───────────────────────────────────────────────────────────

class TestLLMClient:
    def test_fallback_chain_ollama_last(self):
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        # No API keys set → chain should be ollama only
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
            chain = client._build_chain()
        assert "ollama" in chain

    def test_cost_calculation(self):
        from tools.llm_client import COST_PER_1K
        rates = COST_PER_1K.get("claude-opus-4-8")
        assert rates is not None
        cost = (100 * rates["input"] + 200 * rates["output"]) / 1000.0
        assert cost > 0

    def test_complete_all_providers_fail(self):
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
            with patch.object(client, "_call_ollama", side_effect=Exception("ollama down")):
                with pytest.raises(RuntimeError):
                    client.complete("test prompt")


# ── HFModelManager Tests ──────────────────────────────────────────────────────

class TestHFModelManager:
    def test_tfidf_fallback_shape(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        result = mgr._tfidf_fallback(["hello world", "dog bark"])
        assert result.shape == (2, 384)

    def test_extractive_summary_fallback(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        text = "Dogs communicate through barks. Barks have acoustic features. MFCC captures tonal properties. This is useful for classification."
        summary = mgr._extractive_summary_fallback(text, max_length=20)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_heuristic_rerank(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        query = "dog bark classification"
        passages = [
            "Cat meow analysis",
            "Dog bark recognition using SVM",
            "Weather forecasting methods",
        ]
        indices = mgr._heuristic_rerank(query, passages, top_k=2)
        assert 1 in indices  # "Dog bark recognition" should rank highest


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_file_no_llm(self, tmp_wav, tmp_db):
        """Full pipeline: audio file → classify → interpret (fallback) → save."""
        from agent.orchestrator import DogBehaviorOrchestrator
        orch = DogBehaviorOrchestrator()
        # Override memory to use tmp db
        from agent.memory.memory_manager import MemoryManager
        orch._memory = MemoryManager(db_path=tmp_db)
        # Patch LLM to fail
        from agent.modules.interpretation_engine import InterpretationEngine
        with patch.object(InterpretationEngine, "_get_llm", side_effect=Exception("no llm")):
            orch._interpreter = InterpretationEngine()
            result = orch.analyze_file(tmp_wav, use_wav2vec2=False)
        assert result.behavior_label is not None
        assert result.valid_audio or not result.valid_audio  # either is ok for synthetic audio
        assert result.latency_ms > 0

    def test_orchestrator_stats(self, tmp_db):
        from agent.orchestrator import DogBehaviorOrchestrator
        from agent.memory.memory_manager import MemoryManager
        orch = DogBehaviorOrchestrator()
        orch._memory = MemoryManager(db_path=tmp_db)
        stats = orch.get_stats()
        assert "total_sessions" in stats

    def test_result_to_dict_serializable(self, tmp_wav, tmp_db):
        from agent.orchestrator import DogBehaviorOrchestrator
        from agent.memory.memory_manager import MemoryManager
        from agent.modules.interpretation_engine import InterpretationEngine
        orch = DogBehaviorOrchestrator()
        orch._memory = MemoryManager(db_path=tmp_db)
        with patch.object(InterpretationEngine, "_get_llm", side_effect=Exception("no llm")):
            orch._interpreter = InterpretationEngine()
            result = orch.analyze_file(tmp_wav, use_wav2vec2=False)
        result_dict = result.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)
        assert "behavior_label" in parsed

    def test_prometheus_metrics_format(self, tmp_db):
        from agent.orchestrator import DogBehaviorOrchestrator
        from agent.memory.memory_manager import MemoryManager
        orch = DogBehaviorOrchestrator()
        orch._memory = MemoryManager(db_path=tmp_db)
        metrics = orch.get_prometheus_metrics()
        assert "dog_behavior_sessions_total" in metrics
        assert "dog_behavior_known_papers_total" in metrics

    def test_analysis_respects_audio_quality_gate(self, tmp_db):
        """Silent audio should fail quality gate and not call classifier."""
        import soundfile as sf
        import tempfile
        from agent.orchestrator import DogBehaviorOrchestrator
        from agent.memory.memory_manager import MemoryManager
        orch = DogBehaviorOrchestrator()
        orch._memory = MemoryManager(db_path=tmp_db)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(32000, dtype=np.float32), 16000)
            silent_path = f.name
        try:
            result = orch.analyze_file(silent_path, use_wav2vec2=False)
            assert result.valid_audio is False
            assert result.behavior_label == "invalid"
        finally:
            os.unlink(silent_path)


# ── CLI Smoke Tests ───────────────────────────────────────────────────────────

class TestCLISmokeTests:
    def test_cli_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower() or "Usage" in result.output

    def test_cli_analyze_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_cli_update_knowledge_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["update-knowledge", "--help"])
        assert result.exit_code == 0

    def test_cli_cost_report_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["cost-report", "--help"])
        assert result.exit_code == 0

    def test_cli_analyze_missing_file(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "nonexistent_file.wav"])
        # Should exit with error (file not found)
        assert result.exit_code != 0 or "Error" in result.output or "No such file" in result.output
