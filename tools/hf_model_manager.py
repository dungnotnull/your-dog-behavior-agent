"""
HuggingFace model manager for Dog Behavior Agent.
Lazy-loading singleton with CUDA auto-detect and idle unload after 600s.
Models: wav2vec2-base, whisper-large-v3, bge-large-en-v1.5, bart-large-cnn
"""

import threading
import numpy as np
from typing import Optional, List, Union


class HFModelManager:
    """Singleton lazy-loading manager for HuggingFace models."""

    _instance: Optional["HFModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "HFModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._loaded: dict = {}
        self._timers: dict = {}
        self._model_lock = threading.Lock()
        self._device = self._detect_device()

    def _detect_device(self):
        try:
            import torch
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return None

    def _reset_idle_timer(self, key: str):
        """Restart 600s idle unload timer for a model."""
        if key in self._timers:
            self._timers[key].cancel()
        timer = threading.Timer(600.0, self._unload_model, args=[key])
        timer.daemon = True
        timer.start()
        self._timers[key] = timer

    def _unload_model(self, key: str):
        with self._model_lock:
            if key in self._loaded:
                del self._loaded[key]

    # ── wav2vec2 ──────────────────────────────────────────────────────────────

    def extract_wav2vec2(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract wav2vec2-base 768-dim embedding from audio array."""
        key = "wav2vec2"
        with self._model_lock:
            if key not in self._loaded:
                try:
                    from transformers import Wav2Vec2Processor, Wav2Vec2Model
                    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                    model.eval()
                    if self._device and str(self._device) == "cuda":
                        model = model.cuda()
                    self._loaded[key] = {"processor": processor, "model": model}
                except Exception as e:
                    raise RuntimeError(f"Failed to load wav2vec2: {e}") from e
            self._reset_idle_timer(key)

        import torch
        proc = self._loaded[key]["processor"]
        mdl = self._loaded[key]["model"]
        audio_f32 = audio.astype(np.float32)
        inputs = proc(audio_f32, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        if self._device and str(self._device) == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl(**inputs)
        hidden = outputs.last_hidden_state  # (1, T, 768)
        embedding = hidden.mean(dim=1).squeeze().cpu().numpy()  # (768,)
        return embedding.astype(np.float32)

    # ── BGE text embeddings ───────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to BGE-large embedding (1024-dim)."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts. Returns (N, 1024) float32 array."""
        key = "bge_large"
        with self._model_lock:
            if key not in self._loaded:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
                    self._loaded[key] = model
                except Exception:
                    return self._tfidf_fallback(texts)
            self._reset_idle_timer(key)

        try:
            model = self._loaded[key]
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        except Exception:
            return self._tfidf_fallback(texts)

    def _tfidf_fallback(self, texts: List[str]) -> np.ndarray:
        """Deterministic 384-dim TF-IDF character bigram fallback."""
        dim = 384
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            bigrams = [text_lower[j:j+2] for j in range(len(text_lower)-1)]
            for bg in bigrams:
                idx = (ord(bg[0]) * 31 + ord(bg[1])) % dim
                result[i, idx] += 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result

    # ── BGE reranker ──────────────────────────────────────────────────────────

    def rerank(self, query: str, passages: List[str], top_k: int = 3) -> List[int]:
        """Cross-encoder reranking. Returns indices of top_k passages."""
        key = "bge_reranker"
        with self._model_lock:
            if key not in self._loaded:
                try:
                    from sentence_transformers import CrossEncoder
                    model = CrossEncoder("BAAI/bge-reranker-large")
                    self._loaded[key] = model
                except Exception:
                    return self._heuristic_rerank(query, passages, top_k)
            self._reset_idle_timer(key)

        try:
            model = self._loaded[key]
            pairs = [[query, p] for p in passages]
            scores = model.predict(pairs)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return ranked[:top_k]
        except Exception:
            return self._heuristic_rerank(query, passages, top_k)

    def _heuristic_rerank(self, query: str, passages: List[str], top_k: int) -> List[int]:
        query_words = set(query.lower().split())
        scores = [len(query_words & set(p.lower().split())) for p in passages]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # ── BART summarization ────────────────────────────────────────────────────

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Summarize text using BART-large-CNN."""
        key = "bart_cnn"
        with self._model_lock:
            if key not in self._loaded:
                try:
                    from transformers import pipeline
                    summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-large-cnn",
                        device=0 if (self._device and str(self._device) == "cuda") else -1,
                    )
                    self._loaded[key] = summarizer
                except Exception:
                    return self._extractive_summary_fallback(text, max_length)
            self._reset_idle_timer(key)

        try:
            summarizer = self._loaded[key]
            input_text = text[:1024]
            result = summarizer(
                input_text,
                max_length=max_length,
                min_length=min(min_length, len(input_text.split())),
                do_sample=False,
            )
            return result[0]["summary_text"]
        except Exception:
            return self._extractive_summary_fallback(text, max_length)

    def _extractive_summary_fallback(self, text: str, max_length: int) -> str:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        summary = ". ".join(sentences[:3])
        words = summary.split()
        if len(words) > max_length:
            summary = " ".join(words[:max_length]) + "..."
        return summary

    # ── Whisper voice detection gate ─────────────────────────────────────────

    def is_human_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Quick check if audio contains human speech using Whisper.
        Returns True if detected language confidence is high for non-dog audio.
        """
        key = "whisper"
        with self._model_lock:
            if key not in self._loaded:
                try:
                    import whisper
                    model = whisper.load_model("base")
                    self._loaded[key] = model
                except Exception:
                    return False  # Can't tell; assume not human
            self._reset_idle_timer(key)

        try:
            import whisper
            model = self._loaded[key]
            audio_f32 = audio.astype(np.float32)
            audio_padded = whisper.pad_or_trim(audio_f32)
            mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)
            _, probs = model.detect_language(mel)
            top_prob = max(probs.values()) if probs else 0.0
            return top_prob > 0.7
        except Exception:
            return False

    def preload(self, model_keys: List[str]):
        """Preload specified models to warm up on startup."""
        for key in model_keys:
            try:
                if key == "bge_large":
                    self.encode_batch(["warmup"])
                elif key == "wav2vec2":
                    dummy = np.zeros(16000, dtype=np.float32)
                    self.extract_wav2vec2(dummy)
                elif key == "bart_cnn":
                    self.summarize("warmup text for preloading model")
            except Exception:
                pass
