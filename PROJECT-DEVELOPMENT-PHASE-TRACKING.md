# Dog Behavior Agent — Development Phase Tracking

## Quantified Improvement Targets

| Metric | Baseline (Heuristic) | Phase 3 Target | Phase 6 Target |
|--------|---------------------|----------------|----------------|
| 8-class classification accuracy | ~45% (rule-based) | ≥72% (SVM trained) | ≥80% (ensemble fine-tuned) |
| Mean confidence calibration ECE | N/A | ≤0.15 | ≤0.10 |
| API response latency (p95) | N/A | ≤2s (CPU) | ≤800ms (GPU) |
| Research papers in knowledge base | 15 (seed) | ≥40 (1 month) | ≥100 (6 months) |
| Owner satisfaction (guidance quality) | N/A | ≥4.0/5.0 | ≥4.3/5.0 |

---

## Phase 0: Research & Architecture (Week 1–2)
**Goal:** Understand the domain, identify datasets, finalize architecture.

### Tasks
- [x] Review ethology literature on canine vocalization (Yin 2004, Pongrácz 2010, Molnár 2008)
- [x] Identify available datasets: BarkNet, ESC-50, Kaggle Dog Barks, Zenodo canine audio
- [x] Define 8-class taxonomy based on ethological research
- [x] Choose audio feature pipeline: MFCC + wav2vec2
- [x] Choose classifier: SVM + RandomForest ensemble + heuristic fallback
- [x] Define visual analysis approach: MediaPipe Pose proxy
- [x] Finalize file layout and module responsibilities

**Deliverables:** CLAUDE.md, PROJECT-detail.md, PROJECT-DEVELOPMENT-PHASE-TRACKING.md
**Effort:** 3 person-days
**Success criteria:** Clear architecture diagram; 8 classes with acoustic signatures documented; 2+ datasets identified

---

## Phase 1: Core Agent Modules (Week 3–5)
**Goal:** Implement AudioAnalyzer and BehaviorClassifier.

### Tasks
- [x] `AudioAnalyzer.analyze_file()` — load file, extract MFCC + spectral features
- [x] `AudioAnalyzer.analyze_microphone()` — sounddevice real-time capture
- [x] `AudioAnalyzer.to_feature_vector()` — 138-dim flat feature vector
- [x] `BehaviorClassifier.HeuristicClassifier` — rule-based 8-class with acoustic thresholds
- [x] `BehaviorClassifier.SVMClassifier` — scikit-learn SVC RBF + StandardScaler pipeline
- [x] `BehaviorClassifier.classify()` — ensemble voting with calibrated confidence
- [x] `BehaviorClassifier.train()` — train from labeled audio files

**Deliverables:** `audio_analyzer.py`, `behavior_classifier.py`
**Effort:** 7 person-days
**Success criteria:** AudioAnalyzer extracts valid features from 10 diverse audio files; HeuristicClassifier achieves > 40% accuracy on manually labeled 30-sample test

---

## Phase 2: Orchestrator + Quality Gates (Week 6–7)
**Goal:** Wire modules into the orchestration loop.

### Tasks
- [x] `DogBehaviorOrchestrator.analyze()` — main E2E pipeline
- [x] Audio validity quality gate (RMS, duration, spectral range)
- [x] Safety gate for pain/distress + aggression
- [x] `MemoryManager` — SQLite WAL, 5 tables
- [x] Session persistence + dog profile tracking

**Deliverables:** `orchestrator.py`, `memory_manager.py`
**Effort:** 5 person-days
**Success criteria:** Full E2E pipeline produces valid `BehaviorAnalysisResult` on test audio files; all 7 quality gates operational

---

## Phase 3: HuggingFace Model Integration (Week 8–9)
**Goal:** Integrate wav2vec2, BGE, BART, Whisper.

### Tasks
- [x] `HFModelManager.extract_wav2vec2()` — 768-dim embedding extraction
- [x] `HFModelManager.encode()` — BGE text embeddings for knowledge retrieval
- [x] `HFModelManager.summarize()` — BART-CNN research paper summarization
- [x] `HFModelManager` CUDA auto-detect + idle unload 600s
- [x] `BehaviorClassifier` extended with wav2vec2 fusion (concatenate to MFCC vector)
- [x] `InterpretationEngine` knowledge retrieval via BGE FAISS search

**Deliverables:** `hf_model_manager.py` (with `extract_wav2vec2`), updated modules
**Effort:** 5 person-days
**Success criteria:** wav2vec2 embedding improves classification accuracy by ≥5%; BGE retrieval returns relevant papers

---

## Phase 4: LLM API Integration (Week 10–11)
**Goal:** Claude/GPT/Ollama interpretation layer.

### Tasks
- [x] `llm_client.py` — Claude/OpenAI/Ollama unified client with retry + cost tracking
- [x] `InterpretationEngine.interpret()` — full LLM interpretation pipeline
- [x] `InterpretationEngine._fallback_interpretation()` — template-based when all APIs down
- [x] JSON response validation and retry logic
- [x] Breed-specific context injection
- [x] Owner guidance generation (3 actionable steps)

**Deliverables:** `llm_client.py`, `interpretation_engine.py`
**Effort:** 5 person-days
**Success criteria:** Claude interpretation passes JSON schema validation; fallback template works when API mocked to fail; owner guidance rated ≥4.0/5.0 on 10 manual samples

---

## Phase 5: SECOND-KNOWLEDGE-BRAIN Pipeline (Week 12–13)
**Goal:** Research paper crawler + self-updating knowledge base.

### Tasks
- [x] `knowledge_updater.py` — ArXiv cs.SD + cs.CV + cs.LG XML API crawl
- [x] `knowledge_updater.py` — Semantic Scholar graph API 5 queries
- [x] SHA256 deduplication via `MemoryManager.is_known_paper()`
- [x] Recency × relevance scoring for paper ranking
- [x] Append to SECOND-KNOWLEDGE-BRAIN.md knowledge log
- [x] APScheduler weekly Sunday 02:00 trigger
- [x] FAISS index build from knowledge base for BGE retrieval

**Deliverables:** `knowledge_updater.py`, `SECOND-KNOWLEDGE-BRAIN.md` (15 seed papers)
**Effort:** 4 person-days
**Success criteria:** Crawler adds ≥3 new papers per run; zero duplicates on second run; FAISS index serves BGE queries in < 50ms

---

## Phase 6: VisualAnalyzer + Optional Multimodal (Week 14)
**Goal:** MediaPipe body language proxy features.

### Tasks
- [x] `visual_analyzer.py` — MediaPipe Pose landmark extraction
- [x] Posture score computation (erect vs cowering proxy)
- [x] Feature fusion with audio features in BehaviorClassifier
- [x] Graceful skip if no pose detected or camera unavailable

**Deliverables:** `visual_analyzer.py`, updated `behavior_classifier.py`
**Effort:** 3 person-days
**Success criteria:** VisualAnalyzer returns VisualFeatures for any video with visible subject; classification still works if visual features unavailable

---

## Phase 7: Docker + Testing (Week 15–16)
**Goal:** Containerize, run all test scenarios, achieve all quality gate targets.

### Tasks
- [x] `docker/docker-compose.yml` — dog-behavior-agent + ollama services
- [x] `docker/Dockerfile` — python:3.12-slim, system libs, non-root user, EXPOSE 8009
- [x] `tests/test_agent.py` — 40+ automated tests
- [x] `tests/test-scenarios.md` — 8 end-to-end scenarios
- [x] `config/agent_config.yaml` — full runtime configuration
- [x] `config/.env.example` — all required environment variables
- [x] `agent/main.py` — Click CLI + FastAPI REST server

**Deliverables:** All deployment + test files
**Effort:** 6 person-days
**Success criteria:** All 40 automated tests pass; all 8 scenarios documented with expected outputs; Docker build succeeds

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No labeled dog vocalization dataset available | Medium | High | HeuristicClassifier works without labels; BarkNet is publicly available |
| wav2vec2 latency too high for real-time | Medium | Medium | wav2vec2 is optional; MFCC fallback is fast (< 50ms CPU) |
| MediaPipe misses dog body landmarks | High | Low | VisualAnalyzer is fully optional; audio-only mode always works |
| LLM API rate limiting | Low | Medium | Exponential backoff + Ollama offline fallback |
| SECOND-KNOWLEDGE-BRAIN grows too large | Low | Low | Rolling window: keep top 500 papers by recency×relevance |
