# Dog Behavior Agent — CLAUDE.md

**Agent Name:** dog-behavior-agent
**Tagline:** Decode what your dog is saying — real-time audio behavior analysis via ML/DL + LLM interpretation
**Build Phase:** Phase 1 — Core Modules

## Problem Statement

Dogs communicate exclusively through vocalizations and body language. Most owners misinterpret or entirely miss the nuanced signals their dogs send — leading to stress, behavioral problems, and missed welfare concerns. This agent captures real-time dog audio (barks, growls, whines, howls, whimpers), extracts rich acoustic features (MFCC + chroma + spectral statistics + wav2vec2 embeddings), classifies behavioral intent across 8 ethologically-grounded categories using a trained ML ensemble, optionally enhances classification with MediaPipe body posture analysis from webcam, and delivers natural-language owner guidance via Claude LLM. The system continuously self-improves by ingesting the latest bioacoustics and animal behavior research.

## Agent Architecture

```
Audio Input (mic/file) + Optional Video (webcam/file)
              ↓
┌─────────────────────────────────────────────────────────┐
│  DogBehaviorOrchestrator (agent/orchestrator.py)        │
│  ┌───────────────┐   ┌──────────────┐                  │
│  │ AudioAnalyzer │→  │  VisualAna-  │ (optional)       │
│  │ (MFCC+wav2vec2│   │  lyzer       │                  │
│  └───────────────┘   └──────────────┘                  │
│          ↓                   ↓                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  BehaviorClassifier (SVM Ensemble + wav2vec2)     │  │
│  │  8 classes: aggression/fear/excitement/pain/      │  │
│  │  attention/play/alert/greeting                    │  │
│  └───────────────────────────────────────────────────┘  │
│          ↓                                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  InterpretationEngine (Claude LLM)                │  │
│  │  Natural language + breed context + owner advice  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          ↓              ↓              ↓
     LLM API       HuggingFace    SECOND-KNOWLEDGE-
   (llm_client)  (wav2vec2/BGE)    BRAIN.md
```

**Steps:**
1. Capture or load audio → validate dog vocalization (energy/frequency heuristic gate)
2. Extract 40-coeff MFCC + deltas + chroma + spectral stats → 134-dim feature vector
3. Optionally extract wav2vec2-base 768-dim embedding via HuggingFace
4. Optionally extract MediaPipe body posture proxy features from video
5. Fuse features → SVM + RandomForest ensemble → 8-class behavioral intent + confidence
6. Retrieve relevant context from SECOND-KNOWLEDGE-BRAIN.md via BGE embeddings
7. LLM (Claude): interpret classification in natural language + breed context + owner response
8. Persist session to SQLite + update behavior pattern history

## Module List
| File | Description |
|------|-------------|
| `agent/modules/audio_analyzer.py` | Real-time mic capture or file load; MFCC/chroma/spectral/ZCR/RMS feature extraction; wav2vec2 embedding via HF |
| `agent/modules/behavior_classifier.py` | 8-class SVM + RandomForest ensemble; heuristic rule-based fallback; confidence calibration; breed-specific priors |
| `agent/modules/interpretation_engine.py` | LLM-powered natural language interpretation; breed context retrieval; owner response strategy generation |
| `agent/modules/visual_analyzer.py` | Optional MediaPipe Pose/Holistic body language features; tail/ear/posture proxy estimation |

## HuggingFace Models
| Model ID | Task | Why Chosen |
|----------|------|------------|
| `facebook/wav2vec2-base` | Audio feature embedding (768-dim) | Self-supervised audio representation; domain-transferable to animal sounds; strong on ESC-50 via fine-tuning |
| `openai/whisper-large-v3` | Human speech detection gate | Distinguishes dog audio from background human speech noise |
| `BAAI/bge-large-en-v1.5` | Knowledge retrieval embeddings | #1 MTEB English embedding; used for semantic search in SECOND-KNOWLEDGE-BRAIN.md |
| `facebook/bart-large-cnn` | Research paper summarization | Condenses retrieved research context for LLM prompt |

## LLM API Integration
| Provider | Model | Use Cases |
|----------|-------|-----------|
| Claude (primary) | `claude-opus-4-8` | Behavioral interpretation narrative, breed-specific context, owner response strategy, research synthesis |
| OpenAI (fallback) | `gpt-4o` | Same as Claude when ANTHROPIC_API_KEY unavailable |
| Ollama (offline) | `llama3` | Privacy mode — no data leaves the machine; PRIVACY_MODE=true env var |

## Knowledge Crawl Sources
- **ArXiv:** cs.SD (sound), cs.CV, cs.LG — weekly Sunday 02:00
- **Semantic Scholar:** 5 queries — dog vocalization ML, canine behavior AI, bioacoustics classification, animal sound recognition, MFCC dog bark
- **Applied Animal Behaviour Science** — DOI-based abstracts
- **Animal Cognition journal** — DOI-based abstracts
- **Interspeech proceedings** — animal audio track papers

## Active Development Tasks
- [x] Architecture design & dataset sourcing plan
- [x] AudioAnalyzer module (MFCC + wav2vec2)
- [x] BehaviorClassifier 8-class ensemble
- [x] InterpretationEngine (Claude)
- [x] VisualAnalyzer (MediaPipe optional)
- [x] MemoryManager SQLite
- [x] SECOND-KNOWLEDGE-BRAIN pipeline
- [x] Docker + testing
- [x] CLI + REST API
