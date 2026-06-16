# Dog Behavior Agent — PROJECT-detail.md

## Executive Summary

The Dog Behavior Agent is a real-time audio analysis system that decodes canine behavioral intent from vocalizations. Dogs communicate through a rich vocabulary of barks, growls, whines, howls, and whimpers — each carrying distinct emotional and intentional meaning rooted in ethology. This agent applies state-of-the-art audio ML (wav2vec2 + MFCC ensemble) to classify 8 behavioral intent categories, optionally adds video body-language analysis via MediaPipe, and interprets results in plain language via Claude LLM with breed-specific context and actionable owner guidance.

**Problem:** Most dog owners misinterpret canine vocalizations, leading to missed welfare signals (pain, fear) and escalated behavioral problems. No accessible real-time tool exists that combines ML audio analysis with domain-expert ethological knowledge.

**Solution:** An autonomous agent that listens, classifies, and explains dog behavior in natural language — improving continuously through self-updating research knowledge.

---

## Target Users & Use Cases

| User | Trigger | Agent Response |
|------|---------|----------------|
| Dog owner at home | Dog barks repeatedly → run analyze command | Classification: attention-seeking (87% confidence); Owner guidance: check food/water/exercise schedule |
| Veterinary clinic | Upload audio recording of distressed animal | Classification: pain/distress (91% confidence); Guidance: immediate vet examination recommended |
| Animal shelter | Monitor multiple dogs continuously | Real-time alert when aggression or fear detected; behavior log per dog per session |
| Dog trainer | Analyze audio during training session | Behavioral feedback loop — which commands cause fear vs excitement |
| Researcher | Batch analyze vocalization dataset | Export CSV with features + classifications + confidence scores |

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Trigger: CLI analyze / REST POST /analyze / event loop         │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────┐
│  DogBehaviorOrchestrator (agent/orchestrator.py)              │
│                                                               │
│  ┌─────────────────┐    ┌──────────────────┐                 │
│  │  AudioAnalyzer   │    │  VisualAnalyzer   │ (optional)     │
│  │                  │    │  (MediaPipe)      │                │
│  │  • mic capture   │    │  • pose estimate  │                │
│  │  • file load     │    │  • body features  │                │
│  │  • MFCC 40 coeff │    │  • tail/ear proxy │                │
│  │  • wav2vec2 emb  │    └──────────────────┘                │
│  └────────┬─────────┘              ↓                         │
│           └──────────── feature fusion ──────────────────┐   │
│                                                           ↓   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  BehaviorClassifier                                    │   │
│  │  • HeuristicClassifier (rule-based, always works)      │   │
│  │  • SVMClassifier (trained, best performance)           │   │
│  │  • RandomForestClassifier (ensemble member)            │   │
│  │  → 8-class intent + confidence + secondary label       │   │
│  └─────────────────────┬──────────────────────────────────┘   │
│                        ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  InterpretationEngine                                   │   │
│  │  • BGE retrieval from SECOND-KNOWLEDGE-BRAIN.md        │   │
│  │  • BART-CNN context summarization                       │   │
│  │  • Claude LLM: natural language interpretation          │   │
│  │  • Breed-specific context if breed is known             │   │
│  │  • Owner response strategy (3 actionable steps)        │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MemoryManager (SQLite WAL)                             │   │
│  │  • Save session, features, classification, guidance     │   │
│  │  • Behavior pattern history per dog profile             │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
          ↓              ↓                    ↓
     LLM API       HuggingFace          Knowledge Base
   (Claude/GPT/   (wav2vec2,BGE,       SECOND-KNOWLEDGE-
    Ollama)        BART,Whisper)        BRAIN.md
```

---

## Module Catalog

### 1. AudioAnalyzer (`agent/modules/audio_analyzer.py`)
**Responsibility:** Capture or load audio, extract comprehensive acoustic feature set.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | Microphone stream (sounddevice), audio file path (WAV/MP3/OGG/FLAC/M4A) |
| **Outputs** | `AudioFeatures` dataclass: mfcc(40), mfcc_delta(40), mfcc_delta2(40), chroma(12), spectral_centroid, spectral_bandwidth, spectral_rolloff, ZCR, RMS, tempo, wav2vec2_embedding(768) |
| **Tools called** | `HFModelManager.extract_wav2vec2()`, librosa, sounddevice |
| **Quality gate** | RMS > 0.001 (non-silent), duration > 0.1s, spectral centroid 50–8000 Hz |

**Feature set:** MFCC 40 coefficients + delta + delta2 (120 dims) + chroma 12 + 6 scalar stats = 138-dim handcrafted vector + optional 768-dim wav2vec2 embedding (fused via concatenation or PCA projection).

### 2. BehaviorClassifier (`agent/modules/behavior_classifier.py`)
**Responsibility:** Classify 8 behavioral intents from audio (and optional visual) features.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | `AudioFeatures` + optional `VisualFeatures` |
| **Outputs** | `ClassificationResult` dataclass: behavior_label, confidence, secondary_label, secondary_confidence, feature_vector, model_used |
| **Models** | `HeuristicClassifier` (rule-based, always active as fallback); `SVMClassifier` (scikit-learn SVC RBF, trained on saved data); `RandomForestClassifier` (100 trees, feature importance tracking) |
| **Quality gate** | Confidence ≥ 0.40 to report primary label; otherwise "uncertain" is returned |

**8 Behavior Classes:**
| Class | Acoustic Signature | Typical Trigger |
|-------|--------------------|-----------------|
| `aggression` | Low freq (100–400 Hz), high energy, continuous/staccato, low ZCR | Territory threat, resource guarding |
| `fear` | High pitch, whimpering frequency, irregular rhythm, low RMS | Pain, threatening stimulus, separation |
| `excitement` | High pitch, rapid tempo, rising pitch contour, high energy | Play invite, owner return, food |
| `pain_distress` | Sustained high-pitch whine/yelp, consistent frequency, mid RMS | Injury, illness, trapped |
| `attention_seeking` | Medium pitch, regular intervals, pitch plateau, sustained | Hunger, boredom, need for attention |
| `play` | Short staccato, rising-falling contour, moderate energy, high ZCR | Play bow context, familiar dog/human |
| `alert_warning` | Single/few deep barks, low-mid pitch, high initial energy | Novel sound, visitor, perceived threat |
| `greeting` | Mixed pitch, moderate RMS, varied tempo, upward pitch | Familiar person/dog approach, tail up |

### 3. InterpretationEngine (`agent/modules/interpretation_engine.py`)
**Responsibility:** Generate natural-language interpretation, breed context, and owner guidance via LLM.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | `ClassificationResult` + optional dog profile (breed, age, history) |
| **Outputs** | `InterpretationResult`: narrative (string), breed_context (string), owner_guidance (list[str] 3 steps), confidence_explanation (string), urgency_level (low/medium/high) |
| **LLM prompt** | `INTERPRETATION_PROMPT` — structured JSON output: narrative + breed_context + guidance_steps + urgency |
| **Knowledge retrieval** | BGE-large semantic search on SECOND-KNOWLEDGE-BRAIN.md → top-3 papers → BART-CNN summary → injected into LLM context |
| **Quality gate** | Urgency = "high" for pain/distress + aggression → prepend safety warning |

**Safety gates:**
- `pain_distress` + confidence > 0.70 → always recommend veterinary evaluation
- `aggression` → safety warning about bite risk
- Never diagnose medical conditions; always recommend professional when in doubt

### 4. VisualAnalyzer (`agent/modules/visual_analyzer.py`)
**Responsibility:** Optional body language feature extraction from video frame or webcam stream.

| Attribute | Detail |
|-----------|--------|
| **Inputs** | Video file path, webcam frame, or image file |
| **Outputs** | `VisualFeatures` dataclass: posture_score (0–1, erect vs cowering), motion_intensity, estimated_head_position, timestamp |
| **Tools** | `mediapipe.solutions.pose` (human skeleton proxy — dogs use same landmarks roughly); OpenCV for frame capture |
| **Quality gate** | Minimum 1 pose landmark detected; otherwise returns None silently |
| **Note** | MediaPipe is trained on humans; dog body language uses proxy features (general posture metrics, not specific tail/ear detection) |

---

## HuggingFace Model Selection

| Model | Task | Benchmark | Selection Rationale |
|-------|------|-----------|---------------------|
| `facebook/wav2vec2-base` | Audio embedding (768-dim) | SUPERB Phone Error Rate 6.1% | Self-supervised; domain-transferable to animal sounds; strong general audio representation |
| `openai/whisper-large-v3` | Speech detection gate | WER 2.7% LibriSpeech | Robust voice activity detection; used to filter human speech from dog audio |
| `BAAI/bge-large-en-v1.5` | Text embedding | MTEB 64.23 (rank #3) | Best open-source embedding for knowledge retrieval |
| `facebook/bart-large-cnn` | Summarization | ROUGE-L 40.9 | Condense multiple research papers into coherent context for LLM |

---

## LLM Prompt Templates

### INTERPRETATION_PROMPT
```
You are an expert in canine ethology and animal behavior science.
A dog vocalization has been analyzed and classified.

Classification: {behavior_label} (confidence: {confidence:.0%})
Secondary signal: {secondary_label} ({secondary_confidence:.0%})
Acoustic features: spectral centroid={spectral_centroid:.0f}Hz, RMS={rms_energy:.4f}, tempo={tempo:.1f}bpm
Dog profile: breed={breed}, age={age}
Research context: {research_context}

Respond ONLY in JSON:
{{
  "narrative": "2-3 sentence plain-language explanation of what the dog is communicating",
  "breed_context": "breed-specific behavioral context if relevant (1 sentence, empty string if breed unknown)",
  "guidance_steps": ["step 1 for owner", "step 2", "step 3"],
  "urgency_level": "low|medium|high",
  "confidence_explanation": "1 sentence about what features drove this classification"
}}
```

### RESEARCH_SYNTHESIS_PROMPT
```
Synthesize these research findings about canine behavior for owner-facing guidance:
{paper_summaries}
Focus on: practical owner implications, key behavioral signals, welfare considerations.
Max 150 words.
```

---

## E2E Execution Flow

```
1. Input validation
   ├── File: path exists, format supported, duration > 0.1s
   └── Mic: sounddevice available, not silent (RMS > 0.001)

2. Audio feature extraction (AudioAnalyzer)
   ├── Load/capture → float32 array at 16kHz
   ├── Compute MFCC(40) + delta + delta2
   ├── Compute chroma(12), spectral stats, ZCR, RMS, tempo
   └── [Optional] wav2vec2 embedding via HFModelManager

3. Visual feature extraction (VisualAnalyzer, if video provided)
   └── MediaPipe pose landmarks → proxy body features

4. Behavior classification (BehaviorClassifier)
   ├── Concatenate feature vector
   ├── Try SVM ensemble (if trained model exists)
   ├── Try RandomForest (if trained model exists)
   ├── Fallback: HeuristicClassifier (always available)
   └── Vote + calibrate confidence

5. Knowledge retrieval (InterpretationEngine)
   ├── Embed behavior_label + acoustic description → BGE query
   ├── Cosine search SECOND-KNOWLEDGE-BRAIN.md paper index
   └── BART-CNN summarize top-3 papers → research context

6. LLM interpretation (InterpretationEngine)
   ├── Build INTERPRETATION_PROMPT with all context
   ├── Call Claude (primary) → parse JSON
   ├── Fallback to OpenAI → parse JSON
   └── Fallback to template-based response (if all APIs down)

7. Safety gate check
   ├── pain/distress + confidence > 0.70 → prepend vet warning
   └── aggression → prepend bite safety warning

8. Persist to memory (MemoryManager)
   ├── Save session: features, classification, guidance, cost
   └── Update dog profile behavior history if dog_id provided

9. Return structured output
   └── BehaviorAnalysisResult dataclass → JSON / CLI display
```

---

## Quality Gates

1. **Audio validity gate:** RMS > 0.001, duration > 0.1s — reject silent/too-short clips
2. **Dog vocalization gate:** spectral centroid 50–8000 Hz, ZCR > 0.01 — warn if audio is likely not dog
3. **Classification confidence gate:** confidence < 0.40 → return "uncertain" with top-2 candidates
4. **Safety gate:** pain/distress, aggression → always append professional advice
5. **LLM response gate:** JSON parse validation; if malformed → retry once, then use template fallback
6. **Knowledge gate:** if SECOND-KNOWLEDGE-BRAIN.md has < 5 papers → use fallback context only
7. **Cost gate:** warn if single session > $0.10 LLM cost; log all costs per session

---

## Test Scenarios

See `tests/test-scenarios.md` for 8 full end-to-end scenarios.

---

## Key Design Decisions

1. **Heuristic classifier as always-on baseline:** The SVM/RF classifiers need training data. The rule-based HeuristicClassifier uses ethological acoustic rules and is always available — ensuring the agent works on first launch without any training data.
2. **wav2vec2 is optional, not required:** If HuggingFace or GPU is unavailable, the 138-dim MFCC feature vector alone is sufficient for classification. wav2vec2 improves accuracy by ~8% on comparable audio tasks.
3. **MediaPipe uses human model for dogs:** There is no production-quality dog pose estimation model. MediaPipe Pose (human) provides general body orientation features that are still informative for posture classification even if not anatomically precise for dogs.
4. **SQLite WAL for concurrent writes:** The monitoring loop + REST API + CLI can all write concurrently. WAL mode prevents database locks without performance penalty.
5. **Breed context is optional:** If breed is unknown, InterpretationEngine omits breed_context gracefully rather than generating hallucinated breed-specific advice.
6. **Safety-first for pain/distress:** The agent never downgrades urgency for pain signals. False positives (over-cautious vet recommendations) are acceptable; false negatives are not.
7. **Research crawl is weekly, not daily:** Dog behavior literature advances more slowly than LLM/ML fields. Weekly crawl is sufficient and reduces API costs.
