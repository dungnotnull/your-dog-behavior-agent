# Dog Behavior Agent — Test Scenarios

## Scenario 1: Golden Path — Analyze Alert Bark (Heuristic Classifier)
**Setup:** Audio file `tests/fixtures/alert_bark.wav` (3 seconds, single deep bark)
**Steps:**
1. Run `python -m agent.main analyze tests/fixtures/alert_bark.wav --breed "Labrador" --age "3 years"`
2. Agent extracts MFCC + spectral features (wav2vec2 optional)
3. HeuristicClassifier evaluates: centroid ~600 Hz, RMS ~0.04, tempo ~60 bpm
4. Classification: `alert_warning` with confidence ≥ 0.65
5. InterpretationEngine retrieves 2 relevant papers from SECOND-KNOWLEDGE-BRAIN.md
6. Claude LLM generates JSON interpretation with 3 guidance steps
7. Urgency level: "low" (alert bark is normal sentinel behavior)
**Expected output:**
```
Behavior: ALERT WARNING (confidence: 67%)
Narrative: Your dog has detected something novel in the environment...
Guidance: 1. Acknowledge your dog calmly...
```
**Pass criteria:** behavior_label = "alert_warning", confidence ≥ 0.55, urgency = "low", 3 guidance steps present

---

## Scenario 2: Pain/Distress Safety Gate Activation
**Setup:** Audio file `tests/fixtures/distress_whine.wav` (sustained high-pitch whine, 5 seconds)
**Steps:**
1. Run `python -m agent.main analyze tests/fixtures/distress_whine.wav --json-output`
2. Classifier identifies: centroid > 1500 Hz, low energy, sustained
3. Classification: `pain_distress` with confidence ≥ 0.70
4. Safety gate activates: urgency_level forced to "high"
5. Vet recommendation prepended to guidance_steps
**Expected output (JSON):**
```json
{
  "behavior_label": "pain_distress",
  "urgency_level": "high",
  "guidance_steps": ["URGENT: Contact your veterinarian immediately...", ...]
}
```
**Pass criteria:** urgency = "high", guidance_steps[0] contains "vet" or "veterinar"

---

## Scenario 3: Silent Audio Rejected by Quality Gate
**Setup:** Audio file `tests/fixtures/silent.wav` (3 seconds of silence, RMS ≈ 0.0001)
**Steps:**
1. Run `python -m agent.main analyze tests/fixtures/silent.wav --json-output`
2. AudioAnalyzer validates: RMS < 0.001 → reject
3. Agent returns invalid result without calling classifier or LLM
**Expected output:**
```json
{
  "behavior_label": "invalid",
  "valid_audio": false,
  "validation_message": "Audio appears to be silent..."
}
```
**Pass criteria:** valid_audio = false, no LLM call made, latency < 200ms

---

## Scenario 4: All LLM Providers Unavailable — Fallback Template
**Setup:** Set invalid API keys (ANTHROPIC_API_KEY=invalid, OPENAI_API_KEY=invalid, Ollama not running)
**Steps:**
1. Provide audio with excitement pattern (high pitch, rapid tempo)
2. Classifier correctly identifies: `excitement`
3. All LLM providers fail (authentication errors + connection refused)
4. InterpretationEngine falls back to FALLBACK_INTERPRETATIONS["excitement"]
5. Returns valid BehaviorAnalysisResult with llm_provider = "fallback"
**Expected output:**
- behavior_label = "excitement"
- narrative contains "positive excitement"
- guidance_steps has 3 steps
- llm_provider_used = "fallback"
**Pass criteria:** Full result returned without exception; llm_provider = "fallback"

---

## Scenario 5: Dog Profile Tracking + History
**Setup:** Fresh database (delete data/memory.db)
**Steps:**
1. Create dog profile: `python -m agent.main profile buddy --breed "German Shepherd" --age "2 years"`
2. Analyze 5 audio files with dog_id="buddy": 3 alert barks + 1 excitement + 1 fear
3. Run `python -m agent.main history buddy`
4. Check behavior distribution in database
**Expected output:**
```
[2026-06-11 12:00] alert_warning        67%  urgency=low
[2026-06-11 12:01] alert_warning        71%  urgency=low
...
```
**Pass criteria:** 5 sessions saved, behavior_distribution shows alert_warning=3, dog profile updated with most_common_behavior="alert_warning"

---

## Scenario 6: ArXiv Knowledge Crawler — Deduplication
**Setup:** Running knowledge base with 15 seed papers
**Steps:**
1. Run `python -m agent.main update-knowledge` (first run — all papers new)
2. Check SECOND-KNOWLEDGE-BRAIN.md: N new rows added to Key Research Papers table
3. Run `python -m agent.main update-knowledge` again immediately (second run)
4. Same papers found — all already in knowledge_hashes table
**Expected output (second run):**
```
Knowledge update complete: {'added': 0, 'skipped': 10, 'total_candidates': 100}
```
**Pass criteria:** added = 0 on second run; no duplicate rows in SECOND-KNOWLEDGE-BRAIN.md

---

## Scenario 7: Visual Analysis Integration (MediaPipe Optional)
**Setup:** Audio file + video file with visible dog in erect posture
**Steps:**
1. Run `python -m agent.main analyze audio.wav --video dog_video.mp4`
2. VisualAnalyzer extracts MediaPipe pose proxy features
3. VisualFeatures fused with AudioFeatures in BehaviorClassifier
4. Classification uses 141-dim vector (138 audio + 3 visual)
5. has_visual = true in result
**Expected output:**
- has_visual = true
- feature_vector_dim ≥ 141
**Pass criteria:** No crash; has_visual = true when video provided; result still valid if MediaPipe not installed (graceful skip)

---

## Scenario 8: REST API Full Integration Test
**Setup:** Server running at localhost:8009
**Steps:**
1. GET `/health` → `{"status": "ok"}`
2. POST `/api/v1/analyze/upload` with audio file multipart upload
3. POST `/api/v1/dogs/profile` with dog profile JSON
4. GET `/api/v1/dogs/buddy/sessions` → session list
5. GET `/api/v1/stats` → behavior distribution
6. GET `/metrics` → Prometheus text format
7. POST `/api/v1/knowledge/update` → `{"status": "ok", "papers_added": N}`
**Pass criteria:** All 7 endpoints return 200; classify + interpret cycle completes < 5s; Prometheus metrics include `dog_behavior_sessions_total`
