# SECOND-KNOWLEDGE-BRAIN.md — Dog Behavior Agent

> **Self-updating domain knowledge base.** Updated weekly via `tools/knowledge_updater.py`.
> The longer this agent runs, the more accurate and contextually rich its interpretations become.

---

## Core Concepts & Frameworks

### Canine Vocalization Ethology
Dogs use vocalizations as a primary communication channel with conspecifics and humans. The **Motivation-Structural Rules** (Morton 1977) describe a universal gradient: low-pitched, harsh sounds signal aggression/dominance while high-pitched, tonal sounds signal appeasement/submission. Dog barks are multi-functional: the same bark frequency can encode different meanings based on context (Yin & McCowan 2004).

### Key Acoustic Features for Canine Behavior Classification
- **Fundamental Frequency (F0):** Low F0 (100–400 Hz) → aggression/alert; High F0 (600–2000 Hz) → fear/pain/excitement
- **Mel-Frequency Cepstral Coefficients (MFCC):** Capture timbral characteristics that encode vocal tract shape
- **Zero Crossing Rate (ZCR):** High ZCR → noisy/rough sounds (barks); Low ZCR → tonal/whines
- **Spectral Centroid:** Frequency center of mass; higher in high-pitched vocalizations
- **Temporal Envelope (RMS + Tempo):** Rapid repetitive calls → excitement; sustained calls → distress
- **wav2vec2 Embeddings:** Self-supervised audio representations that capture latent acoustic structure beyond handcrafted features

### Attachment Theory in Dog-Human Bond
Dogs form secure attachment bonds with their owners (Topál et al. 1998). Separation anxiety vocalizations (whining, howling) are directly linked to attachment disruption. The owner is the "safe haven" — distress signals escalate until owner returns.

### 8-Class Behavioral Taxonomy (Ethological Basis)
| Class | Ethological Basis | Key References |
|-------|-------------------|----------------|
| `aggression` | Morton 1977 motivation-structural rules; resource guarding | Rugaas 2006 |
| `fear` | Threat avoidance system; high-pitch appeasement | Schenkel 1967 |
| `excitement` | Approach motivation; dopaminergic activation | Panksepp 1998 |
| `pain_distress` | Nociceptive vocalization; separation anxiety | Molnár et al. 2009 |
| `attention_seeking` | Learned instrumental behavior; owner reinforcement | Yin & McCowan 2004 |
| `play` | Play vocalizations distinct acoustic signature | Simonet et al. 2001 |
| `alert_warning` | Territorial behavior; sentinel function | Pongrácz et al. 2010 |
| `greeting` | Social bonding; affiliative behavior | Faragó et al. 2010 |

---

## Key Research Papers

| # | Title | Authors | Year | Venue | DOI/Link | Key Finding | Relevance |
|---|-------|---------|------|-------|----------|-------------|-----------|
| 1 | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Baevski et al. | 2020 | NeurIPS | https://arxiv.org/abs/2006.11477 | Self-supervised pre-training on unlabeled audio achieves SOTA on speech benchmarks; transferable representations | Audio embedding backbone for feature extraction |
| 2 | Barking in domestic dogs: context specificity and individual identification | Yin & McCowan | 2004 | Animal Behaviour | https://doi.org/10.1016/j.anbehav.2003.07.016 | Dog barks encode specific contexts; MFCC features differentiate bark types | Foundational validation for acoustic classification |
| 3 | Acoustic parameters of dog barks carry emotional valence for human listeners | Pongrácz et al. | 2010 | Applied Animal Behaviour Science | https://doi.org/10.1016/j.applanim.2010.06.006 | Humans correctly categorize dog emotional states from acoustic parameters | Justifies acoustic feature use for behavior classification |
| 4 | Classification of dog barks: a machine learning approach | Molnár et al. | 2008 | Animal Cognition | https://doi.org/10.1007/s10071-007-0129-9 | Machine learning achieves 43% accuracy on 6-class bark classification, better than untrained humans | Establishes ML feasibility and baseline accuracy |
| 5 | ESC-50: Dataset for Environmental Sound Classification | Piczak | 2015 | ACM Multimedia | https://doi.org/10.1145/2733373.2806390 | 50-class environmental sound dataset including dog/cat categories; MFCC + CNN achieves 73.7% | ESC-50 provides 50 dog bark examples for augmentation |
| 6 | PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition | Kong et al. | 2020 | IEEE TASLP | https://arxiv.org/abs/1912.10211 | CNN14 achieves mAP 0.431 on AudioSet; pretrained models transfer well to new sound domains | Audio CNN pretrained features as alternative to wav2vec2 |
| 7 | CNN architectures for large-scale audio classification | Hershey et al. | 2017 | ICASSP | https://arxiv.org/abs/1609.09430 | VGGish features from AudioSet pretraining provide strong transfer learning for sound classification | VGGish as additional audio feature option |
| 8 | On the occurrence and significance of motivation-structural rules in some bird and mammal sounds | Morton | 1977 | American Naturalist | https://doi.org/10.1086/283219 | Universal gradient: low-pitched harsh → aggression; high-pitched tonal → submission/fear | Theoretical foundation for 8-class taxonomy |
| 9 | Audio Set: An Ontology and Human-Labeled Dataset for Audio Events | Gemmeke et al. | 2017 | ICASSP | https://doi.org/10.1109/ICASSP.2017.7952261 | 632 audio event classes; 2M 10s YouTube clips; includes animal sounds | Large-scale pretraining data source for audio models |
| 10 | Dog bark recognition based on acoustic features and support vector machines | Kim et al. | 2019 | Applied Sciences | https://doi.org/10.3390/app9091743 | SVM with MFCC features achieves 84% accuracy on 5-class dog bark dataset | Validates SVM + MFCC as strong baseline |
| 11 | A study on the acoustic properties of the play pant of the domestic dog | Simonet et al. | 2001 | ASA Meeting | https://doi.org/10.1121/1.1399591 | Play vocalizations have distinct acoustic signature (forced breathy exhalation) separate from threat | Validates play as distinct acoustic class |
| 12 | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | https://arxiv.org/abs/1706.03762 | Transformer architecture; foundation for wav2vec2 and modern audio models | Architectural foundation of audio embedding models |
| 13 | Dogs' expectation about signalers' body size by virtue of their growls | Faragó et al. | 2010 | PLOS ONE | https://doi.org/10.1371/journal.pone.0015175 | Dog growls encode body size information; pitch predicts size across contexts | Acoustic features encode multiple information channels |
| 14 | The Emotional Affective Tone of Barks: Human Categorization and Physiological Correlates | Pongrácz et al. | 2017 | Animal Cognition | https://doi.org/10.1007/s10071-016-1058-4 | Humans distinguish emotional tone of barks across contexts reliably | Cross-cultural validity of bark emotion classification |
| 15 | Text-to-Audio Grounding | Ghosh et al. | 2023 | Interspeech | https://arxiv.org/abs/2309.03454 | Audio-text alignment models; relevant to grounding behavior descriptions in audio | Multimodal audio-text alignment for richer interpretation |

---

## State-of-the-Art Models

| Model | Task | Score | Venue/Date | HuggingFace ID |
|-------|------|-------|------------|----------------|
| wav2vec2-base | Audio representation | SUPERB PER 6.1% | NeurIPS 2020 | `facebook/wav2vec2-base` |
| wav2vec2-large-960h | Speech/audio embedding | SUPERB PER 3.4% | NeurIPS 2020 | `facebook/wav2vec2-large-960h` |
| Whisper-large-v3 | Speech recognition | WER 2.7% LibriSpeech | OpenAI 2023 | `openai/whisper-large-v3` |
| BGE-large-en-v1.5 | Text embedding | MTEB 64.23 | BAAI 2023 | `BAAI/bge-large-en-v1.5` |
| BART-large-CNN | Summarization | ROUGE-L 40.9 | Facebook 2020 | `facebook/bart-large-cnn` |
| CNN14 (PANN) | Audio classification | mAP 0.431 AudioSet | TASLP 2020 | `PANNs/CNN14` |
| BEATs | Audio classification | mAP 0.486 AudioSet | ICML 2023 | `microsoft/BEATs-iter3-AS2M` |

---

## LLM Prompt Patterns

### INTERPRETATION_PROMPT (Primary)
```
You are an expert in canine ethology and applied animal behavior science.
A dog vocalization recording has been analyzed:

Classification: {behavior_label} (confidence: {confidence:.0%})
Secondary signal: {secondary_label} ({secondary_confidence:.0%})
Acoustic features: centroid={spectral_centroid:.0f}Hz, energy={rms_energy:.4f}, tempo={tempo:.1f}bpm, ZCR={zcr:.4f}
Dog profile: breed={breed}, age={age}
Research context: {research_context}

Respond ONLY in valid JSON with exactly these keys:
{{"narrative": "...", "breed_context": "...", "guidance_steps": ["step1","step2","step3"], "urgency_level": "low|medium|high", "confidence_explanation": "..."}}
```

### FALLBACK_TEMPLATE_PROMPT (when LLM unavailable)
```python
FALLBACK_INTERPRETATIONS = {
    "aggression": {"narrative": "Your dog is showing signs of aggression or territorial behavior. This vocalization indicates discomfort or threat perception.", "urgency_level": "high"},
    "fear": {"narrative": "Your dog appears fearful or anxious. These vocalizations signal distress and a desire for safety.", "urgency_level": "medium"},
    "pain_distress": {"narrative": "Your dog may be in pain or significant distress. These vocalizations require immediate attention.", "urgency_level": "high"},
    ...
}
```

### RESEARCH_SYNTHESIS_PROMPT
```
Synthesize these canine behavior research findings for practical owner guidance:
{paper_summaries}
Focus: behavioral signals, owner response strategies, welfare implications.
Max 120 words. Plain language.
```

### BREED_CONTEXT_PROMPT
```
Based on this dog breed ({breed}), provide one sentence of breed-specific behavioral context
relevant to the classified behavior ({behavior_label}).
Focus on breed tendencies, historical working function, or known communication patterns.
If breed is unknown, respond with empty string.
```

---

## Authoritative Data Sources

| Source | URL | Access Method | Update Frequency |
|--------|-----|---------------|-----------------|
| ArXiv cs.SD + cs.LG + cs.CV | https://arxiv.org/search/ | XML API (free) | Daily |
| Semantic Scholar | https://api.semanticscholar.org/graph/v1/ | REST API (free, 100 req/5min) | Daily |
| Applied Animal Behaviour Science | https://www.sciencedirect.com/journal/applied-animal-behaviour-science | DOI metadata | Monthly |
| Animal Cognition (Springer) | https://link.springer.com/journal/10071 | DOI metadata | Monthly |
| Interspeech Proceedings | https://www.isca-archive.org/ | HTML scrape | Annual |
| BarkNet Dataset | https://www.kaggle.com/datasets | Kaggle API | Static (v1) |
| ESC-50 Dataset | https://github.com/karolpiczak/ESC-50 | GitHub releases | Static |
| Papers with Code — Audio Classification | https://paperswithcode.com/task/audio-classification | HTML scrape | Weekly |

---

## Self-Update Protocol

```yaml
knowledge_updater:
  schedule: "weekly Sunday 02:00 (APScheduler CronTrigger)"
  sources:
    arxiv:
      categories: ["cs.SD", "cs.CV", "cs.LG"]
      queries:
        - "dog vocalization classification"
        - "canine behavior audio recognition"
        - "animal sound classification deep learning"
        - "bioacoustics machine learning"
        - "MFCC animal vocalization"
    semantic_scholar:
      queries:
        - "dog bark classification neural network"
        - "canine acoustic behavior analysis"
        - "animal vocalization intent recognition"
        - "wav2vec audio animal sounds"
        - "dog behavior machine learning"
  scoring:
    recency_weight: 0.6
    relevance_weight: 0.4
    keywords: ["dog", "canine", "bark", "vocalization", "animal sound", "bioacoustics",
               "behavior classification", "MFCC", "wav2vec", "audio classification"]
    recency_window_days: 90
  deduplication:
    method: SHA256 of (title + doi/url)
    stored_in: memory.db knowledge_hashes table
  top_n_per_run: 10
  max_total_papers: 500
```

---

## Knowledge Update Log

| Date | Papers Added | New Sources | Notes |
|------|-------------|-------------|-------|
| 2026-06-11 | 15 (seed) | ArXiv, Semantic Scholar | Initial seed: 15 foundational papers on canine behavior, audio classification, wav2vec2, and ethology |
