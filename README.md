# Dog Behavior Agent

AI-powered real-time analysis of dog vocalizations and body-language proxies.
Classifies dog sounds into one of eight ethologically-grounded behavior classes,
retrieves relevant peer-reviewed research, and generates plain-language owner
guidance via Claude / OpenAI / Ollama with robust offline fallbacks.

## Key Features

- **Audio analysis**: MFCC (40) + deltas + chroma + spectral statistics, with optional `wav2vec2` embeddings.
- **8-class taxonomy**: `aggression`, `fear`, `excitement`, `pain_distress`, `attention_seeking`, `play`, `alert_warning`, `greeting`.
- **Ensemble classification**: SVM + RandomForest + always-available heuristic fallback.
- **Visual proxy**: Optional MediaPipe Pose posture/motion features.
- **LLM interpretation**: Unified Claude/OpenAI/Ollama client with JSON validation, retry logic, and cost tracking.
- **Research knowledge base**: Weekly ArXiv + Semantic Scholar crawler with SHA256 deduplication and FAISS+BGE retrieval.
- **REST API + CLI**: FastAPI server with health/metrics endpoints, plus Click CLI for file, microphone, profile, and knowledge management.
- **Production ready**: Docker + docker-compose, Prometheus metrics, SQLite WAL memory, structured logging.

## Quickstart

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Copy environment template and configure optional API keys
cp config/.env.example .env

# 3. Analyze an audio file (no LLM keys required; uses fallback templates)
python -m agent.main analyze tests/fixtures/alert_bark.wav --breed "Labrador"

# 4. Start the REST server
python -m agent.main serve --host 0.0.0.0 --port 8009
```

## Architecture

```
Audio file / Microphone
        |
        v
AudioAnalyzer (librosa + optional wav2vec2)
        |
        v
BehaviorClassifier (SVM + RF + heuristic)
        |
        v
InterpretationEngine (BGE/FAISS retrieval + Claude/OpenAI/Ollama JSON)
        |
        v
MemoryManager (SQLite WAL) + REST/CLI output
```

## Configuration

Runtime behavior is controlled by `config/agent_config.yaml` and environment
variables in `.env`. See `config/.env.example` for all supported variables.

Key flags:

- `USE_WAV2VEC2=true|false` — enable/disable wav2vec2 audio embeddings.
- `USE_VISUAL_ANALYSIS=true|false` — enable/disable MediaPipe visual analysis.
- `PRIVACY_MODE=true|false` — force local Ollama only (no cloud LLM calls).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Service health check |
| POST   | `/api/v1/analyze/file` | Analyze a local audio file |
| POST   | `/api/v1/analyze/upload` | Upload and analyze an audio file |
| POST   | `/api/v1/analyze/microphone` | Record from microphone and analyze |
| POST   | `/api/v1/dogs/profile` | Create or update dog profile |
| GET    | `/api/v1/dogs/{dog_id}/profile` | Retrieve dog profile |
| GET    | `/api/v1/dogs/{dog_id}/sessions` | Retrieve recent sessions |
| POST   | `/api/v1/knowledge/update` | Trigger knowledge-base crawl |
| GET    | `/api/v1/stats` | Session / behavior statistics |
| GET    | `/api/v1/cost` | LLM cost summary (last 30 days) |
| GET    | `/metrics` | Prometheus text metrics |

## CLI

```bash
python -m agent.main analyze <audio.wav> [--breed <breed>] [--age <age>] [--video <video.mp4>]
python -m agent.main listen --duration 3
python -m agent.main profile <dog_id> --breed <breed> --age <age>
python -m agent.main history <dog_id>
python -m agent.main update-knowledge
python -m agent.main rebuild-knowledge-index
python -m agent.main serve --host 0.0.0.0 --port 8009 --start-scheduler
```

## Testing

```bash
# Fast unit/integration tests (no model download required by default)
pytest tests/test_agent.py -v
```

## Docker

```bash
cd docker
docker-compose --profile ollama up --build
```

The stack includes the `dog-behavior-agent` service and an optional `ollama`
local LLM backend. A GPU profile is available via `--profile gpu`.

## Knowledge Base

The agent maintains `SECOND-KNOWLEDGE-BRAIN.md` as a living literature summary.
Run `update-knowledge` to crawl ArXiv and Semantic Scholar, then
`rebuild-knowledge-index` to create a FAISS index for fast BGE retrieval.

## License

MIT License — see `LICENSE` for details.
