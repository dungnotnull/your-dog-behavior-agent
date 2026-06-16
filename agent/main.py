"""
Dog Behavior Agent — Entry Point
CLI interface + FastAPI REST server.
"""

import json
import os
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import click
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional

from agent.orchestrator import DogBehaviorOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dog Behavior Agent",
    description="Real-time dog vocalization analysis via ML/DL + LLM interpretation",
    version="1.0.0",
)

_orchestrator: Optional[DogBehaviorOrchestrator] = None


def get_orchestrator() -> DogBehaviorOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DogBehaviorOrchestrator()
    return _orchestrator


class AnalyzeFileRequest(BaseModel):
    audio_path: str
    video_path: Optional[str] = None
    breed: str = "unknown"
    age: str = "unknown"
    dog_id: Optional[str] = None
    use_wav2vec2: bool = True
    show_all_probabilities: bool = False


class MicrophoneRequest(BaseModel):
    duration: float = 3.0
    breed: str = "unknown"
    age: str = "unknown"
    dog_id: Optional[str] = None
    use_wav2vec2: bool = True


class DogProfileRequest(BaseModel):
    dog_id: str
    name: str = ""
    breed: str = ""
    age: str = ""
    notes: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "service": "dog-behavior-agent"}


@app.post("/api/v1/analyze/file")
async def analyze_file(req: AnalyzeFileRequest):
    """Analyze a local audio file (+ optional video)."""
    if not Path(req.audio_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.audio_path}")
    result = get_orchestrator().analyze_file(
        audio_path=req.audio_path,
        video_path=req.video_path,
        breed=req.breed,
        age=req.age,
        dog_id=req.dog_id,
        use_wav2vec2=req.use_wav2vec2,
        show_all_probabilities=req.show_all_probabilities,
    )
    return JSONResponse(result.to_dict())


@app.post("/api/v1/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    breed: str = Form("unknown"),
    age: str = Form("unknown"),
    dog_id: Optional[str] = Form(None),
    use_wav2vec2: bool = Form(True),
):
    """Upload an audio file for analysis (saves to temp dir)."""
    import tempfile
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = get_orchestrator().analyze_file(
            audio_path=tmp_path,
            breed=breed,
            age=age,
            dog_id=dog_id,
            use_wav2vec2=use_wav2vec2,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return JSONResponse(result.to_dict())


@app.post("/api/v1/analyze/microphone")
async def analyze_microphone(req: MicrophoneRequest):
    """Record from microphone and analyze."""
    result = get_orchestrator().analyze_microphone(
        duration=req.duration,
        breed=req.breed,
        age=req.age,
        dog_id=req.dog_id,
        use_wav2vec2=req.use_wav2vec2,
    )
    return JSONResponse(result.to_dict())


@app.post("/api/v1/dogs/profile")
async def upsert_dog_profile(req: DogProfileRequest):
    """Create or update a dog profile."""
    get_orchestrator()._get_memory().upsert_dog_profile(
        dog_id=req.dog_id, name=req.name, breed=req.breed, age=req.age, notes=req.notes
    )
    return {"status": "ok", "dog_id": req.dog_id}


@app.get("/api/v1/dogs/{dog_id}/profile")
async def get_dog_profile(dog_id: str):
    profile = get_orchestrator()._get_memory().get_dog_profile(dog_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Dog profile not found: {dog_id}")
    return profile


@app.get("/api/v1/dogs/{dog_id}/sessions")
async def get_dog_sessions(dog_id: str, limit: int = 20):
    sessions = get_orchestrator()._get_memory().get_recent_sessions(limit=limit, dog_id=dog_id)
    return {"sessions": sessions}


@app.post("/api/v1/knowledge/update")
async def update_knowledge():
    """Trigger a manual knowledge base update."""
    result = get_orchestrator().update_knowledge()
    return {"status": "ok", "papers_added": result.get("added", 0)}


@app.get("/api/v1/cost")
async def get_cost():
    return get_orchestrator().get_cost_report()


@app.get("/api/v1/stats")
async def get_stats():
    return get_orchestrator().get_stats()


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        get_orchestrator().get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4",
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Dog Behavior Agent — AI-powered canine vocalization analysis."""


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--video", default=None, help="Optional video file for body language analysis")
@click.option("--breed", default="unknown", help="Dog breed (for context)")
@click.option("--age", default="unknown", help="Dog age (for context)")
@click.option("--dog-id", default=None, help="Dog ID to track history")
@click.option("--no-wav2vec2", is_flag=True, default=False, help="Skip wav2vec2 embedding (faster)")
@click.option("--probabilities", is_flag=True, default=False, help="Show all class probabilities")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON")
def analyze(audio_path, video, breed, age, dog_id, no_wav2vec2, probabilities, json_output):
    """Analyze a dog audio file and interpret the behavior."""
    orch = DogBehaviorOrchestrator()
    result = orch.analyze_file(
        audio_path=audio_path,
        video_path=video,
        breed=breed,
        age=age,
        dog_id=dog_id,
        use_wav2vec2=not no_wav2vec2,
        show_all_probabilities=probabilities,
    )
    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        interp = orch._get_interpreter()
        from agent.modules.behavior_classifier import ClassificationResult
        from agent.modules.interpretation_engine import InterpretationResult
        # Build display objects
        from dataclasses import dataclass as dc_
        mock_cls = type("CR", (), {
            "behavior_label": result.behavior_label,
            "confidence": result.confidence,
            "secondary_label": result.secondary_label,
            "secondary_confidence": result.confidence,
            "all_probabilities": result.all_probabilities,
            "model_used": result.model_used,
        })()
        mock_interp = type("IR", (), {
            "narrative": result.narrative,
            "breed_context": result.breed_context,
            "guidance_steps": result.guidance_steps,
            "urgency_level": result.urgency_level,
            "confidence_explanation": result.confidence_explanation,
        })()
        click.echo(interp.format_result(mock_interp, mock_cls, include_probabilities=probabilities))
        click.echo(f"\nSession ID: {result.session_id}  |  Latency: {result.latency_ms:.0f}ms")


@cli.command()
@click.option("--duration", default=3.0, help="Recording duration in seconds")
@click.option("--breed", default="unknown")
@click.option("--age", default="unknown")
@click.option("--dog-id", default=None)
def listen(duration, breed, age, dog_id):
    """Record from microphone and analyze dog vocalizations."""
    click.echo(f"Recording {duration}s from microphone...")
    orch = DogBehaviorOrchestrator()
    result = orch.analyze_microphone(duration=duration, breed=breed, age=age, dog_id=dog_id)
    click.echo(f"\nBehavior: {result.behavior_label.upper()} ({result.confidence:.0%})")
    click.echo(f"Urgency: {result.urgency_level}")
    click.echo(f"\n{result.narrative}\n")
    for i, step in enumerate(result.guidance_steps, 1):
        click.echo(f"  {i}. {step}")


@cli.command()
@click.argument("dog_id")
@click.option("--name", default="")
@click.option("--breed", default="")
@click.option("--age", default="")
@click.option("--notes", default="")
def profile(dog_id, name, breed, age, notes):
    """Create or update a dog profile."""
    orch = DogBehaviorOrchestrator()
    orch._get_memory().upsert_dog_profile(dog_id=dog_id, name=name, breed=breed, age=age, notes=notes)
    click.echo(f"Profile updated for dog_id={dog_id}")


@cli.command()
@click.argument("dog_id")
def history(dog_id):
    """Show recent behavior analysis sessions for a dog."""
    orch = DogBehaviorOrchestrator()
    sessions = orch._get_memory().get_recent_sessions(limit=10, dog_id=dog_id)
    if not sessions:
        click.echo(f"No sessions found for dog_id={dog_id}")
        return
    for s in sessions:
        click.echo(f"[{s['timestamp'][:16]}] {s['behavior_label']:20} {s['confidence']:.0%}  urgency={s['urgency_level']}")


@cli.command(name="update-knowledge")
def update_knowledge_cmd():
    """Manually trigger research paper knowledge base update."""
    orch = DogBehaviorOrchestrator()
    click.echo("Crawling ArXiv and Semantic Scholar...")
    result = orch.update_knowledge()
    click.echo(f"Done. Papers added: {result.get('added', 0)}")


@cli.command(name="cost-report")
def cost_report():
    """Show LLM API cost summary (last 30 days)."""
    orch = DogBehaviorOrchestrator()
    report = orch.get_cost_report()
    if not report:
        click.echo("No LLM cost data yet.")
        return
    for provider, data in report.items():
        click.echo(f"{provider}: ${data['total_usd']:.4f}  tokens={data['total_tokens']}  calls={data['call_count']}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8009, type=int)
@click.option("--start-scheduler", is_flag=True, default=False)
def serve(host, port, start_scheduler):
    """Start the FastAPI REST server."""
    if start_scheduler:
        click.echo("Starting background scheduler for weekly knowledge updates...")
        get_orchestrator().start_scheduler()
    click.echo(f"Starting Dog Behavior Agent server on {host}:{port}")
    uvicorn.run("agent.main:app", host=host, port=port, reload=False)



@cli.command(name="rebuild-knowledge-index")
def rebuild_knowledge_index_cmd():
    """Rebuild the FAISS knowledge index from SECOND-KNOWLEDGE-BRAIN.md."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.faiss_index_manager import build_knowledge_index
    result = build_knowledge_index()
    click.echo(f"Index status: {result['status']}, papers indexed: {result['indexed_count']}")

if __name__ == "__main__":
    cli()
