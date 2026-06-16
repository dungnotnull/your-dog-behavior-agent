"""
Persistent memory manager for the Dog Behavior Agent.
SQLite WAL mode with thread safety.
Tables: analysis_sessions, dog_profiles, behavior_patterns, llm_cost_log, knowledge_hashes
"""

import sqlite3
import threading
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent.parent.parent / "data" / "memory.db"


def _ensure_data_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class MemoryManager:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        _ensure_data_dir()
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    audio_source TEXT,
                    duration_seconds REAL,
                    behavior_label TEXT,
                    confidence REAL,
                    secondary_label TEXT,
                    urgency_level TEXT,
                    model_used TEXT,
                    has_wav2vec2 INTEGER,
                    has_visual INTEGER,
                    llm_provider TEXT,
                    dog_id TEXT,
                    feature_vector_dim INTEGER,
                    spectral_centroid REAL,
                    rms_energy REAL,
                    tempo REAL,
                    zcr REAL
                );

                CREATE TABLE IF NOT EXISTS dog_profiles (
                    dog_id TEXT PRIMARY KEY,
                    name TEXT,
                    breed TEXT,
                    age TEXT,
                    session_count INTEGER DEFAULT 0,
                    most_common_behavior TEXT,
                    last_seen TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dog_id TEXT,
                    behavior_label TEXT,
                    count INTEGER DEFAULT 1,
                    last_seen TEXT,
                    UNIQUE(dog_id, behavior_label)
                );

                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    task TEXT,
                    session_id TEXT
                );

                CREATE TABLE IF NOT EXISTS knowledge_hashes (
                    hash TEXT PRIMARY KEY,
                    title TEXT,
                    added_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON analysis_sessions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_sessions_dog_id ON analysis_sessions(dog_id);
                CREATE INDEX IF NOT EXISTS idx_cost_provider ON llm_cost_log(provider);
            """)
            conn.commit()
            conn.close()

    def save_session(
        self,
        session_id: str,
        audio_source: str,
        duration_seconds: float,
        behavior_label: str,
        confidence: float,
        secondary_label: str,
        urgency_level: str,
        model_used: str,
        has_wav2vec2: bool,
        has_visual: bool,
        llm_provider: str,
        dog_id: Optional[str],
        feature_vector_dim: int,
        spectral_centroid: float,
        rms_energy: float,
        tempo: float,
        zcr: float,
    ):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR REPLACE INTO analysis_sessions
                   (session_id, timestamp, audio_source, duration_seconds, behavior_label,
                    confidence, secondary_label, urgency_level, model_used, has_wav2vec2,
                    has_visual, llm_provider, dog_id, feature_vector_dim,
                    spectral_centroid, rms_energy, tempo, zcr)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    audio_source,
                    duration_seconds,
                    behavior_label,
                    confidence,
                    secondary_label,
                    urgency_level,
                    model_used,
                    int(has_wav2vec2),
                    int(has_visual),
                    llm_provider,
                    dog_id,
                    feature_vector_dim,
                    spectral_centroid,
                    rms_energy,
                    tempo,
                    zcr,
                ),
            )
            conn.commit()
            conn.close()

        if dog_id:
            self._update_dog_profile(dog_id, behavior_label)

    def _update_dog_profile(self, dog_id: str, behavior_label: str):
        with self._lock:
            conn = self._connect()
            now = datetime.utcnow().isoformat()
            conn.execute(
                """INSERT INTO dog_profiles (dog_id, session_count, last_seen)
                   VALUES (?,1,?) ON CONFLICT(dog_id) DO UPDATE SET
                   session_count=session_count+1, last_seen=excluded.last_seen""",
                (dog_id, now),
            )
            conn.execute(
                """INSERT INTO behavior_patterns (dog_id, behavior_label, count, last_seen)
                   VALUES (?,?,1,?) ON CONFLICT(dog_id, behavior_label) DO UPDATE SET
                   count=count+1, last_seen=excluded.last_seen""",
                (dog_id, behavior_label, now),
            )
            # Update most common behavior
            row = conn.execute(
                "SELECT behavior_label FROM behavior_patterns WHERE dog_id=? ORDER BY count DESC LIMIT 1",
                (dog_id,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE dog_profiles SET most_common_behavior=? WHERE dog_id=?",
                    (row[0], dog_id),
                )
            conn.commit()
            conn.close()

    def upsert_dog_profile(
        self, dog_id: str, name: str = "", breed: str = "", age: str = "", notes: str = ""
    ):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO dog_profiles (dog_id, name, breed, age, notes, last_seen)
                   VALUES (?,?,?,?,?,?) ON CONFLICT(dog_id) DO UPDATE SET
                   name=COALESCE(NULLIF(excluded.name,''),name),
                   breed=COALESCE(NULLIF(excluded.breed,''),breed),
                   age=COALESCE(NULLIF(excluded.age,''),age),
                   notes=COALESCE(NULLIF(excluded.notes,''),notes),
                   last_seen=excluded.last_seen""",
                (dog_id, name, breed, age, notes, datetime.utcnow().isoformat()),
            )
            conn.commit()
            conn.close()

    def get_dog_profile(self, dog_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM dog_profiles WHERE dog_id=?", (dog_id,)
            ).fetchone()
            conn.close()
            return dict(row) if row else None

    def get_recent_sessions(self, limit: int = 20, dog_id: Optional[str] = None) -> List[Dict]:
        with self._lock:
            conn = self._connect()
            if dog_id:
                rows = conn.execute(
                    "SELECT * FROM analysis_sessions WHERE dog_id=? ORDER BY timestamp DESC LIMIT ?",
                    (dog_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM analysis_sessions ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        task: str = "",
        session_id: str = "",
    ):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO llm_cost_log
                   (timestamp, provider, model, input_tokens, output_tokens, cost_usd, task, session_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    datetime.utcnow().isoformat(),
                    provider, model, input_tokens, output_tokens, cost_usd, task, session_id,
                ),
            )
            conn.commit()
            conn.close()

    def get_cost_summary(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                """SELECT provider, SUM(cost_usd) as total_usd, SUM(input_tokens+output_tokens) as total_tokens,
                   COUNT(*) as call_count FROM llm_cost_log
                   WHERE timestamp >= datetime('now','-30 days') GROUP BY provider"""
            ).fetchall()
            conn.close()
            return {r["provider"]: {"total_usd": r["total_usd"], "total_tokens": r["total_tokens"],
                                    "call_count": r["call_count"]} for r in rows}

    def is_known_paper(self, title: str, doi: str = "") -> bool:
        content = (title + doi).encode()
        h = hashlib.sha256(content).hexdigest()
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT 1 FROM knowledge_hashes WHERE hash=?", (h,)).fetchone()
            conn.close()
            return row is not None

    def mark_paper_known(self, title: str, doi: str = ""):
        content = (title + doi).encode()
        h = hashlib.sha256(content).hexdigest()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes (hash, title, added_at) VALUES (?,?,?)",
                (h, title[:255], datetime.utcnow().isoformat()),
            )
            conn.commit()
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            sessions = conn.execute("SELECT COUNT(*) as c FROM analysis_sessions").fetchone()["c"]
            dogs = conn.execute("SELECT COUNT(*) as c FROM dog_profiles").fetchone()["c"]
            papers = conn.execute("SELECT COUNT(*) as c FROM knowledge_hashes").fetchone()["c"]
            behavior_counts = conn.execute(
                "SELECT behavior_label, COUNT(*) as c FROM analysis_sessions GROUP BY behavior_label ORDER BY c DESC"
            ).fetchall()
            conn.close()
            return {
                "total_sessions": sessions,
                "dog_profiles": dogs,
                "known_papers": papers,
                "behavior_distribution": {r["behavior_label"]: r["c"] for r in behavior_counts},
            }
