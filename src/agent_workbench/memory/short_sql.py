from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from pydantic import BaseModel

from agent_workbench.settings import Settings


class MessageRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    tag: Optional[str] = "episodic"


class ToolEvent(BaseModel):
    id: Optional[int] = None
    session_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime


class ReflectionRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    step_number: int
    reflection_text: str
    usefulness_score: float
    memory_updates: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ShortTermMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = Path(settings.paths.sqlite_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    tag TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_input TEXT NOT NULL,
                    tool_output TEXT,
                    error TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    reflection_text TEXT NOT NULL,
                    usefulness_score REAL,
                    memory_updates TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            await db.commit()
            try:
                await db.execute("ALTER TABLE messages ADD COLUMN tag TEXT")
                await db.commit()
            except Exception:
                pass
    
    async def create_session(self, session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO sessions (id) VALUES (?)",
                (session_id,)
            )
            await db.commit()
    
    async def add_message(self, message: MessageRecord) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            metadata_json = json.dumps(message.metadata) if message.metadata else None
            async with db.execute("PRAGMA table_info(messages)") as cursor:
                cols = [row[1] async for row in cursor]
            if "tag" in cols:
                await db.execute(
                    """INSERT INTO messages (session_id, role, content, timestamp, metadata, tag)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (message.session_id, message.role, message.content, message.timestamp, metadata_json, message.tag)
                )
            else:
                await db.execute(
                    """INSERT INTO messages (session_id, role, content, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (message.session_id, message.role, message.content, message.timestamp, metadata_json)
                )
            await db.commit()
    
    async def add_tool_event(self, event: ToolEvent) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            input_json = json.dumps(event.tool_input)
            output_json = json.dumps(event.tool_output) if event.tool_output else None
            await db.execute(
                """INSERT INTO tool_events (session_id, tool_name, tool_input, tool_output, error, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event.session_id, event.tool_name, input_json, output_json, event.error, event.timestamp)
            )
            await db.commit()
    
    async def add_reflection(self, reflection: ReflectionRecord) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            updates_json = json.dumps(reflection.memory_updates) if reflection.memory_updates else None
            await db.execute(
                """INSERT INTO reflections (session_id, step_number, reflection_text, usefulness_score, memory_updates, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (reflection.session_id, reflection.step_number, reflection.reflection_text,
                 reflection.usefulness_score, updates_json, reflection.timestamp)
            )
            await db.commit()
    
    async def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[MessageRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = """
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query, (session_id,)) as cursor:
                rows = await cursor.fetchall()
                
            messages = []
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                tag_value = row["tag"] if "tag" in row.keys() else "episodic"
                messages.append(MessageRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=metadata,
                    tag=tag_value
                ))
            
            return messages[::-1]  # Reverse to get chronological order
    
    async def get_tool_events(self, session_id: str, limit: Optional[int] = None) -> List[ToolEvent]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            query = """
                SELECT * FROM tool_events 
                WHERE session_id = ? 
                ORDER BY timestamp DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query, (session_id,)) as cursor:
                rows = await cursor.fetchall()
                
            events = []
            for row in rows:
                tool_input = json.loads(row["tool_input"])
                tool_output = json.loads(row["tool_output"]) if row["tool_output"] else None
                events.append(ToolEvent(
                    id=row["id"],
                    session_id=row["session_id"],
                    tool_name=row["tool_name"],
                    tool_input=tool_input,
                    tool_output=tool_output,
                    error=row["error"],
                    timestamp=datetime.fromisoformat(row["timestamp"])
                ))
            
            return events[::-1]  # Reverse to get chronological order
    
    async def get_reflections(self, session_id: str) -> List[ReflectionRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            async with db.execute(
                "SELECT * FROM reflections WHERE session_id = ? ORDER BY step_number",
                (session_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                
            reflections = []
            for row in rows:
                memory_updates = json.loads(row["memory_updates"]) if row["memory_updates"] else None
                reflections.append(ReflectionRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    step_number=row["step_number"],
                    reflection_text=row["reflection_text"],
                    usefulness_score=row["usefulness_score"],
                    memory_updates=memory_updates,
                    timestamp=datetime.fromisoformat(row["timestamp"])
                ))
            
            return reflections
    
    async def summarize_session(self, session_id: str) -> str:
        messages = await self.get_session_history(session_id)
        reflections = await self.get_reflections(session_id)
        
        summary_parts = []
        
        if messages:
            summary_parts.append(f"Session had {len(messages)} messages.")
            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]
            summary_parts.append(f"User: {len(user_messages)}, Assistant: {len(assistant_messages)}")
        
        if reflections:
            avg_usefulness = sum(r.usefulness_score for r in reflections) / len(reflections)
            summary_parts.append(f"Average reflection usefulness: {avg_usefulness:.2f}")
        
        return "; ".join(summary_parts) if summary_parts else "No session data found."
