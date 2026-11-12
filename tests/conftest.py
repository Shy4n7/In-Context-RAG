"""Shared fixtures.

The Gemini client is replaced with a fake so tests never hit the network and
do not need the llama-index package or an API key.
"""
import sys
import types

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool


class FakeLLM:
    """Scriptable stand-in for llama_index GoogleGenAI."""

    calls = []          # list of (model, prompt)
    behaviour = {}      # model -> exception instance or reply string
    default_reply = "Stub answer."

    def __init__(self, model, api_key=None):
        self.model = model

    def complete(self, prompt):
        FakeLLM.calls.append((self.model, prompt))
        outcome = FakeLLM.behaviour.get(self.model, FakeLLM.default_reply)
        if isinstance(outcome, Exception):
            raise outcome
        return types.SimpleNamespace(text=f"  {outcome}  ")


def _install_fake_llm_module():
    pkg_names = ["llama_index", "llama_index.llms", "llama_index.llms.google_genai"]
    for name in pkg_names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["llama_index.llms.google_genai"].GoogleGenAI = FakeLLM


_install_fake_llm_module()

import fastapi_rag  # noqa: E402  (must come after the fake module is installed)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    FakeLLM.calls = []
    FakeLLM.behaviour = {}
    FakeLLM.default_reply = "Stub answer."
    fastapi_rag._response_cache.clear()
    fastapi_rag._reference_cache = {"data": None, "expiry": 0}
    fastapi_rag._engine_instance = None
    monkeypatch.setattr(fastapi_rag, "GoogleGenAI", FakeLLM)
    monkeypatch.setattr(fastapi_rag.time, "sleep", lambda _s: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    yield


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with eng.begin() as c:
        c.execute(text("CREATE TABLE trainers (id INTEGER PRIMARY KEY, name TEXT)"))
        c.execute(text("CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT)"))
        c.execute(text("CREATE TABLE batches (id INTEGER PRIMARY KEY, batch_code TEXT)"))
        c.execute(text("CREATE TABLE upload_sessions (id INTEGER PRIMARY KEY, filename TEXT)"))
        c.execute(text(
            "CREATE TABLE feedback_records (id INTEGER PRIMARY KEY, created_at TEXT, "
            "student_name TEXT, feedback_text TEXT, rating REAL, department TEXT, "
            "trainer_id INTEGER, upload_session_id INTEGER)"
        ))
        c.execute(text("INSERT INTO trainers VALUES (1,'Priya Sharma'),(2,'Dr. Arjun Rao'),(3,'Arjun D')"))
        c.execute(text("INSERT INTO courses VALUES (1,'Intro to Python')"))
        c.execute(text("INSERT INTO batches VALUES (1,'B-2026-01')"))
        c.execute(text("INSERT INTO upload_sessions VALUES (1,'jan.xlsx')"))
        c.execute(text(
            "INSERT INTO feedback_records VALUES "
            "(1, date('now','-5 day'), 'S1', 'Priya explains clearly', 4.5, 'CS', 1, 1),"
            "(2, date('now','-4 day'), 'S2', 'Rao moves too fast', 3.0, 'IT', 2, 1),"
            "(3, date('now','-3 day'), 'S3', 'Arjun D is engaging', 4.0, 'CS', 3, 1)"
        ))
    fastapi_rag._engine_instance = eng
    return eng


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(fastapi_rag.app)
