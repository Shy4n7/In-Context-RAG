import pytest

import fastapi_rag
from conftest import FakeLLM


# ---------- response cache ----------

def test_cache_returns_value_before_expiry_and_drops_after(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(fastapi_rag.time, "time", lambda: now[0])
    fastapi_rag.set_cached_response("k", "v", ttl_seconds=10)
    assert fastapi_rag.get_cached_response("k") == "v"
    now[0] = 1011.0
    assert fastapi_rag.get_cached_response("k") is None
    assert "k" not in fastapi_rag._response_cache


# ---------- reference data cache ----------

def test_reference_cache_hits_db_once_within_ttl(engine):
    first = fastapi_rag.get_reference_data(engine)
    with engine.begin() as c:
        c.exec_driver_sql("INSERT INTO trainers VALUES (99, 'New Person')")
    second = fastapi_rag.get_reference_data(engine)
    assert second is first
    assert second["total_trainers"] == 3


def test_reference_cache_serves_stale_data_when_db_fails(engine, monkeypatch):
    fastapi_rag.get_reference_data(engine, ttl_seconds=0)

    class Broken:
        def connect(self):
            raise RuntimeError("db down")

    data = fastapi_rag.get_reference_data(Broken(), ttl_seconds=0)
    assert data["total_trainers"] == 3


# ---------- context building ----------

def test_metadata_question_skips_feedback_rows(engine):
    ctx = fastapi_rag.get_database_context(engine, "how many trainers are there")
    assert "No student feedback records were loaded" in ctx
    assert "Total Trainers in Database: 3" in ctx


def test_exact_trainer_name_filters_feedback(engine):
    ctx = fastapi_rag.get_database_context(engine, "feedback for Priya Sharma")
    assert "Priya explains clearly" in ctx
    assert "Rao moves too fast" not in ctx


def test_department_keyword_filters_feedback(engine):
    ctx = fastapi_rag.get_database_context(engine, "how is the IT department doing")
    assert "Rao moves too fast" in ctx
    assert "Priya explains clearly" not in ctx


def test_ambiguous_first_name_adds_system_alert(engine):
    ctx = fastapi_rag.get_database_context(engine, "how is Arjun doing")
    assert "[SYSTEM ALERT" in ctx
    assert "Dr. Arjun Rao" in ctx and "Arjun D" in ctx


def test_department_abbreviation_needs_word_boundary(engine):
    ctx = fastapi_rag.get_database_context(engine, "do students criticize anything")
    # "IT"/"CS" must not match inside "criticize"; falls back to the general slice
    assert "Priya explains clearly" in ctx and "Rao moves too fast" in ctx


# ---------- model fallback ----------

def test_fallback_moves_to_next_model_on_quota_error():
    first, second = fastapi_rag._FALLBACK_MODELS[:2]
    FakeLLM.behaviour = {first: Exception("429 RESOURCE_EXHAUSTED"), second: "from backup"}
    out = fastapi_rag.safe_llm_complete_fallback("hi", "key")
    assert out == "from backup"
    tried = [m for m, _ in FakeLLM.calls]
    assert tried[:2] == [first, first] and tried[2] == second  # 2 retries, then next model


def test_non_quota_error_propagates_immediately():
    first = fastapi_rag._FALLBACK_MODELS[0]
    FakeLLM.behaviour = {first: ValueError("bad request")}
    with pytest.raises(ValueError):
        fastapi_rag.safe_llm_complete_fallback("hi", "key")
    assert len(FakeLLM.calls) == 1


def test_all_models_exhausted_raises_last_error():
    FakeLLM.behaviour = {m: Exception("quota exceeded") for m in fastapi_rag._FALLBACK_MODELS}
    with pytest.raises(Exception, match="quota"):
        fastapi_rag.safe_llm_complete_fallback("hi", "key")


# ---------- HTTP endpoint ----------

def test_greeting_never_calls_llm(client, engine):
    r = client.post("/api/chat", json={"question": "hello"})
    assert r.status_code == 200
    assert "AI Database Assistant" in r.json()["answer"]
    assert FakeLLM.calls == []


def test_repeat_question_is_served_from_cache(client, engine):
    body = {"question": "feedback for Priya Sharma"}
    a = client.post("/api/chat", json=body).json()["answer"]
    b = client.post("/api/chat", json=body).json()["answer"]
    assert a == b == "Stub answer."
    assert len(FakeLLM.calls) == 1


def test_pronoun_is_resolved_from_history(client, engine):
    history = [{"role": "assistant", "content": "Priya Sharma has a 4.5 rating."}]
    client.post("/api/chat", json={"question": "what do students say about her", "history": history})
    prompt = FakeLLM.calls[0][1]
    assert "For trainer Priya Sharma:" in prompt


def test_missing_api_key_returns_500(client, engine, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = client.post("/api/chat", json={"question": "feedback for Priya Sharma"})
    assert r.status_code == 500
