import os
import sys
import time
import datetime
import re
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from llama_index.llms.google_genai import GoogleGenAI

# Load environment variables
load_dotenv()

app = FastAPI(title="Profice SQL RAG API", description="AI In-Context RAG feedback querying service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SINGLETON DB ENGINE  (pool reuse across requests)
# ─────────────────────────────────────────────────────────────
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found, falling back to SQLite profice.db")
        _engine_instance = create_engine("sqlite:///profice.db")
        return _engine_instance

    if db_url.startswith("mysql://"):
        db_url = db_url.replace("mysql://", "mysql+pymysql://")

    _engine_instance = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True
    )
    return _engine_instance


# ─────────────────────────────────────────────────────────────
# GLOBAL CACHES (Thread-safe)
# ─────────────────────────────────────────────────────────────
_response_cache = {}
_cache_lock = threading.Lock()

def get_cached_response(cache_key: str):
    with _cache_lock:
        if cache_key in _response_cache:
            data, expiry = _response_cache[cache_key]
            if time.time() < expiry:
                return data
            else:
                del _response_cache[cache_key]
    return None

def set_cached_response(cache_key: str, data, ttl_seconds: int = 180):
    with _cache_lock:
        _response_cache[cache_key] = (data, time.time() + ttl_seconds)


def get_database_context(engine, question: str) -> str:
    try:
        with engine.connect() as conn:
            # 1. Fetch reference lists
            trainers_all = conn.execute(text("SELECT id, name FROM trainers LIMIT 100")).fetchall()
            courses_all = conn.execute(text("SELECT id, title FROM courses LIMIT 100")).fetchall()
            batches_all = conn.execute(text("SELECT id, batch_code FROM batches LIMIT 100")).fetchall()
            
        # 2. Check for trainer names or departments in the question
        matched_trainers = []
        for t in trainers_all:
            if t[1].lower() in question.lower():
                matched_trainers.append(t[0])
                
        # Scan for common department keywords
        departments = ["CS", "Computer Science", "IT", "Information Technology", "Cloud", "Data Science", "DS", "MERN", "Full Stack", "UIUX", "UX"]
        matched_depts = []
        for d in departments:
            if d.lower() in question.lower():
                matched_depts.append(d)
                
        # 3. Build dynamic SQL query
        where_clauses = []
        cutoff_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        where_clauses.append(f"f.created_at >= '{cutoff_date}'")
        
        if matched_trainers:
            trainer_ids_str = ",".join([str(tid) for tid in matched_trainers])
            where_clauses.append(f"f.trainer_id IN ({trainer_ids_str})")
            
        if matched_depts:
            dept_filters = " OR ".join([f"f.department LIKE '%{d}%'" for d in matched_depts])
            where_clauses.append(f"({dept_filters})")
            
        where_str = " AND ".join(where_clauses)
        
        # If no specific entities matched, limit to 30 most recent feedbacks to keep prompt fast.
        # If specific entities matched, fetch up to 150 rows of data.
        limit_val = 150 if (matched_trainers or matched_depts) else 30
        
        query = (
            "SELECT f.created_at, f.student_name, f.feedback_text, f.rating, f.department, "
            "       t.name AS trainer_name, u.filename AS source_file "
            "FROM feedback_records f "
            "LEFT JOIN trainers t ON f.trainer_id = t.id "
            "LEFT JOIN upload_sessions u ON f.upload_session_id = u.id "
            f"WHERE {where_str} "
            f"ORDER BY f.created_at DESC LIMIT {limit_val}"
        )
        
        with engine.connect() as conn:
            feedback_all = conn.execute(text(query)).fetchall()
            
        # Format lists into structured context strings
        trainers_str = "\n".join([f"- ID: {t[0]} | Name: {t[1]}" for t in trainers_all])
        courses_str = "\n".join([f"- ID: {c[0]} | Title: {c[1]}" for c in courses_all])
        batches_str = "\n".join([f"- ID: {b[0]} | Code: {b[1]}" for b in batches_all])
        
        feedback_list = []
        for f in feedback_all:
            created_at = f[0].strftime('%Y-%m-%d') if isinstance(f[0], (datetime.datetime, datetime.date)) else str(f[0])
            feedback_list.append(
                f"Date: {created_at} | Trainer: {f[5]} | Dept: {f[4]} | Rating: {f[3]} | "
                f"Student: {f[1]} | File: {f[6]} | Comment: {f[2]}"
            )
        feedback_str = "\n".join(feedback_list)
        
        # Combine into unified prompt representation
        context = (
            f"=== SYSTEM DATABASE SNAPSHOT (LAST 90 DAYS) ===\n\n"
            f"TRAINERS IN SYSTEM:\n{trainers_str if trainers_str else 'None'}\n\n"
            f"COURSES IN SYSTEM:\n{courses_str if courses_str else 'None'}\n\n"
            f"BATCHES IN SYSTEM:\n{batches_str if batches_str else 'None'}\n\n"
            f"STUDENT FEEDBACK RECORDS (dynamic context slice):\n"
            f"{feedback_str if feedback_str else 'No matching feedback records in this timeframe.'}\n"
            f"==============================================="
        )
        return context
        
    except Exception as e:
        print("Error loading database context:", str(e))
        raise e



# ─────────────────────────────────────────────────────────────
# PRE-COMPILED PATTERNS & MODEL CONFIG
# ─────────────────────────────────────────────────────────────
_GREETING_PATTERN = re.compile(
    r'^\s*(hello|hi|hey|greetings|good morning|good afternoon|good evening|who are you|what can you do|help me|hola|hi there|hello there)\b',
    re.IGNORECASE
)
_PRONOUN_PATTERN = re.compile(
    r'\b(his|her|he|him|she|this trainer|the trainer|their|they|this person|that person|person)\b',
    re.IGNORECASE
)

_FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro"
]


# ─────────────────────────────────────────────────────────────
# RETRY & MODEL FALLBACK RUNNER
# ─────────────────────────────────────────────────────────────
def safe_llm_complete_fallback(prompt: str, api_key: str, retries: int = 2, delay: int = 1) -> str:
    last_error = None
    for model_name in _FALLBACK_MODELS:
        llm = GoogleGenAI(model=model_name, api_key=api_key)
        for i in range(retries):
            try:
                res = llm.complete(prompt)
                return res.text.strip()
            except Exception as e:
                err_msg = str(e)
                is_quota_error = "ResourceExhausted" in err_msg or "429" in err_msg or "quota" in err_msg.lower()
                
                if is_quota_error:
                    if i < retries - 1:
                        print(f"Rate limit hit for {model_name}. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Model {model_name} exhausted. Trying fallback model in chain...")
                        last_error = e
                        break
                raise e  # Propagate non-quota errors immediately
                
    if last_error:
        raise last_error
    raise HTTPException(status_code=500, detail="LLM execution failed across all configured models.")


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list = []
    role: str = "admin"
    trainer_id: str = None


class ChatResponse(BaseModel):
    answer: str
    sql_query: str = None


# ─────────────────────────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        # ══════════════════════════════════════════════════════
        # STEP 0: INSTANT FAST-PATHS  (<1ms — zero DB, zero LLM)
        # ══════════════════════════════════════════════════════

        # 0a. Greeting fast-path
        if _GREETING_PATTERN.search(req.question.strip()):
            return ChatResponse(
                answer="Hello! I am your AI Database Assistant. Ask me any question about "
                       "trainers, ratings, courses, feedback records, or departments, and I "
                       "will query the database for you."
            )

        # 0b. Pronoun resolution (history scan only)
        has_pronoun = bool(_PRONOUN_PATTERN.search(req.question))
        last_mentioned_trainer = None
        if has_pronoun and req.history:
            for h_item in reversed(req.history):
                text_val = h_item.get("content", "")
                name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+)\b', text_val)
                if name_match:
                    last_mentioned_trainer = name_match.group(1)
                    break

        processed_query = req.question
        if has_pronoun and last_mentioned_trainer and last_mentioned_trainer.lower() not in processed_query.lower():
            processed_query = f"For trainer {last_mentioned_trainer}: " + processed_query

        # 0c. Response cache fast-path (TTL 3 min)
        cache_key = f"{req.role}_{req.trainer_id}_{processed_query.strip().lower()}"
        cached_res = get_cached_response(cache_key)
        if cached_res:
            return cached_res

        # ══════════════════════════════════════════════════════
        # STEP 1: LOAD DB CONTEXT AND API CONFIGURATION
        # ══════════════════════════════════════════════════════
        engine = get_engine()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API Key is not set in environment.")

        # Load database snapshot into memory dynamically filtered
        db_context = get_database_context(engine, processed_query)


        # ══════════════════════════════════════════════════════
        # STEP 2: CONSTRUCT CHAT PROMPT WITH RAW DATA CONTEXT
        # ══════════════════════════════════════════════════════
        history_str = ""
        if req.history:
            for item in req.history:
                role_name = "User" if item.get("role") == "user" else "Assistant"
                history_str += f"{role_name}: {item.get('content')}\n"

        system_instructions = (
            "You are the SQL Database Assistant for the Profice Feedback Portal.\n"
            "You have direct access to a snapshot of the database content below (including feedback, trainers, courses).\n\n"
            "Guidelines:\n"
            "- Answer the user's question directly, clearly, and concisely in 1-2 simple English sentences.\n"
            "- Use the database snapshot context to verify trainer performance, ratings, departments, batch codes, or file names.\n"
            "- Format all average values to strictly 1 digit after the decimal point (e.g. 4.3 or 3.8).\n"
            "- Do not invent facts or mention database structures, internal IDs, query syntax, or columns.\n"
            "- Keep your response conversational and factually precise according to the snapshot data."
        )

        full_prompt = (
            f"{system_instructions}\n\n"
            f"{db_context}\n\n"
            f"Conversation History:\n"
            f"{history_str if history_str else 'None'}\n\n"
            f"User Question: {processed_query}\n"
            f"Assistant:"
        )

        # ══════════════════════════════════════════════════════
        # STEP 3: EXECUTE SINGLE ROUND LLM COMPLETION
        # ══════════════════════════════════════════════════════
        response_text = safe_llm_complete_fallback(full_prompt, api_key)
        
        res_obj = ChatResponse(answer=response_text)
        set_cached_response(cache_key, res_obj)
        return res_obj

    except Exception as e:
        print("SQL RAG Execution Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Database Chatbot Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("RAG_PORT", 8000))
    uvicorn.run("fastapi_rag:app", host="0.0.0.0", port=port, reload=True)
