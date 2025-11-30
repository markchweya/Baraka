# app.py — Baraka (Modern UI) + Complaint Management + Department Routing + Multilingual Chatbot
#
# Demo accounts:
#   admin / admin123
#   user  / user123
#
# requirements:
# streamlit
# openai
# pandas
# scikit-learn
# pyarrow
# huggingface-hub
# fsspec

import os
import re
import json
import sqlite3
import base64
import hashlib
from typing import Tuple, Optional

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# CONFIG
# ----------------------------
APP_NAME = "Baraka"
DB_PATH = "bankbot.db"

BASE_PARQUET_URL = (
    "hf://datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/"
    "bitext-retail-banking-llm-chatbot-training-dataset.parquet"
)

TOPK = 3
SIM_THRESHOLD_CUSTOM = 0.40
SIM_THRESHOLD_BASE = 0.35
SIM_THRESHOLD_ROUTE = 0.25

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSLATION_MODEL = "gpt-4.1-mini"
FALLBACK_MODEL = "gpt-4.1-mini"


# ----------------------------
# LANGUAGES (expanded)
# ----------------------------
LANGS = {
    "en": "English",
    "sw": "Kiswahili",
    "am": "Amharic",
    "so": "Somali",
    "ar": "Arabic",
    "sn": "Shona",
    "ki": "Gikuyu (Kikuyu)",
    "luo": "Dholuo",
    "kam": "Kikamba",
    "guz": "Ekegusii (Kisii/Gusii)",
    "mer": "Kimeru (Meru)",
}

SUPPORTED_LANG_CODES = set(LANGS.keys())


# ----------------------------
# DEPARTMENTS
# ----------------------------
DEPARTMENTS = [
    "ACCOUNT", "ATM", "CARD", "CONTACT", "FEES",
    "FIND", "LOAN", "PASSWORD", "TRANSFER"
]

DEPT_LABELS = {
    "ACCOUNT": "Accounts & Onboarding",
    "ATM": "ATM / Channel Support",
    "CARD": "Cards & Wallets",
    "CONTACT": "Customer Care",
    "FEES": "Charges & Pricing",
    "FIND": "ATM / Branch Locator",
    "LOAN": "Loans & Mortgages",
    "PASSWORD": "Security & Passwords",
    "TRANSFER": "Payments & Transfers"
}

# Router training samples (English)
DEPT_TRAIN = {
    "ACCOUNT": [
        "open account", "create account", "close account", "account frozen",
        "recent transactions", "bank statement", "account verification", "kyc update",
        "check balance", "account balance", "account statement"
    ],
    "ATM": [
        "atm swallowed my card", "no cash but debited", "failed withdrawal",
        "atm reversal", "withdrawal dispute"
    ],
    "CARD": [
        "activate card", "block card", "cancel card", "card not working",
        "international usage", "annual fee", "card balance"
    ],
    "CONTACT": [
        "customer care", "speak to agent", "human agent", "call center", "contact support"
    ],
    "FEES": [
        "charges too high", "check fees", "fee dispute", "pricing", "charges"
    ],
    "FIND": [
        "find atm", "nearest atm", "find branch", "branch near me"
    ],
    "LOAN": [
        "apply for loan", "loan repayment", "loan status", "interest rate",
        "borrow money", "take a loan", "get a loan"
    ],
    "PASSWORD": [
        "reset password", "forgot password", "set up password", "login problem",
        "reset pin", "forgot pin"
    ],
    "TRANSFER": [
        "cancel transfer", "make transfer", "wrong transfer", "pending transfer",
        "reverse transaction", "send money"
    ]
}

DEPT_KEYWORDS = {
    "ATM": ["atm", "withdrawal", "no cash", "swallowed", "debited"],
    "CARD": ["card", "visa", "mastercard", "debit card", "credit card"],
    "LOAN": ["loan", "borrow", "interest", "repayment", "mortgage"],
    "TRANSFER": ["transfer", "send money", "reversal", "pending", "reverse transaction"],
    "PASSWORD": ["password", "pin", "reset", "forgot", "login problem"],
    "FEES": ["fees", "charges", "pricing", "annual fee"],
    "FIND": ["find atm", "nearest atm", "branch", "locator"],
    "CONTACT": ["agent", "customer care", "call center", "contact support"],
    "ACCOUNT": ["account", "statement", "transactions", "balance", "kyc"]
}


# ----------------------------
# MODERN LIGHT UI (No big “box containers”)
# ----------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root{
  --bg1:#f7fbff;
  --bg2:#fbf7ff;
  --ink:#0a1020;
  --muted:#56607a;
  --line:rgba(15,23,42,0.10);

  --p1:#5b7cfa;
  --p2:#7c3aed;
  --p3:#22c55e;
  --p4:#06b6d4;
  --warn:#f59e0b;
  --danger:#ef4444;

  --r:18px;
  --shadow: 0 18px 50px rgba(15,23,42,0.12);
  --shadow2: 0 10px 25px rgba(15,23,42,0.08);
}

#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display:none !important;}
[data-testid="stDecoration"] {display:none !important;}

html, body, [data-testid="stAppViewContainer"]{
  font-family: "Plus Jakarta Sans", system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  color: var(--ink) !important;
  background:
    radial-gradient(1000px 700px at 12% 5%, rgba(91,124,250,0.20), transparent 60%),
    radial-gradient(900px 650px at 92% 12%, rgba(124,58,237,0.16), transparent 55%),
    radial-gradient(900px 650px at 50% 105%, rgba(34,197,94,0.12), transparent 52%),
    linear-gradient(180deg, var(--bg1), var(--bg2)) !important;
}

.block-container{
  max-width: 1200px;
  padding-top: 1.2rem !important;
  padding-bottom: 1.6rem !important;
}

h1,h2,h3{
  letter-spacing: -0.03em;
}
h1{ font-weight: 800; }
h2{ font-weight: 800; }
h3{ font-weight: 800; }
p{ color: var(--muted); }

.hr{
  height:1px; background: var(--line); border:none; margin: 14px 0;
}

.pillbar{
  display:flex; gap:10px; flex-wrap:wrap;
  padding:8px 0 6px 0;
}

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 12px; border-radius:999px;
  border:1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.55);
  box-shadow: var(--shadow2);
  font-weight: 650;
  color: rgba(10,16,32,0.88);
  transition: transform 140ms ease, box-shadow 140ms ease;
}
.pill:hover{ transform: translateY(-1px); box-shadow: 0 16px 35px rgba(15,23,42,0.10); }

.primary-btn button{
  border: 1px solid rgba(91,124,250,0.25) !important;
  background: linear-gradient(135deg, var(--p1), var(--p2)) !important;
  color: #fff !important;
  font-weight: 800 !important;
  border-radius: 14px !important;
  padding: 0.72rem 1.05rem !important;
  box-shadow: 0 16px 30px rgba(91,124,250,0.22) !important;
  transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease !important;
}
.primary-btn button:hover{
  transform: translateY(-1px);
  box-shadow: 0 20px 40px rgba(91,124,250,0.26) !important;
  filter: saturate(1.05);
}

.stTextInput input, .stTextArea textarea, .stSelectbox select{
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.14) !important;
  background: rgba(255,255,255,0.74) !important;
  box-shadow: var(--shadow2);
}

label{ font-weight: 750 !important; }

.hero{
  display:flex; justify-content:space-between; align-items:flex-end; gap:18px;
  padding: 8px 0 2px 0;
}
.hero .subtitle{
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.5;
}

.login-wrap{
  margin-top: 16px;
  display:grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 22px;
}
@media (max-width: 900px){
  .login-wrap{ grid-template-columns: 1fr; }
}

.soft-panel{
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: var(--r);
  padding: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
}

.mini{
  font-size: 0.95rem;
  color: var(--muted);
}

kbd{
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.65);
  box-shadow: var(--shadow2);
  font-weight: 700;
  color: rgba(10,16,32,0.92);
}

[data-testid="stChatMessage"]{
  border-radius: 16px;
  border: 1px solid rgba(15,23,42,0.08);
  background: rgba(255,255,255,0.68);
  box-shadow: var(--shadow2);
  padding: 12px 12px;
}

[data-testid="stChatMessage"] *{
  color: rgba(10,16,32,0.94) !important;
}

.chat-drawer-hint{
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: -6px;
}

.stTabs [data-baseweb="tab"]{
  border-radius: 999px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(15,23,42,0.10) !important;
  background: rgba(255,255,255,0.55) !important;
  font-weight: 800 !important;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(91,124,250,0.16), rgba(124,58,237,0.12)) !important;
  border-color: rgba(91,124,250,0.25) !important;
}

[data-testid="stDataFrame"]{
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: var(--shadow);
}
</style>
"""


# ----------------------------
# OPENAI CLIENT
# ----------------------------
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


# ----------------------------
# PLACEHOLDER PROTECTION
# ----------------------------
_PLACEHOLDER_RE = re.compile(r"(\{\{[^{}]*\}\}|\{[^{}]*\}|<[^<>]*>)")

def protect_placeholders(text: str):
    mapping = {}
    def repl(m):
        key = f"@@PH{len(mapping)}@@"
        mapping[key] = m.group(0)
        return key
    return _PLACEHOLDER_RE.sub(repl, text), mapping

def restore_placeholders(text: str, mapping: dict):
    out = text
    for k in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(k, mapping[k])
    return out


def _safe_json_parse(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # try to extract first json object
        try:
            i = s.find("{")
            j = s.rfind("}")
            if i >= 0 and j > i:
                return json.loads(s[i:j+1])
        except Exception:
            return None
    return None


# ----------------------------
# MULTILINGUAL: detect + translate to English + translate back
# ----------------------------
@st.cache_data(show_spinner=False)
def detect_and_translate_to_english(user_text: str) -> Tuple[str, str]:
    """
    Returns (lang_code, english_text).
    If key missing: returns ('en', original_text).
    """
    client = get_openai_client()
    if not client:
        return "en", user_text

    protected, mapping = protect_placeholders(user_text)

    system = (
        "You are a language detector + translator.\n"
        "Return ONLY valid JSON: {\"lang\": \"...\", \"english\": \"...\"}\n"
        "lang MUST be one of: " + ", ".join(sorted(SUPPORTED_LANG_CODES)) + "\n"
        "Translate the INPUT into English.\n"
        "Preserve placeholders exactly (tokens like @@PH0@@)."
    )

    try:
        resp = client.responses.create(
            model=TRANSLATION_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"INPUT:\n{protected}"}
            ],
            temperature=0
        )
        raw = (resp.output_text or "").strip()
        data = _safe_json_parse(raw) or {}
        lang = str(data.get("lang", "en")).strip().lower()
        eng = str(data.get("english", protected))

        if lang not in SUPPORTED_LANG_CODES:
            lang = "en"

        eng = restore_placeholders(eng, mapping)
        return lang, eng

    except Exception:
        return "en", user_text


@st.cache_data(show_spinner=False)
def translate_from_english(text_en: str, target_lang: str) -> str:
    if target_lang == "en":
        return text_en

    client = get_openai_client()
    if not client:
        return text_en

    protected, mapping = protect_placeholders(text_en)

    system = (
        "You are a professional translator.\n"
        f"Translate from English to {LANGS.get(target_lang, target_lang)}.\n"
        "Rules:\n"
        "- Output ONLY the translation.\n"
        "- Preserve placeholders exactly (tokens like @@PH0@@).\n"
        "- Keep numbers/currency/product names unchanged.\n"
        "- Natural, fluent, not robotic."
    )

    try:
        resp = client.responses.create(
            model=TRANSLATION_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": protected}
            ],
            temperature=0
        )
        out = (resp.output_text or "").strip()
        out = restore_placeholders(out, mapping)
        return out
    except Exception:
        return text_en


def handle_language_command(user_text: str):
    """
    Sets preferred language if user says stuff like:
      - jibu kwa kiswahili
      - reply in amharic
      - somali / arabic / shona / kikuyu / dholuo / kamba / kisii / meru
    """
    t = (user_text or "").strip().lower()
    if not t:
        return False, None, None

    name_to_code = {
        "english": "en",
        "kiswahili": "sw", "swahili": "sw",
        "amharic": "am", "አማርኛ": "am",
        "somali": "so", "soomaali": "so",
        "arabic": "ar", "العربية": "ar", "عربي": "ar",
        "shona": "sn",
        "kikuyu": "ki", "gikuyu": "ki",
        "dholuo": "luo", "luo": "luo",
        "kamba": "kam", "kikamba": "kam",
        "kisii": "guz", "ekegusii": "guz", "gusii": "guz",
        "meru": "mer", "kimeru": "mer",
    }

    for k, code in name_to_code.items():
        if k in t:
            msg_en = f"Okay — I’ll reply in {LANGS.get(code, code)} from now on."
            msg = translate_from_english(msg_en, code)
            return True, msg, code

    return False, None, None


# ----------------------------
# PASSWORD HASHING (PBKDF2)
# ----------------------------
def hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.b64encode(salt + key).decode()

def is_pbkdf2_hash(stored) -> bool:
    if stored is None:
        return False
    if isinstance(stored, bytes):
        try:
            stored = stored.decode()
        except Exception:
            return False
    if not isinstance(stored, str):
        return False
    try:
        raw = base64.b64decode(stored.encode())
        return len(raw) >= 48
    except Exception:
        return False

def verify_password(password: str, stored) -> bool:
    if not is_pbkdf2_hash(stored):
        return False
    if isinstance(stored, bytes):
        stored = stored.decode()
    raw = base64.b64decode(stored.encode())
    salt, key = raw[:16], raw[16:]
    new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return new_key == key


# ----------------------------
# DB HELPERS
# ----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def column_exists(conn, table, col):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols

def seed_or_upgrade_user(c, username, password, role):
    c.execute("SELECT pw_hash FROM users WHERE username=?", (username,))
    row = c.fetchone()
    new_hash = hash_password(password)
    if not row:
        c.execute(
            "INSERT INTO users(username,pw_hash,role) VALUES(?,?,?)",
            (username, new_hash, role)
        )
    else:
        stored_hash = row[0]
        if not is_pbkdf2_hash(stored_hash):
            c.execute(
                "UPDATE users SET pw_hash=?, role=? WHERE username=?",
                (new_hash, role, username)
            )

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        pw_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','admin'))
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS custom_faqs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        department TEXT DEFAULT 'CONTACT',
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        tags TEXT,
        created_by TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS complaints(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        text TEXT NOT NULL,
        department TEXT NOT NULL,
        status TEXT DEFAULT 'Open',
        priority TEXT DEFAULT 'Normal',
        summary TEXT,
        internal_notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        user_message TEXT,
        bot_reply TEXT,
        source TEXT,
        score REAL,
        department TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    if not column_exists(conn, "chat_logs", "department"):
        c.execute("ALTER TABLE chat_logs ADD COLUMN department TEXT")

    seed_or_upgrade_user(c, "admin", "admin123", "admin")
    seed_or_upgrade_user(c, "user", "user123", "user")

    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT pw_hash, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    pw_hash, role = row
    if verify_password(password, pw_hash):
        return role
    return None

def fetch_custom_faqs(department=None):
    conn = get_conn()
    if department and department != "ALL":
        df = pd.read_sql_query(
            "SELECT * FROM custom_faqs WHERE department=? ORDER BY updated_at DESC",
            conn, params=(department,)
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM custom_faqs ORDER BY updated_at DESC", conn
        )
    conn.close()
    return df

def add_custom_faq(dept, q, a, tags, created_by):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO custom_faqs(department,question,answer,tags,created_by)
    VALUES(?,?,?,?,?)
    """, (dept, q, a, tags, created_by))
    conn.commit()
    conn.close()

def update_custom_faq(fid, dept, q, a, tags):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE custom_faqs
    SET department=?, question=?, answer=?, tags=?, updated_at=CURRENT_TIMESTAMP
    WHERE id=?
    """, (dept, q, a, tags, fid))
    conn.commit()
    conn.close()

def delete_custom_faq(fid):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM custom_faqs WHERE id=?", (fid,))
    conn.commit()
    conn.close()

def log_chat(username, user_message, bot_reply, source, score, department):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO chat_logs(username,user_message,bot_reply,source,score,department)
    VALUES(?,?,?,?,?,?)
    """, (username, user_message, bot_reply, source, score, department))
    conn.commit()
    conn.close()

def create_complaint(username, text, department, priority="Normal", summary=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO complaints(username,text,department,priority,summary)
    VALUES(?,?,?,?,?)
    """, (username, text, department, priority, summary))
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return cid

def fetch_complaints(dept="ALL", status="ALL"):
    conn = get_conn()
    q = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if dept != "ALL":
        q += " AND department=?"
        params.append(dept)
    if status != "ALL":
        q += " AND status=?"
        params.append(status)
    q += " ORDER BY created_at DESC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df

def update_complaint(cid, status=None, priority=None, internal_notes=None):
    conn = get_conn()
    c = conn.cursor()
    fields = []
    params = []
    if status:
        fields.append("status=?"); params.append(status)
    if priority:
        fields.append("priority=?"); params.append(priority)
    if internal_notes is not None:
        fields.append("internal_notes=?"); params.append(internal_notes)
    fields.append("updated_at=CURRENT_TIMESTAMP")
    q = "UPDATE complaints SET " + ", ".join(fields) + " WHERE id=?"
    params.append(cid)
    c.execute(q, params)
    conn.commit()
    conn.close()


# ----------------------------
# HF DATASET LOADING
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_base_dataset():
    df = pd.read_parquet(BASE_PARQUET_URL)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    qcol = pick("instruction", "question", "user", "utterance", "query", "input")
    acol = pick("response", "answer", "assistant", "output")
    catcol = pick("category", "dept", "department")
    intentcol = pick("intent")

    if not qcol or not acol:
        raise ValueError(f"Could not detect question/answer columns. Found: {df.columns}")

    base_df = df[[qcol, acol]].rename(columns={qcol: "question", acol: "answer"})
    base_df["question"] = base_df["question"].astype(str)
    base_df["answer"] = base_df["answer"].astype(str)

    if catcol:
        base_df["category"] = df[catcol].astype(str).str.upper()
    else:
        base_df["category"] = "CONTACT"

    base_df["intent"] = df[intentcol].astype(str) if intentcol else ""
    base_df.dropna(inplace=True)
    base_df.reset_index(drop=True, inplace=True)
    return base_df


@st.cache_resource(show_spinner=False)
def build_vector_index(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 3:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words=None)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X


# ----------------------------
# ROUTING + RETRIEVAL
# ----------------------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

@st.cache_resource(show_spinner=False)
def build_dept_router():
    dept_texts, dept_labels = [], []
    for d, samples in DEPT_TRAIN.items():
        for s in samples:
            dept_texts.append(s)
            dept_labels.append(d)
    vec, X = build_vector_index(dept_texts)
    return vec, X, dept_labels

def route_department(text_en):
    t = normalize(text_en)

    for dept, kws in DEPT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return dept, 1.0, "rule"

    vec, X, labels = build_dept_router()
    tv = vec.transform([t])
    sims = cosine_similarity(tv, X).flatten()
    best_idx = int(sims.argmax())
    best_dept = labels[best_idx]
    best_score = float(sims[best_idx])

    if best_score < SIM_THRESHOLD_ROUTE:
        return "CONTACT", best_score, "tfidf_lowconf"
    return best_dept, best_score, "tfidf"

def retrieve_best(query_en, faq_df, vec, X, topk=TOPK):
    qn = normalize(query_en)
    qv = vec.transform([qn])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:topk]
    results = faq_df.iloc[idxs].copy()
    results["score"] = sims[idxs]
    return results

def answer_from_custom_first(query_en, dept):
    custom_df = fetch_custom_faqs(dept)
    if custom_df.empty:
        return None

    questions = [q for q in custom_df["question"].astype(str).tolist() if q.strip()]
    if not questions:
        return None

    vec_c, X_c = build_vector_index(questions)
    res = retrieve_best(query_en, custom_df, vec_c, X_c, topk=TOPK)
    best = res.iloc[0]
    if float(best["score"]) >= SIM_THRESHOLD_CUSTOM:
        return best["answer"], float(best["score"]), "custom"
    return None

def answer_from_base(query_en, dept, base_df):
    base_dept = base_df[base_df["category"] == dept]
    if base_dept.empty:
        base_dept = base_df

    vec_d, X_d = build_vector_index(base_dept["question"].tolist())
    res = retrieve_best(query_en, base_dept, vec_d, X_d, topk=TOPK)
    best = res.iloc[0]
    if float(best["score"]) >= SIM_THRESHOLD_BASE:
        return best["answer"], float(best["score"]), "base"
    return None


def openai_fallback(query_en, context_snippets, out_lang="en"):
    client = get_openai_client()
    if not client:
        return ("I’m not fully confident yet. Please rephrase or add more detail.", 0.0, "fallback")

    try:
        system = (
            "Your name is Baraka. "
            "You are a helpful Kenyan retail-banking & SACCO support assistant. "
            "Answer ONLY using the provided context. "
            "If context is insufficient, ask one short follow-up question. "
            "Never request PINs, passwords, or OTPs. "
            f"Reply in {LANGS.get(out_lang, 'English')}."
        )

        user = (
            f"Customer question (English): {query_en}\n\n"
            "Context (FAQ snippets):\n" + "\n---\n".join(context_snippets)
        )

        resp = client.responses.create(
            model=FALLBACK_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.2
        )
        return (resp.output_text or "").strip(), 0.0, "openai"
    except Exception:
        return ("AI fallback is unavailable right now.", 0.0, "fallback")


def generate_reply(user_message_raw: str, query_en: str, username: str, dept: str, out_lang: str):
    base_df = load_base_dataset()

    custom_hit = answer_from_custom_first(query_en, dept)
    if custom_hit:
        ans_en, score, source = custom_hit
        ans_out = translate_from_english(ans_en, out_lang)
        log_chat(username, user_message_raw, ans_out, source, score, dept)
        return ans_out, source, score

    base_hit = answer_from_base(query_en, dept, base_df)
    if base_hit:
        ans_en, score, source = base_hit
        ans_out = translate_from_english(ans_en, out_lang)
        log_chat(username, user_message_raw, ans_out, source, score, dept)
        return ans_out, source, score

    vec_b, X_b = build_vector_index(base_df["question"].tolist())
    top_base = retrieve_best(query_en, base_df, vec_b, X_b, topk=TOPK)
    snippets = [f"Q: {r.question}\nA: {r.answer}" for r in top_base.itertuples()]

    ans_out, score, source = openai_fallback(query_en, snippets, out_lang=out_lang)
    log_chat(username, user_message_raw, ans_out, source, score, dept)
    return ans_out, source, score


# ----------------------------
# UI HELPERS
# ----------------------------
def top_hero(title: str, subtitle: str):
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <h1 style="margin:0;">{title}</h1>
            <div class="subtitle">{subtitle}</div>
          </div>
          <div class="pillbar">
            <div class="pill">✨ Light UI</div>
            <div class="pill">🌍 Multilingual</div>
            <div class="pill">🧠 Smart routing</div>
          </div>
        </div>
        <div class="hr"></div>
        """,
        unsafe_allow_html=True
    )


def nav_bar(role: str):
    # Simple clean nav (no ugly boxes)
    pages_user = [("Home", "home"), ("Complaint", "complaint"), ("Chat", "chat")]
    pages_admin = [("Admin", "admin")]
    pages = pages_admin if role == "admin" else pages_user

    cols = st.columns(len(pages) + 1)
    for i, (label, key) in enumerate(pages):
        with cols[i]:
            if st.button(label, use_container_width=True):
                st.session_state.page = key
                st.rerun()
    with cols[-1]:
        if st.button("Logout", use_container_width=True):
            for k in ["user", "role", "page", "messages", "active_ticket", "preferred_lang"]:
                st.session_state.pop(k, None)
            st.rerun()

def require_api_key_banner():
    if not OPENAI_API_KEY:
        st.warning("Set OPENAI_API_KEY to enable multilingual translation + AI fallback.")


# ----------------------------
# PAGES
# ----------------------------
def login_page():
    top_hero(APP_NAME, "Sign in, submit complaints, and chat with Baraka — your multilingual support assistant.")
    require_api_key_banner()

    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1.0])

    with left:
        st.markdown(
            """
            <div class="soft-panel">
              <h3 style="margin:0;">Welcome back</h3>
              <div class="mini" style="margin-top:10px;">
                Demo logins:
                <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
                  <kbd>admin / admin123</kbd>
                  <kbd>user / user123</kbd>
                </div>
              </div>
              <div class="hr"></div>
              <div class="mini">
                Baraka auto-detects your language and replies in the same one.<br/>
                Supported: Swahili, Amharic, Somali, Arabic, Shona, Kikuyu, Dholuo, Kamba, Kisii, Meru.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("### Sign in")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g., admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Login")
            st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            role = verify_user(username, password)
            if role:
                st.session_state.user = username
                st.session_state.role = role
                st.session_state.page = "home" if role == "user" else "admin"
                st.rerun()
            else:
                st.error("Invalid username/password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def user_home_page():
    top_hero("Customer Portal", "Pick what you want to do. The chatbot can be opened from the chat drawer.")
    nav_bar("user")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("### 📝 Submit a complaint")
        st.write("Describe your issue, Baraka routes it automatically, and you get an instant reply.")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Open Complaint Form", use_container_width=True):
            st.session_state.page = "complaint"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("### 💬 Chat with Baraka")
        st.write("Open the chat drawer anytime. You can switch language by saying e.g. “reply in kiswahili”.")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Open Chat", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    chat_drawer_widget()


def complaint_page():
    top_hero("Submit Complaint", "Write your issue in any supported language. Baraka will route + reply instantly.")
    nav_bar("user")
    require_api_key_banner()

    with st.form("complaint_form", clear_on_submit=True):
        text = st.text_area("Complaint / Inquiry", height=140,
                            placeholder="e.g., ATM debited me but I got no cash...")
        priority = st.selectbox("Priority", ["Normal", "High", "Urgent"])
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and text.strip():
        if not OPENAI_API_KEY:
            st.error("Please set OPENAI_API_KEY for multilingual support.")
            return

        detected_lang, text_en = detect_and_translate_to_english(text.strip())
        out_lang = st.session_state.get("preferred_lang") or detected_lang

        dept, score, method = route_department(text_en)

        summary = text.strip()[:180] + ("..." if len(text.strip()) > 180 else "")
        ticket_id = create_complaint(st.session_state.user, text.strip(), dept,
                                     priority=priority, summary=summary)
        st.session_state.active_ticket = ticket_id

        st.success(f"Complaint submitted. Ticket #{ticket_id} routed to {dept} ({DEPT_LABELS.get(dept)}).")

        ans, source, sc = generate_reply(text.strip(), text_en, st.session_state.user, dept, out_lang)
        st.markdown("### Baraka’s Reply")
        st.write(ans)
        st.caption(f"Dept: {dept} | Routing: {score:.2f} ({method}) | Source: {source} | Similarity: {sc:.2f}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    chat_drawer_widget()


def chat_page():
    top_hero("Chat", "This is a chatbot drawer — open/close it as needed (not full-screen conversation).")
    nav_bar("user")
    require_api_key_banner()

    # Big fix: we do NOT render fake HTML wrappers. We use real Streamlit chat components.
    chat_drawer_widget(expanded_default=True)


def admin_page():
    top_hero("Admin Console", "Manage FAQs, complaints, and chat logs. Keep answers official + consistent.")
    nav_bar("admin")

    tabs = st.tabs(["➕ Add FAQ", "🛠️ Manage FAQs", "📥 Complaints", "📊 Chat Logs"])

    with tabs[0]:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        dept = st.selectbox("Department", DEPARTMENTS)
        q = st.text_area("Customer Question")
        a = st.text_area("Official Answer")
        tags = st.text_input("Tags (comma separated)")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        save = st.button("Save FAQ", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if save:
            if q.strip() and a.strip():
                add_custom_faq(dept, q.strip(), a.strip(), tags.strip(), st.session_state.user)
                st.success("FAQ added."); st.rerun()
            else:
                st.error("Question and answer are required.")

    with tabs[1]:
        filter_dept = st.selectbox("Filter by Department", ["ALL"] + DEPARTMENTS)
        df = fetch_custom_faqs(filter_dept)
        if df.empty:
            st.info("No custom FAQs yet.")
        else:
            st.dataframe(df[["id","department","question","answer","tags","updated_at"]], use_container_width=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            edit_id = st.selectbox("Select FAQ ID to edit/delete", df["id"].tolist())
            row = df[df["id"] == edit_id].iloc[0]

            new_dept = st.selectbox("Department", DEPARTMENTS,
                                    index=DEPARTMENTS.index(row["department"]) if row["department"] in DEPARTMENTS else 0)
            new_q = st.text_area("Question", value=row["question"])
            new_a = st.text_area("Answer", value=row["answer"])
            new_tags = st.text_input("Tags", value=row.get("tags","") or "")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                if st.button("Update FAQ", use_container_width=True):
                    update_custom_faq(edit_id, new_dept, new_q.strip(), new_a.strip(), new_tags.strip())
                    st.success("Updated."); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                if st.button("Delete FAQ", use_container_width=True):
                    delete_custom_faq(edit_id)
                    st.warning("Deleted."); st.rerun()

    with tabs[2]:
        STATUS_OPTS = ["Open", "In Review", "Resolved", "Rejected"]
        PRIORITY_OPTS = ["Normal", "High", "Urgent"]

        colA, colB = st.columns(2)
        with colA:
            dept_filter = st.selectbox("Department", ["ALL"] + DEPARTMENTS)
        with colB:
            status_filter = st.selectbox("Status", ["ALL"] + STATUS_OPTS)

        comp_df = fetch_complaints(dept_filter, status_filter)
        if comp_df.empty:
            st.info("No complaints found.")
        else:
            st.dataframe(comp_df[["id","username","department","priority","status","summary","created_at"]],
                         use_container_width=True)

            cid = st.selectbox("Open Ticket", comp_df["id"].tolist())
            row = comp_df[comp_df["id"] == cid].iloc[0]

            st.markdown("### Ticket Details")
            st.write(f"Customer: {row['username']}")
            st.write(f"Department: {row['department']} — {DEPT_LABELS.get(row['department'], row['department'])}")
            st.write(f"Priority: {row['priority']} | Status: {row['status']}")
            st.write(row["text"])

            new_status = st.selectbox("Update Status", STATUS_OPTS,
                                      index=STATUS_OPTS.index(row["status"]) if row["status"] in STATUS_OPTS else 0)
            new_priority = st.selectbox("Update Priority", PRIORITY_OPTS,
                                        index=PRIORITY_OPTS.index(row["priority"]) if row["priority"] in PRIORITY_OPTS else 0)
            notes = st.text_area("Internal Notes", value=row.get("internal_notes") or "", height=120)

            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            if st.button("Save Updates", use_container_width=True):
                update_complaint(cid, status=new_status, priority=new_priority, internal_notes=notes)
                st.success("Saved."); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        conn = get_conn()
        logs = pd.read_sql_query("SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 800", conn)
        conn.close()
        if logs.empty:
            st.info("No chats yet.")
        else:
            st.dataframe(logs, use_container_width=True)


# ----------------------------
# CHAT DRAWER WIDGET (retractable chatbot)
# ----------------------------
def chat_drawer_widget(expanded_default: bool = False):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "preferred_lang" not in st.session_state:
        st.session_state.preferred_lang = None

    # Seed welcome once
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hi — I’m Baraka. Ask about loans, accounts, ATM issues, transfers, fees. I won’t ask for PINs/OTPs."}
        )

    with st.expander("💬 Baraka Chat (open / close)", expanded=expanded_default):
        st.markdown('<div class="chat-drawer-hint">Tip: say “reply in Kiswahili / Amharic / Somali / Arabic / Shona / Kikuyu / Dholuo / Kamba / Kisii / Meru”.</div>', unsafe_allow_html=True)

        # Show messages using Streamlit chat (looks like a real chatbot)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # Input row
        with st.form("chat_send", clear_on_submit=True):
            q = st.text_input("Message Baraka…", placeholder="e.g., nataka kuchukua mkopo")
            col1, col2 = st.columns([1, 1])
            send = col1.form_submit_button("Send")
            clear = col2.form_submit_button("Clear chat")

        if clear:
            st.session_state.messages = [{"role": "assistant", "content": "Hi — I’m Baraka. How can I help today?"}]
            st.rerun()

        if send and q.strip():
            user = st.session_state.get("user", "guest")
            st.session_state.messages.append({"role": "user", "content": q.strip()})

            # Language command?
            handled, msg, forced = handle_language_command(q.strip())
            if handled:
                st.session_state.preferred_lang = forced
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()

            if not OPENAI_API_KEY:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "OPENAI_API_KEY is not set. Please set it to enable translation + multilingual replies."
                })
                st.rerun()

            # Detect language and translate to English for routing/retrieval
            detected_lang, q_en = detect_and_translate_to_english(q.strip())
            out_lang = st.session_state.get("preferred_lang") or detected_lang

            dept, dscore, dmethod = route_department(q_en)
            ans, source, score = generate_reply(q.strip(), q_en, user, dept, out_lang)

            footer_en = f"Dept: {dept} ({DEPT_LABELS.get(dept)}) • Routing: {dscore:.2f} ({dmethod}) • Source: {source}"
            footer = translate_from_english(footer_en, out_lang)

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{ans}\n\n— {footer}"
            })
            st.rerun()


# ----------------------------
# APP ROUTER
# ----------------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide")
    init_db()

    if "page" not in st.session_state:
        st.session_state.page = "login"

    page = st.session_state.page
    role = st.session_state.get("role")

    if page == "login":
        login_page()
        return

    if role == "admin":
        st.session_state.page = "admin" if page != "admin" else "admin"
        admin_page()
        return

    # user pages
    if page == "home":
        user_home_page()
    elif page == "complaint":
        complaint_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
