# app.py — Baraka (Modern Light UI) + Complaint Management + Admin + Multilingual Chat
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

TRANSLATION_MODEL = "gpt-4.1-mini"
FALLBACK_MODEL = "gpt-4.1-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----------------------------
# LANGUAGES
# ----------------------------
LANGS = {
    "auto": "Auto-detect",
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
SUPPORTED_LANG_CODES = {k for k in LANGS.keys() if k != "auto"}


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

DEPT_TRAIN = {
    "ACCOUNT": [
        "open account", "create account", "close account", "account frozen",
        "recent transactions", "bank statement", "account verification", "kyc update",
        "check balance", "account balance"
    ],
    "ATM": [
        "atm swallowed my card", "no cash but debited", "failed withdrawal",
        "atm reversal", "withdrawal dispute"
    ],
    "CARD": [
        "apply for card", "activate card", "block card", "cancel card", "card not working",
        "international usage", "annual fee", "mastercard", "visa", "debit card", "credit card"
    ],
    "CONTACT": [
        "customer care", "speak to agent", "human agent", "call center", "contact support"
    ],
    "FEES": [
        "charges too high", "check fees", "annual charges", "fee dispute", "pricing"
    ],
    "FIND": [
        "find atm", "nearest atm", "find branch", "branch near me"
    ],
    "LOAN": [
        "apply for loan", "loan repayment", "mortgage",
        "loan status", "interest rate", "borrow money", "take a loan", "get a loan"
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
    "LOAN": ["loan", "borrow", "interest", "repayment", "mortgage", "kopa", "mkopo"],
    "TRANSFER": ["transfer", "send money", "reversal", "pending", "reverse transaction"],
    "PASSWORD": ["password", "pin", "reset", "forgot", "login problem"],
    "FEES": ["fees", "charges", "pricing", "annual fee"],
    "FIND": ["find atm", "nearest atm", "branch", "locator"],
    "CONTACT": ["agent", "customer care", "call center", "contact support"],
    "ACCOUNT": ["account", "statement", "transactions", "balance", "kyc"]
}


# ----------------------------
# MODERN LIGHT UI (FULL BASEWEB STYLING FIX)
# ----------------------------
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

:root{
  --ink:#0a1220;
  --muted:rgba(10,18,32,.62);
  --muted2:rgba(10,18,32,.40);
  --card:rgba(255,255,255,.74);
  --card2:rgba(255,255,255,.88);
  --border:rgba(10,18,32,.10);
  --shadow: 0 18px 55px rgba(17, 24, 39, 0.10);
  --shadow2: 0 10px 25px rgba(17, 24, 39, 0.10);

  --p1:#5b7cfa;
  --p2:#a855f7;
  --p3:#22c55e;
  --p4:#fb923c;

  --chip: rgba(255,255,255,.82);
  --field: rgba(255,255,255,.96);
}

/* Kill Streamlit "Deploy" toolbar + menu noise */
#MainMenu, footer {visibility: hidden;}
[data-testid="stToolbar"]{display:none !important;}
header[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stStatusWidget"]{display:none !important;}
[data-testid="stHeaderActionElements"]{display:none !important;}

/* Global font + background */
html, body, [data-testid="stAppViewContainer"]{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  color: var(--ink) !important;
  background: transparent !important;
}

/* Bright soft background circles */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(720px 720px at 10% 12%, rgba(183,205,250,.90) 0%, rgba(183,205,250,0) 60%),
    radial-gradient(760px 760px at 92% 12%, rgba(215,198,255,.88) 0%, rgba(215,198,255,0) 58%),
    radial-gradient(900px 900px at 50% 94%, rgba(191,243,225,.85) 0%, rgba(191,243,225,0) 55%),
    radial-gradient(760px 760px at 92% 92%, rgba(255,225,184,.82) 0%, rgba(255,225,184,0) 55%),
    linear-gradient(180deg, #f8fbff 0%, #fdf7ff 100%) !important;
}

/* Container spacing */
.block-container{
  max-width: 1100px;
  padding-top: 1.0rem !important;
  padding-bottom: 3.8rem !important;
}

/* Force text colors everywhere (fix your "white text" issue) */
.stMarkdown, .stMarkdown * ,
[data-testid="stSidebar"], [data-testid="stSidebar"] * ,
label, label * , p, span, div, small, li, h1,h2,h3,h4,h5,h6{
  color: var(--ink) !important;
}

/* Muted helper */
.small{color: var(--muted) !important; font-size: 0.95rem;}
.caption{color: var(--muted2) !important; font-size: 0.86rem;}

/* Cards */
.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: var(--shadow);
  padding: 16px 18px;
  backdrop-filter: blur(12px);
}
.card2{
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: var(--shadow2);
  padding: 14px 16px;
  backdrop-filter: blur(12px);
}
.chip{
  display:inline-flex;
  align-items:center;
  gap:10px;
  padding:10px 14px;
  border-radius: 999px;
  background: var(--chip);
  border: 1px solid var(--border);
  box-shadow: var(--shadow2);
  font-weight: 800;
}

/* -----------------------------
   Inputs / Textareas
----------------------------- */
.stTextInput input, .stTextArea textarea{
  border-radius: 16px !important;
  border: 1px solid rgba(10,18,32,.14) !important;
  background: var(--field) !important;
  color: var(--ink) !important;
  box-shadow: 0 10px 18px rgba(17,24,39,.06);
}

/* BaseWeb SELECT overrides (this is why yours was dark) */
[data-baseweb="select"] > div{
  border-radius: 16px !important;
  border: 1px solid rgba(10,18,32,.14) !important;
  background: var(--field) !important;
  box-shadow: 0 10px 18px rgba(17,24,39,.06) !important;
}
[data-baseweb="select"] *{
  color: var(--ink) !important;
}
[data-baseweb="menu"]{
  background: rgba(255,255,255,.98) !important;
  border: 1px solid rgba(10,18,32,.12) !important;
  border-radius: 16px !important;
  box-shadow: var(--shadow2) !important;
}
[data-baseweb="menu"] *{ color: var(--ink) !important; }

/* -----------------------------
   Buttons
----------------------------- */
.stButton button{
  border-radius: 16px !important;
  border: 1px solid rgba(91,124,250,0.22) !important;
  background: linear-gradient(135deg, rgba(91,124,250,.22), rgba(168,85,247,.16)) !important;
  color: rgba(10,18,32,0.96) !important;
  font-weight: 800 !important;
  padding: 0.72rem 1.05rem !important;
  box-shadow: var(--shadow2);
  transition: transform .12s ease, filter .12s ease;
}
.stButton button:hover{
  transform: translateY(-1px);
  filter: saturate(1.07);
}

/* Make "secondary" small buttons look clean */
button[kind="secondary"]{
  background: rgba(255,255,255,.82) !important;
  border: 1px solid rgba(10,18,32,.12) !important;
}

/* -----------------------------
   Sidebar
----------------------------- */
[data-testid="stSidebar"]{
  background: rgba(255,255,255,.78) !important;
  border-right: 1px solid rgba(10,18,32,.10) !important;
  backdrop-filter: blur(12px);
}
.sidebar-card{
  background: rgba(255,255,255,.90);
  border: 1px solid rgba(10,18,32,.10);
  border-radius: 18px;
  padding: 12px 12px;
  box-shadow: var(--shadow2);
}

/* Fix radio showing dots only */
[data-baseweb="radio"] span{
  color: var(--ink) !important;
  font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
  color: var(--muted) !important;
}

/* -----------------------------
   Chat
----------------------------- */
[data-testid="stChatMessage"]{
  border-radius: 18px !important;
  border: 1px solid rgba(10,18,32,.10) !important;
  background: rgba(255,255,255,.90) !important;
  padding: 10px !important;
  box-shadow: 0 10px 25px rgba(17, 24, 39, 0.06);
}

/* Remove any dark bottom bar */
footer, .stBottom, [data-testid="stBottom"], [data-testid="stBottomBlockContainer"]{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}
[data-testid="stChatInput"] textarea{
  border-radius: 18px !important;
  border: 1px solid rgba(10,18,32,.14) !important;
  background: rgba(255,255,255,.97) !important;
  color: var(--ink) !important;
}

/* -----------------------------
   Segmented control (pill) styling for Admin sections
----------------------------- */
.segWrap [data-baseweb="radio"] > div{
  display:flex !important;
  gap:10px !important;
  flex-wrap:wrap !important;
}
.segWrap label{
  background: rgba(255,255,255,.86) !important;
  border: 1px solid rgba(10,18,32,.10) !important;
  border-radius: 999px !important;
  padding: 10px 14px !important;
  box-shadow: 0 10px 18px rgba(17,24,39,.06) !important;
}
.segWrap input:checked + div{
  filter: saturate(1.05);
}
.segWrap label:has(input:checked){
  background: linear-gradient(135deg, rgba(91,124,250,.20), rgba(168,85,247,.14)) !important;
  border-color: rgba(91,124,250,.25) !important;
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius: 18px !important;
  overflow: hidden !important;
  border: 1px solid rgba(10,18,32,.10) !important;
  box-shadow: 0 10px 25px rgba(17,24,39,.06);
}

/* Subtle page fade-in */
@keyframes fadeUp{from{opacity:0; transform: translateY(6px);} to{opacity:1; transform: translateY(0);}}
.block-container{animation: fadeUp .22s ease-out;}
hr{border:none; border-top:1px solid rgba(10,18,32,.10); margin: 12px 0;}
</style>
"""


# ----------------------------
# OpenAI helper
# ----------------------------
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

def _safe_json_parse(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            i = s.find("{")
            j = s.rfind("}")
            if i >= 0 and j > i:
                return json.loads(s[i:j+1])
        except Exception:
            return None
    return None


# ----------------------------
# Placeholder protection
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


# ----------------------------
# Language detect + translation (API-only)
# ----------------------------
@st.cache_data(show_spinner=False)
def detect_and_translate_to_english(user_text: str) -> Tuple[str, str]:
    client = get_openai_client()
    if not client:
        return "en", user_text

    protected, mapping = protect_placeholders(user_text)

    system = (
        "Return ONLY valid JSON: {\"lang\":\"...\",\"english\":\"...\"}.\n"
        "lang MUST be one of: " + ", ".join(sorted(SUPPORTED_LANG_CODES)) + "\n"
        "Translate INPUT to English.\n"
        "Preserve placeholders like @@PH0@@ exactly."
    )
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

@st.cache_data(show_spinner=False)
def translate_from_english(text_en: str, target_lang: str) -> str:
    if target_lang == "en":
        return text_en
    client = get_openai_client()
    if not client:
        return text_en

    protected, mapping = protect_placeholders(text_en)

    system = (
        f"Translate from English to {LANGS.get(target_lang, target_lang)}.\n"
        "Output ONLY the translation.\n"
        "Preserve placeholders like @@PH0@@ exactly.\n"
        "Make it natural and fluent."
    )
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

def handle_language_command(user_text: str):
    t = (user_text or "").strip().lower()
    name_to_code = {
        "english": "en",
        "kiswahili": "sw", "swahili": "sw",
        "amharic": "am",
        "somali": "so",
        "arabic": "ar",
        "shona": "sn",
        "kikuyu": "ki", "gikuyu": "ki",
        "dholuo": "luo", "luo": "luo",
        "kamba": "kam", "kikamba": "kam",
        "kisii": "guz", "ekegusii": "guz", "gusii": "guz",
        "meru": "mer", "kimeru": "mer",
    }
    if "reply in " in t or "jibu kwa " in t:
        for k, code in name_to_code.items():
            if k in t:
                msg = translate_from_english(f"Okay — I’ll reply in {LANGS.get(code, code)}.", code)
                return True, msg, code
    return False, None, None


# ----------------------------
# Password hashing (PBKDF2)
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
# DB helpers
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
        c.execute("INSERT INTO users(username,pw_hash,role) VALUES(?,?,?)", (username, new_hash, role))
    else:
        stored_hash = row[0]
        if not is_pbkdf2_hash(stored_hash):
            c.execute("UPDATE users SET pw_hash=?, role=? WHERE username=?", (new_hash, role, username))

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        pw_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','admin'))
    );""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS custom_faqs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        department TEXT DEFAULT 'CONTACT',
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        tags TEXT,
        created_by TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")

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
    );""")

    # Chat logs WITH columns (fix older DBs)
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
    );""")

    # Upgrade if older
    if not column_exists(conn, "chat_logs", "source"):
        c.execute("ALTER TABLE chat_logs ADD COLUMN source TEXT")
    if not column_exists(conn, "chat_logs", "score"):
        c.execute("ALTER TABLE chat_logs ADD COLUMN score REAL")
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
    return role if verify_password(password, pw_hash) else None

def fetch_custom_faqs(department=None):
    conn = get_conn()
    if department and department != "ALL":
        df = pd.read_sql_query(
            "SELECT * FROM custom_faqs WHERE department=? ORDER BY updated_at DESC",
            conn, params=(department,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM custom_faqs ORDER BY updated_at DESC", conn)
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
    fields, params = [], []
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
# Dataset + retrieval
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

    if not qcol or not acol:
        raise ValueError(f"Could not detect question/answer columns. Found: {df.columns}")

    base_df = df[[qcol, acol]].rename(columns={qcol: "question", acol: "answer"})
    base_df["question"] = base_df["question"].astype(str)
    base_df["answer"] = base_df["answer"].astype(str)
    base_df["category"] = df[catcol].astype(str).str.upper() if catcol else "CONTACT"
    base_df.dropna(inplace=True)
    base_df.reset_index(drop=True, inplace=True)
    return base_df

@st.cache_resource(show_spinner=False)
def build_vector_index(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2 if len(texts) >= 3 else 1,
        stop_words="english" if len(texts) >= 3 else None
    )
    X = vec.fit_transform(texts)
    return vec, X

def normalize(text):
    text = (text or "").lower().strip()
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
    qv = vec.transform([normalize(query_en)])
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
        return ("Set OPENAI_API_KEY to enable AI answers.", 0.0, "fallback")

    system = (
        "Your name is Baraka. You are a helpful Kenyan retail-banking & SACCO support assistant. "
        "Answer ONLY using the provided context. If insufficient, ask ONE short follow-up question. "
        "Never request PINs, passwords, or OTPs. "
        f"Reply in {LANGS.get(out_lang, 'English')}."
    )
    user = f"Customer question (English): {query_en}\n\nContext:\n" + "\n---\n".join(context_snippets)
    resp = client.responses.create(
        model=FALLBACK_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2
    )
    return (resp.output_text or "").strip(), 0.0, "openai"

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
# UI helpers
# ----------------------------
PAGES = ["Home", "Complaint", "Chat", "Admin"]

def do_logout():
    for k in ["user", "role", "page", "messages", "preferred_lang", "ui_lang_choice", "active_page"]:
        st.session_state.pop(k, None)
    st.session_state.page = "login"
    st.rerun()

def top_bar():
    c1, c2, c3 = st.columns([6, 2, 2])
    with c1:
        st.markdown(f"<span class='chip'>✨ {APP_NAME}</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='caption'>Multilingual SACCO support</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Logout", use_container_width=True):
            do_logout()

def sidebar_panel():
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown(f"### {APP_NAME}")
        user = st.session_state.get("user")
        role = st.session_state.get("role")
        st.markdown(f"<div class='small'>Signed in as <b>{user}</b> ({role})</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card" style="margin-top:10px;">', unsafe_allow_html=True)
        st.markdown("**Navigation**")
        role = st.session_state.get("role", "user")
        opts = ["Home", "Complaint", "Chat"] + (["Admin"] if role == "admin" else [])
        current = st.session_state.get("active_page", "Chat")
        st.session_state.active_page = st.radio(
            " ",
            opts,
            index=opts.index(current) if current in opts else 0
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card" style="margin-top:10px;">', unsafe_allow_html=True)
        st.markdown("**Reply language**")
        if "ui_lang_choice" not in st.session_state:
            st.session_state.ui_lang_choice = "auto"
        st.session_state.ui_lang_choice = st.selectbox(
            "Language",
            list(LANGS.keys()),
            format_func=lambda k: LANGS[k],
            index=list(LANGS.keys()).index(st.session_state.ui_lang_choice),
        )
        st.markdown("<div class='caption'>Tip: type “reply in Kiswahili” in chat to lock language.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card" style="margin-top:10px;">', unsafe_allow_html=True)
        st.session_state.debug = st.toggle("Show debug errors", value=st.session_state.get("debug", False))
        if st.button("Logout (sidebar)", use_container_width=True):
            do_logout()
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Pages
# ----------------------------
def login_page():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <h1 style="margin:0;">{APP_NAME}</h1>
      <div class="small">Sign in to submit complaints, add FAQs, and chat with Baraka.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.markdown("**Demo accounts**")
        st.write("admin / admin123")
        st.write("user / user123")
        if not OPENAI_API_KEY:
            st.info("Set OPENAI_API_KEY to enable multilingual replies + AI fallback.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.markdown("### Sign in")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            role = verify_user(username, password)
            if role:
                st.session_state.user = username
                st.session_state.role = role
                st.session_state.page = "app"
                st.session_state.active_page = "Chat"
                st.rerun()
            else:
                st.error("Invalid username/password.")
        st.markdown("</div>", unsafe_allow_html=True)

def home_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Home")
    st.markdown("<div class='small'>Use the sidebar to navigate. Baraka supports multilingual SACCO help.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def complaint_page():
    debug = st.session_state.get("debug", False)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Complaint / Inquiry")
    st.markdown("<div class='small'>Describe your issue clearly — Baraka routes it automatically.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("<div class='card2'>", unsafe_allow_html=True)
    with st.form("complaint_form", clear_on_submit=True):
        text = st.text_area("Complaint / Inquiry", height=140, placeholder="e.g., ATM debited me but I got no cash…")
        priority = st.selectbox("Priority", ["Normal", "High", "Urgent"])
        submitted = st.form_submit_button("Submit")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and text.strip():
        try:
            user = st.session_state.user
            if OPENAI_API_KEY:
                detected_lang, text_en = detect_and_translate_to_english(text.strip())
                out_lang = detected_lang
                chosen = st.session_state.get("ui_lang_choice", "auto")
                if chosen != "auto":
                    out_lang = chosen
            else:
                text_en = text.strip()
                out_lang = "en"

            dept, score, method = route_department(text_en)
            summary = text.strip()[:180] + ("..." if len(text.strip()) > 180 else "")
            ticket_id = create_complaint(user, text.strip(), dept, priority=priority, summary=summary)

            st.success(f"Submitted successfully. Ticket #{ticket_id}")

            st.markdown("<div class='card2'>", unsafe_allow_html=True)
            st.write(f"**Routed Department:** {dept} — {DEPT_LABELS.get(dept, dept)}")
            st.write(f"**Routing Confidence:** {score:.2f} ({method})")
            st.markdown("</div>", unsafe_allow_html=True)

            if OPENAI_API_KEY:
                ans, source, sc = generate_reply(text.strip(), text_en, user, dept, out_lang)
                st.markdown("<div class='card2'>", unsafe_allow_html=True)
                st.markdown("**Baraka reply:**")
                st.write(ans)
                st.markdown(f"<div class='caption'>Source: {source} | Similarity: {sc:.2f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(repr(e) if debug else "Failed to submit complaint. Turn on debug to see details.")

def chat_page():
    debug = st.session_state.get("debug", False)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Chat")
    st.markdown("<div class='small'>Just chat. FAQs, complaints, and admin tools are in the sidebar.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "preferred_lang" not in st.session_state:
        st.session_state.preferred_lang = None

    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi — I’m Baraka. Ask about accounts, cards, loans, ATM issues, transfers, fees. I won’t ask for PINs/OTPs."
        })

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_text = st.chat_input("Type your message…")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    handled, msg, forced = handle_language_command(user_text)
    if handled:
        st.session_state.preferred_lang = forced
        with st.chat_message("assistant"):
            st.write(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.rerun()

    if not OPENAI_API_KEY:
        assistant_msg = "Please set OPENAI_API_KEY to enable multilingual replies + AI answers."
        with st.chat_message("assistant"):
            st.write(assistant_msg)
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        st.rerun()

    user = st.session_state.get("user", "guest")

    try:
        with st.spinner("Baraka is thinking…"):
            detected_lang, q_en = detect_and_translate_to_english(user_text)

            chosen = st.session_state.get("ui_lang_choice", "auto")
            out_lang = chosen if chosen != "auto" else (st.session_state.get("preferred_lang") or detected_lang)

            dept, dscore, dmethod = route_department(q_en)
            ans, source, score = generate_reply(user_text, q_en, user, dept, out_lang)

            footer_en = f"Dept: {dept} • Routing: {dscore:.2f} ({dmethod}) • Source: {source}"
            footer = translate_from_english(footer_en, out_lang) if OPENAI_API_KEY else footer_en
            assistant_msg = f"{ans}\n\n— {footer}"

        with st.chat_message("assistant"):
            st.write(assistant_msg)

        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        st.rerun()

    except Exception as e:
        assistant_msg = "OpenAI call failed. Check your API key and `openai` package."
        if debug:
            assistant_msg += f"\n\nError: {repr(e)}"
        with st.chat_message("assistant"):
            st.write(assistant_msg)
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        st.rerun()

def safe_index(options, value, fallback_value=None):
    value_norm = "" if value is None else str(value).strip().upper()
    options_norm = [str(o).strip().upper() for o in options]
    if value_norm in options_norm:
        return options_norm.index(value_norm)
    if fallback_value is not None:
        fb = str(fallback_value).strip().upper()
        if fb in options_norm:
            return options_norm.index(fb)
    return 0

def admin_page():
    debug = st.session_state.get("debug", False)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Admin Console")
    st.markdown("<div class='small'>Manage department FAQs, complaints, and chat logs.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Modern segmented control (no Streamlit tabs)
    st.markdown("<div class='segWrap'>", unsafe_allow_html=True)
    section = st.radio(
        " ",
        ["➕ Add FAQ", "🛠️ Manage FAQs", "📥 Complaints", "📊 Chat Logs"],
        index=["➕ Add FAQ", "🛠️ Manage FAQs", "📥 Complaints", "📊 Chat Logs"].index(
            st.session_state.get("admin_section", "➕ Add FAQ")
        ),
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.admin_section = section

    if section == "➕ Add FAQ":
        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        dept = st.selectbox("Department", DEPARTMENTS)
        q = st.text_area("Customer Question", height=110)
        a = st.text_area("Official Answer", height=130)
        tags = st.text_input("Tags / Keywords (comma separated)")
        save = st.button("Save FAQ", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if save:
            if q.strip() and a.strip():
                add_custom_faq(dept, q.strip(), a.strip(), tags.strip(), st.session_state.user)
                st.success("FAQ added.")
                st.rerun()
            else:
                st.error("Question and Answer are required.")

    elif section == "🛠️ Manage FAQs":
        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        filter_dept = st.selectbox("Filter by Department", ["ALL"] + DEPARTMENTS)
        df = fetch_custom_faqs(filter_dept)
        st.markdown("</div>", unsafe_allow_html=True)

        if df.empty:
            st.info("No custom FAQs yet.")
            return

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.dataframe(df[["id","department","question","answer","tags","updated_at"]], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        edit_id = st.selectbox("Select FAQ ID to edit/delete", df["id"].tolist())
        row = df[df["id"] == edit_id].iloc[0]

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        new_dept = st.selectbox(
            "Edit Department",
            DEPARTMENTS,
            index=safe_index(DEPARTMENTS, row.get("department"), fallback_value="CONTACT")
        )
        new_q = st.text_area("Edit Question", value=row["question"], height=110)
        new_a = st.text_area("Edit Answer", value=row["answer"], height=130)
        new_tags = st.text_input("Edit Tags", value=row.get("tags","") or "")

        colE, colD = st.columns(2)
        with colE:
            if st.button("Update FAQ", use_container_width=True):
                update_custom_faq(edit_id, new_dept, new_q.strip(), new_a.strip(), new_tags.strip())
                st.success("FAQ updated.")
                st.rerun()
        with colD:
            if st.button("Delete FAQ", use_container_width=True):
                delete_custom_faq(edit_id)
                st.warning("FAQ deleted.")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "📥 Complaints":
        STATUS_OPTS = ["Open", "In Review", "Resolved", "Rejected"]
        PRIORITY_OPTS = ["Normal", "High", "Urgent"]

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            dept_filter = st.selectbox("Department Queue", ["ALL"] + DEPARTMENTS)
        with colB:
            status_filter = st.selectbox("Status", ["ALL"] + STATUS_OPTS)
        st.markdown("</div>", unsafe_allow_html=True)

        comp_df = fetch_complaints(dept_filter, status_filter)
        if comp_df.empty:
            st.info("No complaints found.")
            return

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.dataframe(
            comp_df[["id","username","department","priority","status","summary","created_at"]],
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        cid = st.selectbox("Open Complaint Ticket", comp_df["id"].tolist())
        row = comp_df[comp_df["id"] == cid].iloc[0]

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.markdown(f"**Ticket #{cid}**")
        st.write(f"**Customer:** {row['username']}")
        st.write(f"**Department:** {row['department']} — {DEPT_LABELS.get(row['department'], row['department'])}")
        st.write(f"**Priority:** {row['priority']}")
        st.write(f"**Status:** {row['status']}")
        st.write("**Complaint Text:**")
        st.write(row["text"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        new_status = st.selectbox(
            "Update Status",
            STATUS_OPTS,
            index=safe_index(STATUS_OPTS, row.get("status"), fallback_value="Open")
        )
        new_priority = st.selectbox(
            "Update Priority",
            PRIORITY_OPTS,
            index=safe_index(PRIORITY_OPTS, row.get("priority"), fallback_value="Normal")
        )
        notes = st.text_area("Internal Notes", value=row.get("internal_notes") or "", height=120)

        if st.button("Save Complaint Updates", use_container_width=True):
            update_complaint(cid, status=new_status, priority=new_priority, internal_notes=notes)
            st.success("Complaint updated.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "📊 Chat Logs":
        conn = get_conn()
        try:
            logs = pd.read_sql_query(
                "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 1500",
                conn
            )
        finally:
            conn.close()

        if logs.empty:
            st.info("No chats yet.")
            return

        st.markdown("<div class='card2'>", unsafe_allow_html=True)
        st.dataframe(logs, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# APP ROUTER
# ----------------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide", initial_sidebar_state="expanded")
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    init_db()

    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Chat"

    if st.session_state.page == "login":
        login_page()
        return

    top_bar()
    sidebar_panel()

    role = st.session_state.get("role", "user")
    page = st.session_state.get("active_page", "Chat")

    if page == "Home":
        home_page()
    elif page == "Complaint":
        complaint_page()
    elif page == "Chat":
        chat_page()
    elif page == "Admin":
        if role != "admin":
            st.warning("Admin only.")
            home_page()
        else:
            admin_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
