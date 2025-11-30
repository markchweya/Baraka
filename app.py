# app.py — Baraka: AI Complaint Management + Department Routing + Banking Chatbot (Streamlit)
#
# Demo accounts (auto-created / auto-upgraded):
#   admin / admin123
#   user  / user123
#
# requirements.txt (repo root):
# streamlit
# openai
# pandas
# scikit-learn
# pyarrow
# huggingface-hub
# fsspec
#
# ✅ Uses OPENAI_API_KEY for:
# - language detect + translation (no offline CPU translator models)
# - optional AI fallback answers


import os
import re
import json
import sqlite3
import base64
import hashlib
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

# Translation + fallback
TRANSLATION_MODEL = "gpt-4.1-mini"
FALLBACK_MODEL = "gpt-4.1-mini"

LANG_NAME = {
    "en": "English",
    "sw": "Kiswahili",
    "am": "Amharic",
    "so": "Somali",
    "ar": "Arabic",
}


# ----------------------------
# DEPARTMENTS (CATEGORIES)
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

# Tiny, local dummy routing bank (rule-based + TF-IDF)
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
        "activate card", "block card", "cancel card", "card not working",
        "international usage", "annual fee", "card balance"
    ],
    "CONTACT": [
        "customer care", "speak to agent", "human agent", "call center", "contact support"
    ],
    "FEES": [
        "charges too high", "check fees", "annual charges", "fee dispute"
    ],
    "FIND": [
        "find atm", "nearest atm", "find branch", "branch near me"
    ],
    "LOAN": [
        "apply for loan", "loan repayment", "mortgage", "cancel loan",
        "loan status", "interest rate", "borrow money"
    ],
    "PASSWORD": [
        "reset password", "forgot password", "set up password", "login problem"
    ],
    "TRANSFER": [
        "cancel transfer", "make transfer", "wrong transfer", "pending transfer",
        "reverse transaction", "send money"
    ]
}

DEPT_KEYWORDS = {
    "ATM": ["atm", "cash withdrawal", "swallowed", "debit but no cash"],
    "CARD": ["card", "visa", "mastercard", "debit card", "credit card"],
    "LOAN": ["loan", "mortgage", "repayment", "interest", "borrow"],
    "TRANSFER": ["transfer", "send money", "reversal", "pending", "reverse transaction"],
    "PASSWORD": ["password", "pin reset", "forgot", "login problem"],
    "FEES": ["fees", "charges", "annual fee", "pricing"],
    "FIND": ["find atm", "branch", "nearest atm", "locator"],
    "CONTACT": ["agent", "customer care", "call center", "contact support"],
    "ACCOUNT": ["account", "statement", "transactions", "close account", "balance"]
}


# ----------------------------
# SAFE INDEX HELPER
# ----------------------------
def safe_index(options, value, fallback_value=None):
    if value is None:
        value_norm = ""
    else:
        value_norm = str(value).strip().upper()

    options_norm = [str(o).strip().upper() for o in options]
    if value_norm in options_norm:
        return options_norm.index(value_norm)

    if fallback_value is not None:
        fallback_norm = str(fallback_value).strip().upper()
        if fallback_norm in options_norm:
            return options_norm.index(fallback_norm)

    return 0


# ----------------------------
# PASSWORD HASHING (stdlib PBKDF2)
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
# MODERN LIGHT UI STYLING (NEW)
# ----------------------------
BANK_CSS = """
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* Theme tokens */
:root{
  --bg: #f6f8ff;
  --bg2:#f8fbff;
  --card: rgba(255,255,255,0.86);
  --card2: rgba(255,255,255,0.72);
  --stroke: rgba(18, 24, 40, 0.10);
  --stroke2: rgba(18, 24, 40, 0.14);
  --text:#0b1220;
  --muted:#556079;
  --muted2:#6b7280;

  --primary:#5b7cfa;
  --primary2:#22c55e;
  --accent:#7c3aed;
  --warn:#f59e0b;
  --danger:#ef4444;

  --shadow: 0 16px 40px rgba(16,24,40,0.12);
  --shadow2: 0 10px 24px rgba(16,24,40,0.10);
  --radius: 18px;
  --radius2: 14px;
}

/* Kill Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display:none !important;}
[data-testid="stDecoration"] {display:none !important;}

html, body, [data-testid="stAppViewContainer"]{
  font-family: "Plus Jakarta Sans", Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  background: radial-gradient(1100px 800px at 8% 0%, rgba(91,124,250,0.18), transparent 60%),
              radial-gradient(1000px 700px at 92% 10%, rgba(124,58,237,0.14), transparent 55%),
              radial-gradient(900px 650px at 50% 100%, rgba(34,197,94,0.10), transparent 50%),
              linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%) !important;
  color: var(--text) !important;
}

/* Global spacing */
.block-container{
  padding-top: 1.1rem !important;
  padding-bottom: 1.4rem !important;
  max-width: 1200px;
}

/* Subtle animated background glows */
@keyframes floaty {
  0% { transform: translateY(0px) translateX(0px); filter: blur(0px); }
  50% { transform: translateY(-10px) translateX(8px); filter: blur(0.2px); }
  100% { transform: translateY(0px) translateX(0px); filter: blur(0px); }
}
.bg-orb{
  position: fixed;
  z-index: 0;
  width: 340px;
  height: 340px;
  border-radius: 999px;
  filter: blur(40px);
  opacity: 0.55;
  animation: floaty 7s ease-in-out infinite;
  pointer-events: none;
}
.bg-orb.one{ left: -120px; top:-120px; background: rgba(91,124,250,0.28); }
.bg-orb.two{ right: -120px; top: 120px; background: rgba(124,58,237,0.22); animation-duration: 9s; }
.bg-orb.three{ left: 35%; bottom: -160px; background: rgba(34,197,94,0.18); animation-duration: 11s; }

/* Card */
.bank-card{
  position: relative;
  z-index: 1;
  background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.78) 100%);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 18px 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  overflow: hidden;
  animation: fadeUp 240ms ease-out both;
}
@keyframes fadeUp{
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0px); }
}

/* Nice header "chip" / badge */
.badge{
  display:inline-flex;
  gap:8px;
  align-items:center;
  padding:6px 12px;
  border-radius: 999px;
  font-size: 12.5px;
  font-weight: 650;
  color: #1f2a44;
  background: rgba(91,124,250,0.12);
  border: 1px solid rgba(91,124,250,0.20);
  box-shadow: 0 6px 16px rgba(91,124,250,0.08);
}
.badge-ok{ background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.20); }
.badge-warn{ background: rgba(245,158,11,0.14); border-color: rgba(245,158,11,0.22); }

.small-muted{
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
}

/* Titles */
h1,h2,h3,h4{
  letter-spacing: -0.02em;
}
h2{ font-weight: 750; }
h3{ font-weight: 750; }
a, a:visited{ color: var(--primary); }

/* Buttons (all) */
.stButton button, button[kind="primary"]{
  border: 1px solid rgba(91,124,250,0.25) !important;
  background: linear-gradient(135deg, rgba(91,124,250,0.98) 0%, rgba(124,58,237,0.94) 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 14px !important;
  padding: 0.68rem 1.05rem !important;
  box-shadow: 0 12px 22px rgba(91,124,250,0.22) !important;
  transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease !important;
}
.stButton button:hover{
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 16px 30px rgba(91,124,250,0.28) !important;
  filter: saturate(1.05);
}
.stButton button:active{
  transform: translateY(0px) scale(0.99);
  box-shadow: 0 10px 18px rgba(91,124,250,0.18) !important;
}

/* Secondary buttons: Streamlit sometimes renders different kinds */
button[kind="secondary"], .stButton button[kind="secondary"]{
  background: rgba(255,255,255,0.72) !important;
  color: var(--text) !important;
  border: 1px solid var(--stroke2) !important;
  box-shadow: var(--shadow2) !important;
}

/* Inputs */
label{ color: #111827 !important; font-weight: 650 !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox select, .stMultiSelect div, .stNumberInput input{
  background: rgba(255,255,255,0.88) !important;
  border: 1px solid var(--stroke2) !important;
  color: var(--text) !important;
  border-radius: 14px !important;
  box-shadow: 0 10px 20px rgba(16,24,40,0.06);
  transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus{
  border-color: rgba(91,124,250,0.55) !important;
  box-shadow: 0 14px 26px rgba(91,124,250,0.14) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap: 10px;
}
.stTabs [data-baseweb="tab"]{
  border-radius: 999px !important;
  border: 1px solid var(--stroke2) !important;
  background: rgba(255,255,255,0.66) !important;
  padding: 10px 14px !important;
  font-weight: 700 !important;
  color: #24324a !important;
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(91,124,250,0.16) 0%, rgba(124,58,237,0.12) 100%) !important;
  border-color: rgba(91,124,250,0.30) !important;
  box-shadow: 0 12px 22px rgba(91,124,250,0.10);
  transform: translateY(-1px);
}

/* Dataframes / tables */
[data-testid="stDataFrame"]{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--stroke);
  box-shadow: 0 16px 30px rgba(16,24,40,0.10);
}
[data-testid="stTable"]{
  border-radius: 16px;
  overflow: hidden;
}

/* Chat container */
.chat-wrap{
  position: relative;
  z-index: 1;
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 12px;
  max-height: 60vh;
  min-height: 200px;
  overflow-y: auto;
  box-shadow: var(--shadow2);
}

/* Nice scrollbar */
.chat-wrap::-webkit-scrollbar{ width: 10px; }
.chat-wrap::-webkit-scrollbar-track{ background: rgba(17,24,39,0.06); border-radius: 999px;}
.chat-wrap::-webkit-scrollbar-thumb{
  background: rgba(91,124,250,0.25);
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.55);
}
.chat-wrap::-webkit-scrollbar-thumb:hover{ background: rgba(91,124,250,0.35); }

/* Chat bubbles */
.bubble{
  width: fit-content;
  max-width: 78%;
  padding: 10px 12px;
  border-radius: 16px;
  margin: 8px 0;
  line-height: 1.55;
  font-size: 0.98rem;
  white-space: pre-wrap;
  animation: popIn 140ms ease-out both;
}
@keyframes popIn{
  from{ opacity:0; transform: translateY(4px) scale(0.985); }
  to{ opacity:1; transform: translateY(0px) scale(1); }
}
.user{
  margin-left: auto;
  background: linear-gradient(135deg, rgba(91,124,250,0.18) 0%, rgba(124,58,237,0.12) 100%);
  border: 1px solid rgba(91,124,250,0.22);
}
.bot{
  margin-right: auto;
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--stroke2);
}

/* Ticket panel */
.ticket{
  background: rgba(255,255,255,0.80);
  border: 1px dashed rgba(91,124,250,0.35);
  border-radius: 16px;
  padding: 12px 14px;
  margin-top: 10px;
  box-shadow: 0 16px 26px rgba(91,124,250,0.10);
}

/* Input card */
.input-card{
  position: relative;
  z-index: 1;
  background: rgba(255,255,255,0.78);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 12px 12px;
  box-shadow: var(--shadow2);
}

/* Reduce horizontal rules */
hr{
  border: none;
  border-top: 1px solid rgba(17,24,39,0.10);
  margin: 12px 0;
}

/* Alerts: make them look cleaner */
[data-testid="stAlert"]{
  border-radius: 16px !important;
  border: 1px solid rgba(17,24,39,0.10) !important;
  box-shadow: 0 14px 26px rgba(16,24,40,0.08) !important;
}

/* Mobile */
@media (max-width: 720px){
  .block-container{ padding: 0.8rem 0.8rem 1.2rem 0.8rem !important; }
  .bubble{ max-width: 90%; }
}
</style>

<div class="bg-orb one"></div>
<div class="bg-orb two"></div>
<div class="bg-orb three"></div>
"""


# ----------------------------
# OPENAI CLIENT (API KEY ONLY)
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
# PLACEHOLDER PROTECTION (so templates like {amount} don't get translated)
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
# LANGUAGE: detect + translate to English + translate back
# ----------------------------
def detect_and_translate_to_english(user_text: str):
    """
    Returns: (detected_lang_code, english_text)
    detected_lang_code in: en, sw, am, so, ar (fallback: en)
    """
    client = get_openai_client()
    if not client:
        return "en", user_text

    protected, mapping = protect_placeholders(user_text)

    system = (
        "You are a language detector and translator.\n"
        "Task:\n"
        "1) Detect the language of the INPUT.\n"
        "2) Translate the INPUT to English.\n\n"
        "Return ONLY valid JSON with keys: lang, english.\n"
        "lang must be one of: en, sw, am, so, ar.\n"
        "Preserve any placeholders exactly (tokens like @@PH0@@). Do NOT change them."
    )
    user = f"INPUT:\n{protected}"

    try:
        resp = client.responses.create(
            model=TRANSLATION_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0
        )
        raw = (resp.output_text or "").strip()
        data = json.loads(raw)

        lang = str(data.get("lang", "en")).strip().lower()
        eng = str(data.get("english", protected))

        if lang not in ("en", "sw", "am", "so", "ar"):
            lang = "en"

        eng = restore_placeholders(eng, mapping)
        return lang, eng
    except Exception:
        return "en", user_text


def translate_from_english(text_en: str, target_lang: str):
    if target_lang == "en":
        return text_en

    client = get_openai_client()
    if not client:
        return text_en

    protected, mapping = protect_placeholders(text_en)

    system = (
        "You are a professional translator.\n"
        f"Translate from English to {LANG_NAME.get(target_lang, target_lang)}.\n"
        "Rules:\n"
        "- Output ONLY the translation, no explanations.\n"
        "- Preserve placeholders exactly (tokens like @@PH0@@). Do NOT change them.\n"
        "- Keep numbers, currency, and product names unchanged unless the target language normally uses a different script."
    )

    try:
        resp = client.responses.create(
            model=TRANSLATION_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": protected},
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
    Optional explicit commands:
      - "jibu kwa kiswahili"
      - "reply in amharic"
      - "somali"
      - "arabic"
      - "english"
    """
    t = str(user_text).strip().lower()
    if not t:
        return False, None, None

    if "kiswahili" in t or "swahili" in t:
        return True, "Sawa—nitajibu kwa Kiswahili kuanzia sasa.", "sw"
    if "amharic" in t or "አማርኛ" in user_text:
        return True, "እሺ — ከአሁን በኋላ በአማርኛ እመልሳለሁ።", "am"
    if "somali" in t or "soomaali" in t:
        return True, "Haye—laga bilaabo hadda waxaan ku jawaabi doonaa Af-Soomaali.", "so"
    if "arabic" in t or "العربية" in user_text or "عربي" in user_text:
        return True, "حسنًا — سأجيب بالعربية من الآن فصاعدًا.", "ar"
    if "english" in t:
        return True, "Okay—I'll reply in English from now on.", "en"

    return False, None, None


# ----------------------------
# DB HELPERS + MIGRATIONS
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
        department TEXT DEFAULT 'GENERAL',
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        tags TEXT,
        created_by TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    if not column_exists(conn, "custom_faqs", "department"):
        c.execute("ALTER TABLE custom_faqs ADD COLUMN department TEXT DEFAULT 'GENERAL'")

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

    c.execute("UPDATE custom_faqs SET department='CONTACT' WHERE UPPER(department)='GENERAL'")

    seed_or_upgrade_user(c, "admin", "admin123", "admin")
    seed_or_upgrade_user(c, "user",  "user123",  "user")

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
# DATASET LOADING
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

    if intentcol:
        base_df["intent"] = df[intentcol].astype(str)
    else:
        base_df["intent"] = ""

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
    dept_texts = []
    dept_labels = []
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
            "If context is insufficient, ask a short follow-up question. "
            "Never request PINs or passwords. "
            f"Reply in {LANG_NAME.get(out_lang, 'English')}."
        )
        user = (
            f"Customer question (English): {query_en}\n\n"
            "Context (FAQ snippets, English):\n" + "\n---\n".join(context_snippets)
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
        return ("AI fallback is unavailable right now. I’ll answer using SACCO FAQs.", 0.0, "fallback")

def generate_reply(user_message_raw, query_en, username, dept, out_lang):
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
# UI PAGES
# ----------------------------
def render_brand_header(title, subtitle, badge_text="Baraka"):
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bank-card" style="display:flex;justify-content:space-between;gap:14px;align-items:flex-start;">
      <div style="min-width:60%;">
        <h2 style="margin:0;line-height:1.1;">{title}</h2>
        <div class="small-muted" style="margin-top:8px;">{subtitle}</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;align-items:flex-end;">
        <span class="badge">{badge_text}</span>
        <span class="badge badge-ok">Modern • Light UI</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def login_page():
    render_brand_header(
        title=f"{APP_NAME}",
        subtitle="Sign in to submit complaints, get routed to the right department, and chat with a support bot.",
        badge_text="Kenya-ready"
    )

    left, right = st.columns([1.2, 1.0], vertical_alignment="top")

    with left:
        st.markdown("""
        <div class="bank-card">
          <h3 style="margin:0;">Welcome back ✨</h3>
          <div class="small-muted" style="margin-top:8px;">
            Demo accounts:<br>
            <b>admin / admin123</b><br>
            <b>user / user123</b>
          </div>
          <hr/>
          <div class="small-muted">
            Tip: You can chat in <b>Swahili, Amharic, Somali</b> or <b>Arabic</b>.
            Baraka will auto-detect and reply in the same language.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### Sign in")
        with st.form("login"):
            username = st.text_input("Username", placeholder="e.g., admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login")

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

    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY is not set. Multilingual auto-translation and AI fallback need the API key.")

def user_home_page():
    user = st.session_state.user
    render_brand_header(
        title=f"Hello, {user}",
        subtitle="Choose what you want to do today.",
        badge_text="Customer Portal"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Submit a Complaint / Inquiry")
        st.markdown("<div class='small-muted'>Your complaint will be routed automatically.</div>", unsafe_allow_html=True)
        if st.button("Open Complaint Form"):
            st.session_state.page = "complaint"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### 💬 Chat with Baraka")
        st.markdown("<div class='small-muted'>Ask questions and get instant help.</div>", unsafe_allow_html=True)
        if st.button("Open Chat"):
            st.session_state.page = "chat"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        for k in ["user", "role", "page", "messages", "active_ticket", "preferred_lang"]:
            st.session_state.pop(k, None)
        st.rerun()

def complaint_page():
    user = st.session_state.user
    render_brand_header(
        title="Submit Complaint / Inquiry",
        subtitle="Describe your issue clearly — Baraka will route it automatically and reply instantly.",
        badge_text="Smart Routing"
    )

    with st.form("complaint_form", clear_on_submit=True):
        text = st.text_area("Complaint / Inquiry", height=140,
                            placeholder="e.g., ATM debited me but I got no cash...")
        priority = st.selectbox("Priority", ["Normal", "High", "Urgent"])
        submitted = st.form_submit_button("Submit")

    if submitted and text.strip():
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is not set. Please set your API key to enable multilingual support.")
            return

        detected_lang, text_en = detect_and_translate_to_english(text.strip())
        out_lang = st.session_state.get("preferred_lang") or detected_lang

        dept, score, method = route_department(text_en)
        summary = text.strip()[:180] + ("..." if len(text.strip()) > 180 else "")
        ticket_id = create_complaint(user, text.strip(), dept, priority=priority, summary=summary)
        st.session_state.active_ticket = ticket_id

        st.success("Complaint submitted successfully.")
        st.markdown(f"""
        <div class="ticket">
          <div><b>Ticket #:</b> {ticket_id}</div>
          <div><b>Routed Department:</b> {dept} — {DEPT_LABELS.get(dept, dept)}</div>
          <div><b>Routing Confidence:</b> {score:.2f} ({method})</div>
          <div class="small-muted" style="margin-top:6px;">
            An agent can review your case. Baraka will assist below.
          </div>
        </div>
        """, unsafe_allow_html=True)

        ans, source, sc = generate_reply(text.strip(), text_en, user, dept, out_lang)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Baraka’s Instant Reply")
        st.markdown(f"<div class='bank-card'>{ans}</div>", unsafe_allow_html=True)
        st.caption(f"Source: {source} | Similarity: {sc:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Home"):
            st.session_state.page = "home"; st.rerun()
    with c2:
        if st.button("Go to Chat"):
            st.session_state.page = "chat"; st.rerun()

def chat_page():
    user = st.session_state.user
    render_brand_header(
        title="Chat with Baraka",
        subtitle="Ask anything — Baraka will auto-detect your language and reply accordingly.",
        badge_text="Live Support"
    )

    # Seed welcome message
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hi! 👋 I’m Baraka.\n"
                "Ask about accounts, cards, loans, ATM issues, transfers, fees — "
                "or submit a complaint and I’ll route it to the right department.\n\n"
                "You can write in Kiswahili, Amharic, Somali, Arabic, or English."
            )
        })

    if st.session_state.get("active_ticket"):
        st.caption(f"Active ticket: #{st.session_state.active_ticket}")

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        cls = "user" if m["role"] == "user" else "bot"
        st.markdown(f'<div class="bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            q = st.text_input("Type your question...", key="user_input", placeholder="e.g., nataka kukopa shilingi elfu kumi")
        with col2:
            send = st.form_submit_button("Send")
    st.markdown("</div>", unsafe_allow_html=True)

    if send and q.strip():
        handled, msg, forced = handle_language_command(q.strip())
        if handled:
            st.session_state.preferred_lang = forced
            st.session_state.messages.append({"role": "user", "content": q.strip()})
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": q.strip()})

        if not OPENAI_API_KEY:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "OPENAI_API_KEY is not set. Please set your API key to enable translation + multilingual support."
            })
            st.rerun()

        detected_lang, q_en = detect_and_translate_to_english(q.strip())
        out_lang = st.session_state.get("preferred_lang") or detected_lang

        dept, dscore, dmethod = route_department(q_en)
        ans, source, score = generate_reply(q.strip(), q_en, user, dept, out_lang)

        if source == "custom":
            footer = f" Dept FAQ ({dept})"
        elif source == "base":
            footer = f" Base dataset ({dept})"
        elif source == "openai":
            footer = f" AI fallback ({dept})"
        else:
            footer = f" Low confidence ({dept})"

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"{ans}\n\n"
                f"— Department: {dept} ({DEPT_LABELS.get(dept)}) | "
                f"Routing: {dscore:.2f} ({dmethod})\n"
                f"— Source: {footer}"
            )
        })
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Home"):
            st.session_state.page = "home"; st.rerun()
    with c2:
        if st.button("Logout"):
            for k in ["user", "role", "page", "messages", "active_ticket", "preferred_lang"]:
                st.session_state.pop(k, None)
            st.rerun()

def admin_page():
    admin = st.session_state.user
    render_brand_header(
        title="Admin Console",
        subtitle=f"Logged in as {admin}. Manage department FAQs, complaints, and logs.",
        badge_text="Admin"
    )

    tabs = st.tabs([
        "➕ Add Dept FAQ",
        "🛠️ Manage FAQs",
        "📥 Complaint Queue",
        "📊 Chat Logs"
    ])

    with tabs[0]:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### Create a new FAQ")
        dept = st.selectbox("Department", DEPARTMENTS)
        q = st.text_area("Customer Question", placeholder="e.g., How do I reset my PIN?")
        a = st.text_area("Official Answer", placeholder="Provide the official guidance…")
        tags = st.text_input("Tags / Keywords (comma separated)", placeholder="pin, reset, password")
        save = st.button("Save FAQ")
        st.markdown("</div>", unsafe_allow_html=True)

        if save:
            if q.strip() and a.strip():
                add_custom_faq(dept, q.strip(), a.strip(), tags.strip(), admin)
                st.success("FAQ added."); st.rerun()
            else:
                st.error("Question and Answer are required.")

    with tabs[1]:
        filter_dept = st.selectbox("Filter by Department", ["ALL"] + DEPARTMENTS)
        df = fetch_custom_faqs(filter_dept)

        if df.empty:
            st.info("No custom FAQs yet.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown("### Existing FAQs")
            st.dataframe(df[["id","department","question","answer","tags","updated_at"]], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown("### Edit / Delete FAQ")
            edit_id = st.selectbox("Select FAQ ID", df["id"].tolist())
            row = df[df["id"] == edit_id].iloc[0]

            new_dept = st.selectbox(
                "Department",
                DEPARTMENTS,
                index=safe_index(DEPARTMENTS, row.get("department"), fallback_value="CONTACT")
            )
            new_q = st.text_area("Question", value=row["question"])
            new_a = st.text_area("Answer", value=row["answer"])
            new_tags = st.text_input("Tags", value=row.get("tags","") or "")

            colE, colD = st.columns(2)
            with colE:
                if st.button("Update FAQ"):
                    update_custom_faq(edit_id, new_dept, new_q.strip(), new_a.strip(), new_tags.strip())
                    st.success("FAQ updated."); st.rerun()
            with colD:
                if st.button("Delete FAQ"):
                    delete_custom_faq(edit_id)
                    st.warning("FAQ deleted."); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        STATUS_OPTS = ["Open", "In Review", "Resolved", "Rejected"]
        PRIORITY_OPTS = ["Normal", "High", "Urgent"]

        colA, colB = st.columns(2)
        with colA:
            dept_filter = st.selectbox("Department Queue", ["ALL"] + DEPARTMENTS)
        with colB:
            status_filter = st.selectbox("Status", ["ALL"] + STATUS_OPTS)

        comp_df = fetch_complaints(dept_filter, status_filter)

        if comp_df.empty:
            st.info("No complaints found.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown("### Complaint Queue")
            st.dataframe(
                comp_df[["id","username","department","priority","status","summary","created_at"]],
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            cid = st.selectbox("Open Complaint Ticket", comp_df["id"].tolist())
            row = comp_df[comp_df["id"] == cid].iloc[0]

            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown(f"### Ticket #{cid}")
            st.write(f"**Customer:** {row['username']}")
            st.write(f"**Department:** {row['department']} — {DEPT_LABELS.get(row['department'], row['department'])}")
            st.write(f"**Priority:** {row['priority']}")
            st.write(f"**Status:** {row['status']}")
            st.write("**Complaint Text:**")
            st.write(row["text"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown("### Update Ticket")
            new_status = st.selectbox(
                "Status",
                STATUS_OPTS,
                index=safe_index(STATUS_OPTS, row.get("status"), fallback_value="Open")
            )
            new_priority = st.selectbox(
                "Priority",
                PRIORITY_OPTS,
                index=safe_index(PRIORITY_OPTS, row.get("priority"), fallback_value="Normal")
            )
            notes = st.text_area("Internal Notes", value=row.get("internal_notes") or "", height=120)

            if st.button("Save Updates"):
                update_complaint(cid, status=new_status, priority=new_priority, internal_notes=notes)
                st.success("Complaint updated."); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        conn = get_conn()
        logs = pd.read_sql_query("SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 800", conn)
        conn.close()

        if logs.empty:
            st.info("No chats yet.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown("### Chat Logs")
            st.dataframe(logs, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Logout"):
        for k in ["user","role","page","messages","active_ticket","preferred_lang"]:
            st.session_state.pop(k, None)
        st.rerun()


# ----------------------------
# APP ROUTER
# ----------------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide")
    init_db()

    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "preferred_lang" not in st.session_state:
        st.session_state.preferred_lang = None

    page = st.session_state.page
    role = st.session_state.get("role")

    if page == "login":
        login_page()
    else:
        if role == "admin":
            admin_page()
        else:
            if page == "home":
                user_home_page()
            elif page == "complaint":
                complaint_page()
            else:
                chat_page()

if __name__ == "__main__":
    main()
