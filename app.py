# app.py — AI Complaint Management + Department Routing + Banking Chatbot (Streamlit)
#
# Demo accounts (auto-created / auto-upgraded):
#   admin / admin123
#   user  / user123
#
# Deployment-safe:
# - No bcrypt (uses stdlib PBKDF2 hashing)
# - No datasets lib (loads Bitext parquet via pandas)
#
# requirements.txt (repo root):
# streamlit
# openai
# pandas
# scikit-learn
# pyarrow
# huggingface-hub
# fsspec

import os
import re
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
APP_NAME = "AI Bank Complaint & Support Bot"
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
        "recent transactions", "bank statement", "account verification", "kyc update"
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
        "loan status", "interest rate"
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
    "LOAN": ["loan", "mortgage", "repayment", "interest"],
    "TRANSFER": ["transfer", "send money", "reversal", "pending"],
    "PASSWORD": ["password", "pin reset", "forgot"],
    "FEES": ["fees", "charges", "annual fee"],
    "FIND": ["find atm", "branch", "nearest atm"],
    "CONTACT": ["agent", "customer care", "call center"],
    "ACCOUNT": ["account", "statement", "transactions", "close account"]
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
# STYLING (restyled chat: no huge empty space)
# ----------------------------
BANK_CSS = """
<style>
:root{
  --bg:#0b1020;
  --card:#111835;
  --muted:#9aa4c7;
  --accent:#4cc9f0;
  --accent2:#80ffdb;
  --text:#e9edff;
  --ok:#59f9b7;
  --warn:#ffd166;
}
.stApp{
  background:
    radial-gradient(1200px 700px at 10% -10%, #1a2250 0%, transparent 60%),
    radial-gradient(1200px 700px at 110% 0%, #10234a 0%, transparent 55%),
    var(--bg) !important;
  color: var(--text);
}
.bank-card{
  background: linear-gradient(180deg, #121a3a 0%, #0e1530 100%);
  border:1px solid rgba(255,255,255,0.06);
  border-radius:18px; padding:16px 18px;
  box-shadow:0 10px 30px rgba(0,0,0,0.35);
}
.small-muted{ color:var(--muted); font-size:0.9rem; }
.badge{
  display:inline-flex; gap:6px; align-items:center;
  background:rgba(76,201,240,0.12); color:var(--accent);
  border:1px solid rgba(76,201,240,0.35);
  padding:4px 10px; border-radius:999px;
  font-size:12px; font-weight:600;
}
.badge-ok{ background:rgba(89,249,183,0.12); color:var(--ok); border-color:rgba(89,249,183,0.35); }
.badge-warn{ background:rgba(255,209,102,0.12); color:var(--warn); border-color:rgba(255,209,102,0.35); }

/* ✅ Chat box now AUTO-SIZES (no huge blank space) */
.chat-wrap{
  background: rgba(17,24,53,0.55);
  border:1px solid rgba(255,255,255,0.05);
  border-radius:18px;
  padding:12px;
  max-height: 60vh;     /* scroll only when needed */
  min-height: 180px;    /* small polite height when few messages */
  overflow-y:auto;
}

/* tighter spacing for bubbles */
.bubble{
  max-width: 78%;
  padding:10px 12px; border-radius:14px; margin:4px 0;
  line-height:1.5; font-size:0.98rem;
  white-space:pre-wrap;
}
.user{
  margin-left:auto; background:rgba(76,201,240,0.16);
  border:1px solid rgba(76,201,240,0.35);
}
.bot{
  margin-right:auto; background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.08);
}
.ticket{
  background:#0f1632; border:1px dashed rgba(255,255,255,0.12);
  border-radius:14px; padding:10px 12px; margin-top:8px;
}

/* input area styling */
.input-card{
  background: rgba(17,24,53,0.7);
  border:1px solid rgba(255,255,255,0.06);
  border-radius:16px;
  padding:10px 12px;
}

.stButton button{
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
  color:#071021; font-weight:700; border:none; border-radius:12px;
  padding:0.6rem 1rem;
}
.stTextInput input, .stTextArea textarea, .stSelectbox select{
  background:#0f1632 !important; color:var(--text) !important;
  border:1px solid rgba(255,255,255,0.12) !important;
  border-radius:12px !important;
}
hr{ border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0; }
</style>
"""


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

    # clean legacy GENERAL
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
# DATASET LOADING (no datasets lib)
# ----------------------------
@st.cache_resource
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


@st.cache_resource
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

@st.cache_resource
def build_dept_router():
    dept_texts = []
    dept_labels = []
    for d, samples in DEPT_TRAIN.items():
        for s in samples:
            dept_texts.append(s)
            dept_labels.append(d)
    vec, X = build_vector_index(dept_texts)
    return vec, X, dept_labels

def route_department(text):
    t = normalize(text)

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

def retrieve_best(query, faq_df, vec, X, topk=TOPK):
    qn = normalize(query)
    qv = vec.transform([qn])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:topk]
    results = faq_df.iloc[idxs].copy()
    results["score"] = sims[idxs]
    return results

def answer_from_custom_first(query, dept):
    custom_df = fetch_custom_faqs(dept)
    if custom_df.empty:
        return None

    questions = [q for q in custom_df["question"].astype(str).tolist() if q.strip()]
    if not questions:
        return None

    vec_c, X_c = build_vector_index(questions)
    res = retrieve_best(query, custom_df, vec_c, X_c, topk=TOPK)
    best = res.iloc[0]
    if float(best["score"]) >= SIM_THRESHOLD_CUSTOM:
        return best["answer"], float(best["score"]), "custom"
    return None

def answer_from_base(query, dept, base_df):
    base_dept = base_df[base_df["category"] == dept]
    if base_dept.empty:
        base_dept = base_df

    vec_d, X_d = build_vector_index(base_dept["question"].tolist())
    res = retrieve_best(query, base_dept, vec_d, X_d, topk=TOPK)
    best = res.iloc[0]
    if float(best["score"]) >= SIM_THRESHOLD_BASE:
        return best["answer"], float(best["score"]), "base"
    return None

def openai_fallback(query, context_snippets):
    if not OPENAI_API_KEY:
        return (
            "I’m not fully confident yet. Please rephrase or add more detail.",
            0.0, "fallback"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system = (
            "You are a helpful Kenyan retail-banking/SACCO support assistant. "
            "Answer ONLY using the provided context. "
            "If context is insufficient, ask a short follow-up question. "
            "Never request PINs or passwords."
        )

        user = (
            f"Customer question: {query}\n\n"
            "Context (FAQ snippets):\n" + "\n---\n".join(context_snippets)
        )

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.2
        )

        return resp.output_text.strip(), 0.0, "openai"

    except Exception:
        return (
            "AI fallback is unavailable right now. I’ll answer using SACCO FAQs.",
            0.0, "fallback"
        )

def generate_reply(query, username, dept):
    base_df = load_base_dataset()

    custom_hit = answer_from_custom_first(query, dept)
    if custom_hit:
        ans, score, source = custom_hit
        log_chat(username, query, ans, source, score, dept)
        return ans, source, score

    base_hit = answer_from_base(query, dept, base_df)
    if base_hit:
        ans, score, source = base_hit
        log_chat(username, query, ans, source, score, dept)
        return ans, source, score

    vec_b, X_b = build_vector_index(base_df["question"].tolist())
    top_base = retrieve_best(query, base_df, vec_b, X_b, topk=TOPK)
    snippets = [f"Q: {r.question}\nA: {r.answer}" for r in top_base.itertuples()]
    ans, score, source = openai_fallback(query, snippets)
    log_chat(username, query, ans, source, score, dept)
    return ans, source, score


# ----------------------------
# UI PAGES
# ----------------------------
def login_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bank-card">
      <h2 style="margin:0;">{APP_NAME}</h2>
      <p class="small-muted">
        Submit complaints, get routed to the right department, and receive instant SACCO/bank support.
      </p>
      <span class="badge">Kenya-ready</span>
      <div class="small-muted" style="margin-top:8px;">
        Demo accounts: <b>admin/admin123</b> or <b>user/user123</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
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


def user_home_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    user = st.session_state.user

    st.markdown(f"""
    <div class="bank-card" style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h3 style="margin:0;">Welcome, {user}</h3>
        <div class="small-muted">Choose what you want to do today.</div>
      </div>
      <div><span class="badge">Customer Portal</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Submit a Complaint / Inquiry")
        st.markdown("<div class='small-muted'>Your complaint will be routed automatically.</div>", unsafe_allow_html=True)
        if st.button("Open Complaint Form"):
            st.session_state.page = "complaint"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        st.markdown("### 💬 Chat with Support Bot")
        st.markdown("<div class='small-muted'>Ask questions and get instant help.</div>", unsafe_allow_html=True)
        if st.button("Open Chat"):
            st.session_state.page = "chat"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Logout"):
        for k in ["user", "role", "page", "messages", "active_ticket"]:
            st.session_state.pop(k, None)
        st.rerun()


def complaint_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    user = st.session_state.user

    st.markdown(f"""
    <div class="bank-card">
      <h3 style="margin:0;">Submit Complaint / Inquiry</h3>
      <div class="small-muted">Describe your issue clearly. We'll route it automatically.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("complaint_form", clear_on_submit=True):
        text = st.text_area("Complaint / Inquiry", height=140,
                            placeholder="e.g., ATM debited me but I got no cash...")
        priority = st.selectbox("Priority", ["Normal", "High", "Urgent"])
        submitted = st.form_submit_button("Submit")

    if submitted and text.strip():
        dept, score, method = route_department(text)
        summary = text.strip()[:180] + ("..." if len(text.strip()) > 180 else "")
        ticket_id = create_complaint(user, text.strip(), dept,
                                     priority=priority, summary=summary)
        st.session_state.active_ticket = ticket_id

        st.success("Complaint submitted successfully.")
        st.markdown(f"""
        <div class="ticket">
          <div><b>Ticket #:</b> {ticket_id}</div>
          <div><b>Routed Department:</b> {dept} — {DEPT_LABELS.get(dept, dept)}</div>
          <div><b>Routing Confidence:</b> {score:.2f} ({method})</div>
          <div class="small-muted" style="margin-top:6px;">
            An agent will review your case. The bot will assist below.
          </div>
        </div>
        """, unsafe_allow_html=True)

        ans, source, sc = generate_reply(text, user, dept)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Instant Support Bot Reply")
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
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    user = st.session_state.user

    st.markdown(f"""
    <div class="bank-card" style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h3 style="margin:0;">Customer Chat</h3>
        <div class="small-muted">Ask anything. We’ll route your message automatically.</div>
      </div>
      <div><span class="badge">Online Support</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Seed a welcome message so chat never looks empty
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hi! 👋 I’m your SACCO support bot.\n"
                "You can ask about accounts, cards, loans, ATM issues, transfers, fees, "
                "or submit a complaint and I’ll route it to the right department."
            )
        })

    if st.session_state.get("active_ticket"):
        st.caption(f"Active ticket: #{st.session_state.active_ticket}")

    # Chat messages (auto height, no huge blank space)
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        cls = "user" if m["role"] == "user" else "bot"
        st.markdown(f'<div class="bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            q = st.text_input("Type your question...", key="user_input")
        with col2:
            send = st.form_submit_button("Send")

    st.markdown("</div>", unsafe_allow_html=True)

    if send and q.strip():
        dept, dscore, dmethod = route_department(q)

        st.session_state.messages.append({"role": "user", "content": q})
        ans, source, score = generate_reply(q, user, dept)

        if source == "custom":
            footer = f"✅ Dept FAQ ({dept})"
        elif source == "base":
            footer = f"📚 Base dataset ({dept})"
        elif source == "openai":
            footer = f"🤖 AI fallback ({dept})"
        else:
            footer = f"ℹ️ Low confidence ({dept})"

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
            for k in ["user", "role", "page", "messages", "active_ticket"]:
                st.session_state.pop(k, None)
            st.rerun()


def admin_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    admin = st.session_state.user

    st.markdown(f"""
    <div class="bank-card">
      <h3 style="margin:0;">Admin Console</h3>
      <div class="small-muted">
        Logged in as <b>{admin}</b>. Manage department FAQs, complaints, and logs.
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "➕ Add Dept FAQ",
        "🛠️ Manage FAQs",
        "📥 Complaint Queue",
        "📊 Chat Logs"
    ])

    with tabs[0]:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        dept = st.selectbox("Department", DEPARTMENTS)
        q = st.text_area("Customer Question")
        a = st.text_area("Official Answer")
        tags = st.text_input("Tags / Keywords (comma separated)")
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
            st.dataframe(df[["id","department","question","answer","tags","updated_at"]],
                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            edit_id = st.selectbox("Select FAQ ID to edit/delete", df["id"].tolist())
            row = df[df["id"] == edit_id].iloc[0]

            new_dept = st.selectbox(
                "Edit Department",
                DEPARTMENTS,
                index=safe_index(DEPARTMENTS, row.get("department"), fallback_value="CONTACT")
            )
            new_q = st.text_area("Edit Question", value=row["question"])
            new_a = st.text_area("Edit Answer", value=row["answer"])
            new_tags = st.text_input("Edit Tags", value=row.get("tags","") or "")

            colE, colD = st.columns(2)
            with colE:
                if st.button("Update FAQ"):
                    update_custom_faq(edit_id, new_dept, new_q.strip(), new_a.strip(), new_tags.strip())
                    st.success("FAQ updated."); st.rerun()
            with colD:
                if st.button("Delete FAQ"):
                    delete_custom_faq(edit_id)
                    st.warning("FAQ deleted."); st.rerun()

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
            st.dataframe(
                comp_df[["id","username","department","priority","status","summary","created_at"]],
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

            cid = st.selectbox("Open Complaint Ticket", comp_df["id"].tolist())
            row = comp_df[comp_df["id"] == cid].iloc[0]

            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.markdown(f"**Ticket #{cid}**")
            st.write(f"**Customer:** {row['username']}")
            st.write(f"**Department:** {row['department']} — {DEPT_LABELS.get(row['department'], row['department'])}")
            st.write(f"**Priority:** {row['priority']}")
            st.write(f"**Status:** {row['status']}")
            st.write("**Complaint Text:**")
            st.write(row["text"])
            st.markdown("</div>", unsafe_allow_html=True)

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

            if st.button("Save Complaint Updates"):
                update_complaint(cid, status=new_status, priority=new_priority, internal_notes=notes)
                st.success("Complaint updated."); st.rerun()

    with tabs[3]:
        conn = get_conn()
        logs = pd.read_sql_query(
            "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 800",
            conn
        )
        conn.close()
        if logs.empty:
            st.info("No chats yet.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.dataframe(logs, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Logout"):
        for k in ["user","role","page","messages","active_ticket"]:
            st.session_state.pop(k, None)
        st.rerun()


# ----------------------------
# APP ROUTER
# ----------------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🏦", layout="wide")
    init_db()

    if "page" not in st.session_state:
        st.session_state.page = "login"

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
