# app.py — Banking Support Chatbot with Admin Console (Streamlit)
# Demo accounts (auto-created / auto-upgraded on first run):
#   admin / admin123
#   user  / user123
#
# Deployment-safe:
# - No bcrypt (uses Python stdlib PBKDF2 hashing)
# - No datasets lib (loads Bitext parquet via pandas)
#
# Fixes included:
# 1) TF-IDF "empty vocabulary" for tiny custom FAQ sets (dynamic min_df).
# 2) Streamlit input reset bug (form clear_on_submit).
# 3) Auto-upgrade old stored hashes to PBKDF2 so login works even if DB existed earlier.
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
APP_NAME = "SACCO/BANK Support Bot"
DB_PATH = "bankbot.db"

# HuggingFace parquet (Bitext retail banking)
BASE_PARQUET_URL = (
    "hf://datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/"
    "bitext-retail-banking-llm-chatbot-training-dataset.parquet"
)

TOPK = 3
SIM_THRESHOLD_CUSTOM = 0.40
SIM_THRESHOLD_BASE = 0.35

# ✅ Environment variable only (no st.secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----------------------------
# PASSWORD HASHING (stdlib PBKDF2)
# ----------------------------
def hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.b64encode(salt + key).decode()

def is_pbkdf2_hash(stored) -> bool:
    """True if stored looks like our PBKDF2 base64 hash."""
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
        return len(raw) >= 48  # 16 salt + 32 key typical
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
# STYLING (banking UI feel)
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
.chat-wrap{
  background: rgba(17,24,53,0.55);
  border:1px solid rgba(255,255,255,0.05);
  border-radius:18px;
  padding:12px; height: 62vh; overflow-y:auto;
}
.bubble{
  max-width: 78%;
  padding:10px 12px; border-radius:14px; margin:6px 0;
  line-height:1.45; font-size:0.98rem;
}
.user{
  margin-left:auto; background:rgba(76,201,240,0.16);
  border:1px solid rgba(76,201,240,0.35);
}
.bot{
  margin-right:auto; background:rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.08);
}
hr{ border:none; border-top:1px solid rgba(255,255,255,0.08); }
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
</style>
"""


# ----------------------------
# DB HELPERS
# ----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def seed_or_upgrade_user(c, username, password, role):
    """
    If user missing -> insert PBKDF2 hash.
    If user exists but hash is old/invalid -> overwrite with PBKDF2 hash.
    """
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
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        tags TEXT,
        created_by TEXT,
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
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # ✅ Auto-create / auto-upgrade BOTH accounts
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

def fetch_custom_faqs():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM custom_faqs ORDER BY updated_at DESC", conn)
    conn.close()
    return df

def add_custom_faq(q, a, tags, created_by):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO custom_faqs(question,answer,tags,created_by)
    VALUES(?,?,?,?)
    """, (q, a, tags, created_by))
    conn.commit()
    conn.close()

def update_custom_faq(fid, q, a, tags):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE custom_faqs
    SET question=?, answer=?, tags=?, updated_at=CURRENT_TIMESTAMP
    WHERE id=?
    """, (q, a, tags, fid))
    conn.commit()
    conn.close()

def delete_custom_faq(fid):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM custom_faqs WHERE id=?", (fid,))
    conn.commit()
    conn.close()

def log_chat(username, user_message, bot_reply, source, score):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO chat_logs(username,user_message,bot_reply,source,score)
    VALUES(?,?,?,?,?)
    """, (username, user_message, bot_reply, source, score))
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

    if not qcol or not acol:
        raise ValueError(f"Could not detect question/answer columns. Found: {df.columns}")

    base_df = df[[qcol, acol]].rename(columns={qcol: "question", acol: "answer"})
    base_df["question"] = base_df["question"].astype(str)
    base_df["answer"] = base_df["answer"].astype(str)
    base_df.dropna(inplace=True)
    base_df.reset_index(drop=True, inplace=True)

    return base_df


# ✅ dynamic min_df so tiny custom FAQ sets don't crash
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
# NLP / RETRIEVAL
# ----------------------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def retrieve_best(query, faq_df, vec, X, topk=TOPK):
    qn = normalize(query)
    qv = vec.transform([qn])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:topk]
    results = faq_df.iloc[idxs].copy()
    results["score"] = sims[idxs]
    return results

def answer_from_custom_first(query, custom_df):
    if custom_df.empty:
        return None

    questions = custom_df["question"].astype(str).tolist()
    questions = [q for q in questions if q.strip()]
    if len(questions) == 0:
        return None

    vec_c, X_c = build_vector_index(questions)
    res = retrieve_best(query, custom_df, vec_c, X_c, topk=TOPK)
    best = res.iloc[0]

    if float(best["score"]) >= SIM_THRESHOLD_CUSTOM:
        return best["answer"], float(best["score"]), "custom"
    return None

def answer_from_base(query, base_df, vec_b, X_b):
    res = retrieve_best(query, base_df, vec_b, X_b, topk=TOPK)
    best = res.iloc[0]
    if float(best["score"]) >= SIM_THRESHOLD_BASE:
        return best["answer"], float(best["score"]), "base"
    return None

def openai_fallback(query, context_snippets):
    if not OPENAI_API_KEY:
        return (
            "I’m not confident yet. Please rephrase your question or ask about "
            "accounts, loans, cards, transfers, or mobile banking."
        ), 0.0, "fallback"

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are a helpful Kenyan retail-banking/SACCO support assistant. "
        "Answer ONLY using the provided context. "
        "If the context is insufficient, ask a short follow-up question. "
        "Never request PINs or passwords. Use a professional banking tone."
    )

    user = (
        f"Customer question: {query}\n\n"
        "Context (FAQ snippets):\n"
        + "\n---\n".join(context_snippets)
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return resp.output_text.strip(), 0.0, "openai"


def generate_reply(query, username):
    base_df = load_base_dataset()
    vec_b, X_b = build_vector_index(base_df["question"].tolist())
    custom_df = fetch_custom_faqs()

    # 1) custom FAQs first
    custom_hit = answer_from_custom_first(query, custom_df)
    if custom_hit:
        ans, score, source = custom_hit
        log_chat(username, query, ans, source, score)
        return ans, source, score

    # 2) base dataset similarity
    base_hit = answer_from_base(query, base_df, vec_b, X_b)
    if base_hit:
        ans, score, source = base_hit
        log_chat(username, query, ans, source, score)
        return ans, source, score

    # 3) OpenAI fallback with top base snippets as context
    top_base = retrieve_best(query, base_df, vec_b, X_b, topk=TOPK)
    snippets = [f"Q: {r.question}\nA: {r.answer}" for r in top_base.itertuples()]
    ans, score, source = openai_fallback(query, snippets)
    log_chat(username, query, ans, source, score)
    return ans, source, score


# ----------------------------
# UI PAGES
# ----------------------------
def login_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bank-card">
      <h2 style="margin:0;">{APP_NAME}</h2>
      <p class="small-muted">Secure virtual assistant for accounts, loans, cards, transfers & mobile banking.</p>
      <span class="badge">Kenya-ready</span>
      <div class="small-muted" style="margin-top:8px;">
        Demo accounts: <b>admin/admin123</b> or <b>user/user123</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        role = verify_user(username, password)
        if role:
            st.session_state.user = username
            st.session_state.role = role
            st.session_state.page = "chat" if role == "user" else "admin"
            st.rerun()
        else:
            st.error("Invalid username/password.")


def chat_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    user = st.session_state.user

    st.markdown(f"""
    <div class="bank-card" style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <h3 style="margin:0;">Customer Chat</h3>
        <div class="small-muted">Welcome, {user}. Ask anything about SACCO/banking services.</div>
      </div>
      <div><span class="badge">Online Support</span></div>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.write("")
    with st.container():
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for m in st.session_state.messages:
            cls = "user" if m["role"] == "user" else "bot"
            st.markdown(f'<div class="bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            q = st.text_input("Type your question...", key="user_input")
        with col2:
            send = st.form_submit_button("Send")

    if send and q.strip():
        st.session_state.messages.append({"role": "user", "content": q})
        ans, source, score = generate_reply(q, user)

        if source == "custom":
            footer = "✅ From bank admin knowledge"
        elif source == "base":
            footer = "📚 From banking FAQ dataset"
        elif source == "openai":
            footer = "🤖 Drafted with AI from FAQs"
        else:
            footer = "ℹ️ Please clarify"

        st.session_state.messages.append({
            "role": "assistant",
            "content": ans + f"\n\n<div class='small-muted'>{footer}</div>"
        })
        st.rerun()

    st.write("")
    if st.button("Logout"):
        for k in ["user", "role", "page", "messages"]:
            st.session_state.pop(k, None)
        st.rerun()


def admin_page():
    st.markdown(BANK_CSS, unsafe_allow_html=True)
    admin = st.session_state.user

    st.markdown(f"""
    <div class="bank-card">
      <h3 style="margin:0;">Admin Console</h3>
      <div class="small-muted">Logged in as <b>{admin}</b>. Manage custom FAQs that override the base dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    tabs = st.tabs(["➕ Add FAQ", "🛠️ Manage FAQs", "📊 Chat Logs"])

    # ADD FAQ
    with tabs[0]:
        st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
        q = st.text_area("Customer Question (as they would ask it)")
        a = st.text_area("Best Answer (official SACCO/bank tone)")
        tags = st.text_input("Tags / Keywords (comma separated)")
        save = st.button("Save FAQ")
        st.markdown("</div>", unsafe_allow_html=True)

        if save:
            if q.strip() and a.strip():
                add_custom_faq(q.strip(), a.strip(), tags.strip(), admin)
                st.success("FAQ added. Users will get this answer immediately.")
                st.rerun()
            else:
                st.error("Question and Answer are required.")

    # MANAGE FAQs
    with tabs[1]:
        df = fetch_custom_faqs()
        if df.empty:
            st.info("No custom FAQs yet.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.dataframe(df[["id", "question", "answer", "tags", "updated_at"]],
                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            edit_id = st.selectbox("Select FAQ ID to edit/delete", df["id"].tolist())
            row = df[df["id"] == edit_id].iloc[0]

            new_q = st.text_area("Edit Question", value=row["question"])
            new_a = st.text_area("Edit Answer", value=row["answer"])
            new_tags = st.text_input("Edit Tags", value=row.get("tags", "") or "")

            colE, colD = st.columns(2)
            with colE:
                if st.button("Update FAQ"):
                    update_custom_faq(edit_id, new_q.strip(), new_a.strip(), new_tags.strip())
                    st.success("FAQ updated.")
                    st.rerun()
            with colD:
                if st.button("Delete FAQ"):
                    delete_custom_faq(edit_id)
                    st.warning("FAQ deleted.")
                    st.rerun()

    # LOGS
    with tabs[2]:
        conn = get_conn()
        logs = pd.read_sql_query(
            "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 500",
            conn
        )
        conn.close()
        if logs.empty:
            st.info("No chats yet.")
        else:
            st.markdown("<div class='bank-card'>", unsafe_allow_html=True)
            st.dataframe(logs, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if st.button("Logout"):
        for k in ["user", "role", "page", "messages"]:
            st.session_state.pop(k, None)
        st.rerun()


# ----------------------------
# APP ROUTER
# ----------------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="💳", layout="wide")
    init_db()

    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.page == "login":
        login_page()
    else:
        role = st.session_state.get("role")
        if role == "admin":
            admin_page()
        else:
            chat_page()

if __name__ == "__main__":
    main()
