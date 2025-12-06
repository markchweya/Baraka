# Baraka ✨ — Multilingual Banking & SACCO Support Assistant (Streamlit)

Baraka is a modern, light-themed **multilingual customer support assistant** for Kenyan retail banking and SACCOs. It combines:
- **FAQ retrieval** (custom FAQs + a base banking dataset),
- **automatic complaint routing** to departments (ATM, Cards, Loans, Transfers, etc.),
- an **Admin Console** for managing FAQs and complaint tickets,
- **multilingual chat** with optional language locking (e.g. “reply in Kiswahili”).

Baraka is designed to **avoid sensitive credential collection** (no PINs, OTPs, passwords).

---

## Features

### ✅ Multilingual Chat (with language lock)
- Auto-detects language (when OpenAI is enabled)
- Replies in the user’s language
- You can force language in chat:  
  **`reply in Kiswahili`**, **`reply in Somali`**, etc.

### ✅ Complaint / Inquiry Ticketing
- Customers submit complaints via the **Complaint** page
- Baraka routes to a department automatically using:
  - keyword rules, then
  - TF-IDF similarity routing
- Saves each complaint into a local SQLite database (`bankbot.db`)

### ✅ FAQ Retrieval (Custom First → Base Dataset → AI fallback)
Baraka answers in this order:
1. **Custom FAQs** (admin-created, per department)
2. **Base banking dataset** (Bitext retail banking dataset from HF)
3. **OpenAI fallback** (only if enabled and only using top retrieved context)

### ✅ Admin Console (role-based)
Admins can:
- Add / edit / delete FAQs
- View and manage complaint tickets (status, priority, internal notes)
- View chat logs (source + similarity scores)

### ✅ Modern Light UI
Includes a glassy, modern, light UI with BaseWeb styling overrides (selectboxes, menus, inputs).

---

## Tech Stack
- **Streamlit** UI
- **SQLite** persistence
- **Pandas** for data handling
- **scikit-learn** TF-IDF retrieval + cosine similarity
- **Hugging Face dataset (parquet)** as base FAQ dataset
- **OpenAI API** for translation + fallback answering

---

## Project Structure

- `app.py` — the entire Streamlit app
- `bankbot.db` — auto-created SQLite database (users, FAQs, complaints, chat logs)

---

## Requirements

Install dependencies:

```bash
pip install streamlit openai pandas scikit-learn pyarrow huggingface-hub fsspec
