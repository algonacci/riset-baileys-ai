# /// script
# dependencies = [
#   "flask",
#   "openai",
#   "qdrant-client",
#   "python-dotenv",
# ]
# ///

import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- CONFIG (Multi-Provider) ---

# 1. Client untuk LLM (Bisa DeepSeek / Gemini / OpenAI)
# Gunakan variabel LLM_MODEL_NAME sesuai .env lo
LLM_CLIENT = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") # https://api.deepseek.com/v1
)
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

# 2. Client Khusus EMBEDDING (Wajib OpenAI karena data di Qdrant pake OpenAI)
EMBED_CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") # Gunakan sk-proj-xxx lo di sini
)
EMBED_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small")

# 3. Client QDRANT
REMOTE_IP = "100.88.77.71"
qdrant_client = QdrantClient(host=REMOTE_IP, port=6333, check_compatibility=False)

chat_sessions = {}
customer_profiles = {}  # Simpan info customer: {jid: {"name": "...", "company": "...", ...}}
MAX_HISTORY_LEN = 10
CUSTOMER_DB_FILE = "customer_profiles.txt"
CHAT_SESSIONS_FILE = "chat_sessions.txt"

COLLECTION_NAME = "omniflow_kb"

def load_customer_profiles():
    """Load customer profiles dari file."""
    global customer_profiles
    if os.path.exists(CUSTOMER_DB_FILE):
        try:
            with open(CUSTOMER_DB_FILE, 'r') as f:
                customer_profiles = json.load(f)
                print(f"✅ Loaded {len(customer_profiles)} customer profiles")
        except Exception as e:
            print(f"⚠️ Error loading customer profiles: {e}")
            customer_profiles = {}
    else:
        customer_profiles = {}

def save_customer_profiles():
    """Simpan customer profiles ke file."""
    try:
        with open(CUSTOMER_DB_FILE, 'w') as f:
            json.dump(customer_profiles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving customer profiles: {e}")

def load_chat_sessions():
    """Load chat sessions dari file."""
    global chat_sessions
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, 'r') as f:
                chat_sessions = json.load(f)
                print(f"✅ Loaded {len(chat_sessions)} chat sessions")
        except Exception as e:
            print(f"⚠️ Error loading chat sessions: {e}")
            chat_sessions = {}
    else:
        chat_sessions = {}

def save_chat_sessions():
    """Simpan chat sessions ke file."""
    try:
        with open(CHAT_SESSIONS_FILE, 'w') as f:
            json.dump(chat_sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving chat sessions: {e}")

# Load data saat app start
load_customer_profiles()
load_chat_sessions()

def get_rag_context(user_query):
    """Retrieve info dari Qdrant pake API query_points yang baru."""
    try:
        # 1. Embed query user
        embed_res = EMBED_CLIENT.embeddings.create(
            input=[user_query], 
            model=EMBED_MODEL
        )
        query_vector = embed_res.data[0].embedding

        # 2. Search di Qdrant pake query_points (Modern API)
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector, # Di sini pakenya 'query'
            limit=3
        )

        # 3. Ambil data dari result.points
        # query_points balikin object yang punya list 'points'
        context_list = []
        for point in search_result.points:
            content = point.payload.get('content', '')
            context_list.append(content)

        return "\n---\n".join(context_list)
    except Exception as e:
        # Log error biar ketauan kalau ada masalah koneksi atau query
        print(f"⚠️ RAG Retrieval Error: {e}")
        return ""

def extract_customer_info(user_message, jid):
    """Extract nama customer dari pesan user dan simpan ke customer_profiles."""
    try:
        # Jika sudah ada nama, skip
        if customer_profiles.get(jid, {}).get("name"):
            return
        
        # Gunakan LLM untuk extract nama dari pesan
        extraction_prompt = f"""Analisis pesan user berikut dan extract informasi personal jika ada:
Pesan: "{user_message}"

Format JSON response (return ONLY JSON, no other text):
{{"name": "nama jika disebutkan atau null", "company": "nama company jika disebutkan atau null"}}"""
        
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3
        )
        
        import json
        response_text = response.choices[0].message.content
        # Handle markdown code blocks
        if "```" in response_text:
            response_text = response_text.split("```")[1].replace("json", "").strip()
        
        info = json.loads(response_text)
        
        if info.get("name"):
            if jid not in customer_profiles:
                customer_profiles[jid] = {}
            customer_profiles[jid]["name"] = info["name"]
            save_customer_profiles()
            print(f"✅ Stored customer name: {info['name']} for {jid}")
        
        if info.get("company"):
            if jid not in customer_profiles:
                customer_profiles[jid] = {}
            customer_profiles[jid]["company"] = info["company"]
            save_customer_profiles()
    
    except Exception as e:
        print(f"⚠️ Info extraction error: {e}")
        pass
    
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    jid = data.get("jid")
    user_message = data.get("message")

    if not jid or not user_message:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    # Extract dan simpan customer info dari pesan
    extract_customer_info(user_message, jid)

    context = get_rag_context(user_message)

    # Build system instruction dengan customer context
    customer_name = customer_profiles.get(jid, {}).get("name", "")
    customer_greeting = f"Pelanggan kami yang terhormat, {customer_name}, " if customer_name else ""
    
    system_instruction = f"""Kamu adalah Omni, AI Sales Assistant Omniflow.

PROFIL ANDA:
- Expert di solusi ERP Omniflow
- Fokus: Understand customer needs → Offer solution → Close demo/consultation
- Keep it short, professional, direct

CUSTOMER CONTEXT:
{customer_greeting if customer_greeting else ""}Nomor: {jid}

RULES PENTING:
1. DENGARKAN dengan empati: Apa pain point customer?
2. RECOMMEND solusi singkat (max 2-3 points)
3. CLOSING: Tawarkan demo atau diskusi dengan tim sales
4. JANGAN over-explain atau kasih informasi yang tidak diminta
5. HARGA: Jika ditanya, kasih range singkat (misal: "Rp 2-5 juta/bulan + setup"), lalu direct ke demo
6. Keep response pendek - 5-7 kalimat MAKSIMAL
7. Write dalam bahasa natural, conversational, tidak formal berlebihan
8. Gunakan nama customer jika sudah tahu

KNOWLEDGE BASE:
{context}

TONE: Helpful, concise, sales-focused - bukan educational lecture"""

    if jid not in chat_sessions:
        chat_sessions[jid] = [{"role": "system", "content": system_instruction}]
    else:
        chat_sessions[jid][0] = {"role": "system", "content": system_instruction}

    history = chat_sessions[jid]
    history.append({"role": "user", "content": user_message})

    try:
        # Pake LLM_CLIENT & LLM_MODEL buat chat
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=history,
            temperature=0.1
        )

        ai_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})
        
        # Simpan chat history
        save_chat_sessions()

        return jsonify({"status": "success", "reply": ai_reply})

    except Exception as e:
        print(f"🔥 Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)