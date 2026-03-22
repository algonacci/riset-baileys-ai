import os
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

MODEL = os.getenv("LLM_MODEL_NAME")

# --- MEMORY CONFIG ---
# Di prod, ganti pake Redis biar gak amnesia pas restart
chat_sessions = {} 
MAX_HISTORY_LEN = 10 # Titik di mana kita bakal compact history

def summarize_history(history):
    """Fungsi buat mengecilkan history jadi ringkasan padat"""
    print("🧹 Compacting history...")
    
    # Prompt khusus buat meringkas
    summary_prompt = (
        "Ringkas percakapan di atas menjadi satu paragraf pendek. "
        "Pastikan poin penting seperti nama user, produk/jasa yang ditanyakan, "
        "dan status terakhir tetap terjaga."
    )
    
    temp_messages = history + [{"role": "user", "content": summary_prompt}]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=temp_messages,
        temperature=0.3 # Low temp biar ringkasannya akurat (gak halu)
    )
    
    summary = response.choices[0].message.content
    return f"Summary of previous conversation: {summary}"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    jid = data.get("jid")
    user_message = data.get("message")

    if not jid or not user_message:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    # 1. Initialize session if new
    if jid not in chat_sessions:
        chat_sessions[jid] = [
            {"role": "system", "content": "Kamu adalah admin CRM Jasa yang solutif. Balas dengan singkat dan santai."}
        ]

    # 2. Get current session
    history = chat_sessions[jid]
    
    # Tambahkan pesan user ke history
    history.append({"role": "user", "content": user_message})

    try:
        # 3. Panggil LLM dengan full history (atau summarized history)
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.7
        )

        ai_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})

        # --- 4. AUTO-COMPACTION LOGIC ---
        # Jika history kepanjangan (misal > 10 pesan assistant/user)
        if len(history) > MAX_HISTORY_LEN:
            summary_text = summarize_history(history)
            
            # Reset history: Sisipkan System Prompt + Summary saja
            chat_sessions[jid] = [
                {"role": "system", "content": "Kamu adalah admin CRM Jasa yang solutif. Balas dengan singkat dan santai."},
                {"role": "system", "content": summary_text}
            ]
            print(f"✅ History for {jid} compacted.")

        return jsonify({
            "status": "success",
            "reply": ai_reply
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": "LLM Failure"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)