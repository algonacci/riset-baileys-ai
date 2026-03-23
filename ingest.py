# /// script
# dependencies = [
#   "qdrant-client",
#   "langchain-text-splitters",
#   "openai",
#   "python-dotenv",
# ]
# ///

import os
import glob
import uuid
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Load .env buat ambil OPENAI_API_KEY
load_dotenv()

# --- CONFIG ---
REMOTE_IP = "100.88.77.71"
COLLECTION_NAME = "omniflow_kb"
# OpenAI text-embedding-3-small (1536 dim) - Paling cost-efficient
EMBED_MODEL = "text-embedding-3-small" 

# Initialize Clients
# check_compatibility=False buat bungkam warning versi Qdrant tadi
qdrant_client = QdrantClient(host=REMOTE_IP, port=6333, check_compatibility=False)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def setup_collection():
    try:
        collections = qdrant_client.get_collections().collections
        # Jika collection sudah ada tapi dimensi beda, sebaiknya hapus/recreate manual.
        # Di sini kita cek saja keberadaannya.
        if not any(c.name == COLLECTION_NAME for c in collections):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created (1536 dim) on {REMOTE_IP}.")
        else:
            print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        exit(1)

def get_embedding(text):
    """Generate embedding via OpenAI API."""
    text = text.replace("\n", " ") # Clean up text
    response = openai_client.embeddings.create(input=[text], model=EMBED_MODEL)
    return response.data[0].embedding

def run_ingestion():
    setup_collection()

    # Strategi Split agar AI paham konteks tiap bagian Markdown
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    files = glob.glob("docs/*.md")
    if not files:
        print("⚠️ Folder 'docs/' kosong, Bang. Pastikan ada file .md di sana.")
        return

    print(f"📂 Memproses {len(files)} file ke Qdrant Remote via OpenAI...")

    for file_path in files:
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = splitter.split_text(raw_text)
        points = []
        
        for chunk in chunks:
            # Bangun konteks hirarki (H1 > H2 > H3)
            header_path = " > ".join([v for k, v in chunk.metadata.items()])
            contextual_text = f"File: {file_name}\nContext: {header_path}\n\nContent: {chunk.page_content}"

            # Ambil vector dari OpenAI
            vector = get_embedding(contextual_text)
            
            payload = {
                "file_name": file_name,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "source_context": header_path
            }

            points.append(PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector, 
                payload=payload
            ))

        # Push ke Qdrant Remote
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ {file_name} synced ({len(chunks)} chunks).")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Mana OPENAI_API_KEY nya di .env? Siapin dulu, Bang.")
    else:
        run_ingestion()
        print(f"\n✨ Selesai! Omniflow Knowledge Base sekarang udah 'live' di Qdrant.")