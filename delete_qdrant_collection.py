# /// script
# dependencies = ["qdrant-client"]
# ///
from qdrant_client import QdrantClient

REMOTE_IP = "100.88.77.71"
COLLECTION_NAME = "omniflow_kb"

client = QdrantClient(host=REMOTE_IP, port=6333, check_compatibility=False)

print(f"🧨 Menghapus koleksi {COLLECTION_NAME} di {REMOTE_IP}...")
client.delete_collection(collection_name=COLLECTION_NAME)
print("✅ Terhapus! Sekarang lo bisa jalanin ingest.py lagi.")