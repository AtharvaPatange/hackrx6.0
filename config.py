import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "hackrx-jina-index"

# --- Model Settings (Using Groq with LLaMA + Jina embeddings) ---
LLM_MODEL = "llama-3.3-70b-versatile"  # Latest LLaMA model on Groq
EMBEDDING_MODEL = "jina-embeddings-v2-base-en"  # Jina embedding model (768 dimensions)