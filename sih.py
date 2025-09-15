
import os
import uuid
import time
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# Modern imports
import requests
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from langdetect import detect

# ---------- Configuration from .env ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ingres-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-en")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "6"))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# Validate required environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables")
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY not set in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment variables")

print(f"🔧 Configuration loaded:")
print(f"   - LLM Model: {LLM_MODEL}")
print(f"   - Embedding Model: {EMBEDDING_MODEL}")
print(f"   - Pinecone Index: {PINECONE_INDEX_NAME}")
print(f"   - Vector Dimension: {VECTOR_DIM}")

# ---------- Initialize clients ----------
print("🚀 Initializing clients...")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client initialized")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone():
    """Initialize Pinecone index"""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"🔨 Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIM,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
        )
    else:
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists")
    return pc.Index(PINECONE_INDEX_NAME)

index = init_pinecone()
print("✅ Pinecone client initialized")

# ---------- Initialize FastAPI ----------
app = FastAPI(
    title="INGRES RAG Backend",
    description="AI-powered groundwater resources information system",
    version="2.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)
# Authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token if AUTH_TOKEN is set"""
    if API_AUTH_TOKEN and credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return True

# ---------- Data Models ----------
class IngestRequest(BaseModel):
    source_url: Optional[str] = None
    title: Optional[str] = None
    region_meta: Optional[Dict[str, Any]] = None
    text: Optional[str] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about groundwater resources")
    mode: Optional[str] = Field("text", description="Query mode: text, voice, image")
    language: Optional[str] = Field(None, description="Language hint (auto-detected if not provided)")
    k: Optional[int] = Field(DEFAULT_TOP_K, description="Number of top results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID for feedback")
    correct: bool = Field(..., description="Whether the response was correct")
    notes: Optional[str] = Field(None, description="Additional feedback notes")

# ---------- Utility Functions ----------
def detect_language_safe(text: str) -> str:
    """Safely detect language, fallback to English"""
    try:
        lang = detect(text)
        return lang if lang else "en"
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"

def get_embeddings_from_jina(texts: list):
    """Generate embeddings using Jina AI API"""
    try:
        # Clean and validate inputs
        clean_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                # Truncate very long texts to avoid API limits
                clean_text = text.strip()[:8000]  # Jina AI has input limits
                clean_texts.append(clean_text)
        
        if not clean_texts:
            raise ValueError("No valid texts provided for embedding")
        
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        
        # Use the correct model name for Jina AI v2
        data = {
            'input': clean_texts,
            'model': 'jina-embeddings-v2-base-en'  # Hardcode the correct model name
            # Removed encoding_format as it's not supported
        }
        
        print(f"Sending embedding request for {len(clean_texts)} texts")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code != 200:
            print(f"Jina AI API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
        response.raise_for_status()
        
        result = response.json()
        embeddings = [item['embedding'] for item in result['data']]
        print(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
        
    except requests.exceptions.RequestException as e:
        print(f"Network error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding API request failed: {e}")
    except ValueError as e:
        print(f"Input validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input for embeddings: {e}")
    except Exception as e:
        print(f"Unexpected error generating embeddings: {e}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

def get_single_embedding(text: str):
    """Get embedding for single text"""
    return get_embeddings_from_jina([text])[0]

def call_groq_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Call Groq LLM for text generation"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert assistant for India groundwater resources (INGRES). Use only the provided context to answer questions accurately. If information is not available, state so clearly."},
                {"role": "user", "content": prompt}
            ],
            model=LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq LLM call failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

def upsert_chunks_to_pinecone(chunks: List[Dict[str, Any]]):
    """Upsert text chunks with embeddings to Pinecone"""
    try:
        vectors_to_upsert = []
        
        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            # Generate embeddings
            embeddings = get_embeddings_from_jina(texts)
            
            # Prepare vectors for upsert
            for j, chunk in enumerate(batch):
                vectors_to_upsert.append({
                    'id': chunk["id"],
                    'values': embeddings[j],
                    'metadata': {
                        'text': chunk["text"],
                        **chunk.get("metadata", {})
                    }
                })
        
        # Upsert to Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"✅ Upserted {len(vectors_to_upsert)} vectors to Pinecone")
        
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Vector upsert failed: {e}")

def query_pinecone(vector: List[float], k: int = 6, filters: Optional[Dict[str, Any]] = None):
    """Query Pinecone for similar vectors"""
    try:
        query_response = index.query(
            vector=vector, 
            top_k=k, 
            filter=filters or {}, 
            include_metadata=True, 
            include_values=False
        )
        
        # Format results
        matches = []
        for match in query_response.get("matches", []):
            matches.append({
                "id": match["id"], 
                "score": match["score"], 
                "metadata": match.get("metadata", {}),
                "text": match.get("metadata", {}).get("text", "")
            })
        return matches
        
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # Only add non-empty chunks
            chunks.append(" ".join(chunk_words))
    
    return chunks if chunks else [text]  # Return original text if chunking fails

# ---------- API Endpoints ----------
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "service": "INGRES RAG Backend",
        "version": "2.0.0",
        "pinecone_connected": index is not None,
        "groq_configured": bool(GROQ_API_KEY),
        "jina_configured": bool(JINA_API_KEY)
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "ok",
        "services": {
            "pinecone": index is not None,
            "groq": bool(GROQ_API_KEY),
            "jina": bool(JINA_API_KEY)
        },
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "vector_dim": VECTOR_DIM,
            "default_top_k": DEFAULT_TOP_K
        }
    }

@app.post("/ingest")
async def ingest(req: IngestRequest, _: bool = Depends(verify_token) if API_AUTH_TOKEN else None):
    """
    Ingest documents into the vector database.
    Supports both direct text and URL-based document ingestion.
    """
    try:
        text = req.text
        
        # Fetch text from URL if not provided directly
        if not text and req.source_url:
            try:
                response = requests.get(req.source_url, timeout=30)
                response.raise_for_status()
                text = response.text
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text content found")

        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to chunk text")

        # Prepare chunks for upsert
        upsert_payload = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only include non-empty chunks
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # Flatten region_meta to avoid nested objects
                metadata = {
                    "doc_id": doc_id,
                    "title": req.title or "Unknown Document",
                    "source_url": req.source_url or "",
                    "chunk_index": i,
                    "created_at": time.time()
                }
                
                # Add region metadata as flattened fields
                if req.region_meta:
                    for key, value in req.region_meta.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"region_{key}"] = value
                        else:
                            metadata[f"region_{key}"] = str(value)
                
                upsert_payload.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata
                })

        # Upsert to Pinecone
        if upsert_payload:
            upsert_chunks_to_pinecone(upsert_payload)
            
        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_processed": len(upsert_payload),
            "total_characters": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks, _: bool = Depends(verify_token) if API_AUTH_TOKEN else None):
    """
    Main RAG query endpoint for groundwater resources information.
    Supports multiple languages with auto-detection.
    """
    try:
        user_query = req.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Empty query provided")

        print(f"🔍 Processing query: {user_query[:100]}...")

        # 1. Language detection
        detected_lang = req.language or detect_language_safe(user_query)
        
        print(f"🌐 Language: {detected_lang}, Query: {user_query[:50]}...")

        # 2. Generate query embedding
        query_vector = get_single_embedding(user_query)

        # 3. Search Pinecone
        matches = query_pinecone(
            vector=query_vector, 
            k=req.k or DEFAULT_TOP_K, 
            filters=req.filters
        )

        print(f"📊 Found {len(matches)} relevant chunks")

        # 4. Build context from retrieved documents
        context_chunks = []
        for match in matches:
            metadata = match.get("metadata", {})
            text = match.get("text", "")
            title = metadata.get("title", "Unknown")
            score = match.get("score", 0.0)
            
            context_chunks.append({
                "text": text[:800],  # Limit chunk size
                "title": title,
                "score": score,
                "metadata": metadata
            })

        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(
                f"Document {i+1} (Title: {chunk['title']}, Relevance: {chunk['score']:.3f}):\n{chunk['text']}"
            )
        
        context_text = "\n\n---\n\n".join(context_parts)

        # 5. Build prompt for LLM
        system_context = """You are INGRES, an expert AI assistant specializing in India's groundwater resources and water management. 

Your expertise includes:
- Groundwater levels, quality, and availability across Indian states
- Water conservation techniques and policies
- Agricultural water usage and irrigation systems
- Government schemes and regulations related to water resources
- Technical aspects of wells, aquifers, and water extraction

Instructions:
- Answer based ONLY on the provided context documents
- Be specific and include relevant data, numbers, and location details when available
- If information is not available in the context, clearly state this
- Provide practical, actionable insights when possible
- Use clear, professional language suitable for water resource professionals"""

        user_prompt = f"""Context Documents:
{context_text}

Question: {user_query}

Please provide a comprehensive answer based on the context documents above. Include specific details, numbers, and actionable insights where available."""

        # 6. Generate answer using Groq
        print("🤖 Generating answer with Groq LLM...")
        answer = call_groq_llm(
            prompt=f"{system_context}\n\n{user_prompt}",
            max_tokens=800,
            temperature=0.1
        )

        # 7. Prepare response
        query_id = str(uuid.uuid4())
        
        # Log query for analytics (background task)
        background_tasks.add_task(log_query, query_id, user_query, answer)

        response = {
            "query_id": query_id,
            "language_detected": detected_lang,
            "answer": answer,
            "sources": [
                {
                    "title": chunk["title"],
                    "relevance_score": chunk["score"],
                    "chunk_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                }
                for chunk in context_chunks[:3]  # Top 3 sources
            ],
            "total_sources": len(matches)
        }

        print(f"✅ Query processed successfully: {len(answer)} chars response")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

@app.post("/feedback")
async def feedback(req: FeedbackRequest, _: bool = Depends(verify_token) if API_AUTH_TOKEN else None):
    """Submit feedback for query responses"""
    try:
        feedback_data = {
            "query_id": req.query_id,
            "correct": req.correct,
            "notes": req.notes,
            "timestamp": time.time(),
            "feedback_id": str(uuid.uuid4())
        }
        
        # Save feedback to file (in production, use database)
        feedback_file = "feedback.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        
        print(f"📝 Feedback received for query {req.query_id}: {'✅' if req.correct else '❌'}")
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_data["feedback_id"]
        }
        
    except Exception as e:
        print(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")

# ---------- Helper Functions ----------
def log_query(query_id: str, query: str, response: str):
    """Log query and response for analytics"""
    try:
        log_data = {
            "query_id": query_id,
            "query": query,
            "response": response,
            "timestamp": time.time()
        }
        
        log_file = "query_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"Logging error: {e}")

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 INGRES RAG Backend starting up...")
    print(f"   📊 Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"   🤖 LLM Model: {LLM_MODEL}")
    print(f"   🔤 Embedding Model: {EMBEDDING_MODEL}")
    print("✅ Startup complete!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
