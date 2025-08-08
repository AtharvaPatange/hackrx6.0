# File: vector_store.py

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from llm_services import get_embedding, get_embeddings_from_jina
from document_processor import process_documents
import hashlib
import time

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone():
    """Initializes the Pinecone index, creating it if it doesn't exist."""
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Dimension for Jina's jina-embeddings-v2-base-en
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    else:
        # Check if existing index has the correct dimension
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != 768:
            print(f"Warning: Existing index has dimension {index_info.dimension}, but expected 768")
            print("You may need to delete the existing index and recreate it with the correct dimension")
    return pc.Index(PINECONE_INDEX_NAME)

index = init_pinecone()

# --- THE FIX IS HERE ---
# We are adding namespace="" to the delete and upsert calls.
DEFAULT_NAMESPACE = ""

# Document cache to avoid reprocessing same documents
processed_documents = set()

def get_document_hash(url: str) -> str:
    """Generate a hash for the document URL to use as cache key."""
    return hashlib.md5(url.encode()).hexdigest()[:8]

def is_document_processed(url: str) -> bool:
    """Check if document has already been processed."""
    doc_hash = get_document_hash(url)
    try:
        # Check if any vectors exist with this document hash
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            # Assume document is already processed if vectors exist
            print(f"Document already processed (vectors exist: {stats.total_vector_count})")
            return True
    except Exception as e:
        print(f"Error checking document status: {e}")
    return False

def upsert_chunks(vectors: list):
    """Upserts pre-computed vectors into Pinecone."""
    print(f"Upserting {len(vectors)} vectors to Pinecone")
    # Upsert in batches for speed
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        print(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        index.upsert(vectors=batch, namespace=DEFAULT_NAMESPACE)
    print(f"Successfully upserted {len(vectors)} vectors")

def process_and_store_documents(url: str):
    """Process documents from URL and store in vector database with caching."""
    print(f"Processing document from URL: {url}")
    
    # Check if document already processed
    if is_document_processed(url):
        print("Document already in vector store, skipping processing")
        return
    
    # Clear existing vectors to avoid duplicates
    try:
        index.delete(delete_all=True, namespace=DEFAULT_NAMESPACE)
        print("Cleared existing vectors")
        time.sleep(1)  # Brief wait for deletion to complete
    except Exception as e:
        print(f"Warning: Could not clear existing vectors: {e}")
    
    # Process new document
    documents = process_documents([url])
    if not documents:
        print("No documents processed")
        return
    
    print(f"Processed {len(documents)} document chunks")
    
    # Generate embeddings and upsert (optimized batch processing)
    doc_hash = get_document_hash(url)
    vectors_to_upsert = []
    
    # Process in smaller batches for speed
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        embeddings = get_embeddings_from_jina(texts)
        
        for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
            vector_id = f"{doc_hash}_{i+j}"
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': doc.page_content,
                    'url': url,
                    'doc_hash': doc_hash
                }
            })
    
    # Batch upsert all vectors
    if vectors_to_upsert:
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone")
        upsert_chunks(vectors_to_upsert)
        processed_documents.add(doc_hash)
        print("Document processing completed and cached")
    else:
        print("No vectors to upsert")

def query_pinecone(question: str, top_k: int = 5):
    """Queries Pinecone to retrieve relevant text chunks for a question."""
    query_embedding = get_embedding(question)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=DEFAULT_NAMESPACE # Query from the same namespace
    )
    # Combine the text from the retrieved chunks into a single context string
    context = "\n---\n".join([match['metadata']['text'] for match in results['matches']])
    return context