# File: vector_store.py

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from llm_services import get_embedding

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

def upsert_chunks(chunks: list[str]):
    """Generates embeddings and upserts chunks into Pinecone."""
    # Clear previous document data from the default namespace (only if it exists)
    try:
        print(f"Clearing old data from namespace: '{DEFAULT_NAMESPACE}'")
        index.delete(delete_all=True, namespace=DEFAULT_NAMESPACE)
        print("Successfully cleared old data.")
    except Exception as e:
        if "Namespace not found" in str(e):
            print(f"Namespace '{DEFAULT_NAMESPACE}' doesn't exist yet, skipping deletion.")
        else:
            print(f"Error clearing old data: {e}")
            # Re-raise if it's not a namespace not found error
            raise
    
    vectors_to_upsert = []
    print(f"Generating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if (i + 1) % 20 == 0:  # Print progress every 20 chunks (reduced frequency)
            print(f"Processing chunk {i + 1}/{len(chunks)}")
        embedding = get_embedding(chunk)
        vectors_to_upsert.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    print("All embeddings generated. Starting upsert to Pinecone...")
    # Upsert in larger batches for speed
    batch_size = 200  # Increased batch size
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
        print(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        index.upsert(vectors=batch, namespace=DEFAULT_NAMESPACE)

    print(f"Upserted {len(vectors_to_upsert)} chunks to Pinecone in namespace '{DEFAULT_NAMESPACE}'.")

def query_pinecone(question: str, top_k: int = 8):
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