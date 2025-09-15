#!/usr/bin/env python3
"""
Direct test of embedding function without server
"""

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")

def get_embeddings_from_jina(texts: list):
    """Generate embeddings using Jina AI API - direct test"""
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
        raise
    except ValueError as e:
        print(f"Input validation error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error generating embeddings: {e}")
        print(f"Error type: {type(e)}")
        raise

def test_embedding_function():
    """Test the embedding function directly"""
    test_texts = [
        "This is a test document about groundwater resources.",
        "India has vast groundwater reserves that need careful management.",
        "Groundwater depletion is a major concern in agricultural regions."
    ]
    
    try:
        print("🧪 Testing embedding function directly...")
        embeddings = get_embeddings_from_jina(test_texts)
        
        print(f"✅ Success! Generated {len(embeddings)} embeddings")
        print(f"📏 Embedding dimensions: {[len(emb) for emb in embeddings]}")
        print(f"🎯 Sample embedding values: {embeddings[0][:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Direct Embedding Function Test\n")
    success = test_embedding_function()
    
    if success:
        print("\n✅ Embedding function is working correctly!")
    else:
        print("\n❌ Embedding function test failed!")
