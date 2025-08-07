#!/usr/bin/env python3
"""
Test script to verify Groq + Jina AI API keys are working
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    """Test Groq API with LLaMA model"""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found!")
            return False
            
        print(f"🔑 Groq API Key: {api_key[:20]}...{api_key[-10:]}")
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Test with LLaMA model
        print("🧪 Testing Groq with LLaMA-3.1-8b-instant...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello! Just respond with 'Groq API is working' to test the connection."}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=50,
        )
        
        response = chat_completion.choices[0].message.content
        print(f"✅ Response: {response}")
        return True
        
    except ImportError:
        print("❌ groq library not installed. Install with:")
        print("pip install groq")
        return False
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def test_jina_embeddings():
    """Test Jina AI embeddings API"""
    try:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            print("❌ JINA_API_KEY not found!")
            return False
            
        print(f"🔑 Jina API Key: {api_key[:20]}...{api_key[-10:]}")
        
        print("🧪 Testing Jina AI embeddings...")
        
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            'input': ['This is a test sentence for embeddings.'],
            'model': 'jina-embeddings-v2-base-en'
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        embedding = result['data'][0]['embedding']
        
        print(f"✅ Embeddings working! Generated {len(embedding)} dimensional vector")
        print(f"First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ Jina Embeddings API Error: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Groq + Jina AI API Setup...\n")
    
    groq_ok = test_groq_api()
    print()
    jina_ok = test_jina_embeddings()
    
    print("\n" + "="*50)
    if groq_ok:
        print("✅ Groq (LLaMA) API is working!")
    else:
        print("❌ Groq API failed!")
        
    if jina_ok:
        print("✅ Jina AI Embeddings API is working!")
    else:
        print("❌ Jina AI Embeddings API failed!")
    
    if groq_ok and jina_ok:
        print("\n🎉 All APIs are ready! You can now run your application.")
    else:
        print("\n⚠️  Some APIs need configuration. Please check the errors above.")
