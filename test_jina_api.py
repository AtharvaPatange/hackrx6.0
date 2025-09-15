#!/usr/bin/env python3
"""
Quick test script to verify Jina AI API functionality
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

def test_jina_api():
    """Test Jina AI API with a simple request"""
    
    if not JINA_API_KEY:
        print("❌ JINA_API_KEY not found in environment")
        return False
    
    print(f"🔑 Using API Key: {JINA_API_KEY[:20]}...")
    
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    
    # Simple test data
    test_data = {
        'input': ['Hello world', 'This is a test'],
        'model': 'jina-embeddings-v2-base-en'
        # Removed encoding_format as it's not supported
    }
    
    try:
        print("📡 Sending test request to Jina AI...")
        response = requests.post(url, headers=headers, json=test_data, timeout=30)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            embeddings_count = len(result.get('data', []))
            embedding_dim = len(result['data'][0]['embedding']) if embeddings_count > 0 else 0
            
            print(f"✅ Success! Generated {embeddings_count} embeddings")
            print(f"📏 Embedding dimension: {embedding_dim}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("🌐 Connection error")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def test_different_models():
    """Test different Jina AI model names"""
    
    models_to_test = [
        'jina-embeddings-v2-base-en',
        'jina-embeddings-v2-small-en',
        'jina-embeddings-v3'
    ]
    
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINA_API_KEY}'
    }
    
    for model in models_to_test:
        print(f"\n🧪 Testing model: {model}")
        
        test_data = {
            'input': ['Test text'],
            'model': model
            # Removed encoding_format as it's not supported
        }
        
        try:
            response = requests.post(url, headers=headers, json=test_data, timeout=15)
            if response.status_code == 200:
                print(f"✅ {model} - Working!")
            else:
                print(f"❌ {model} - Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
        except Exception as e:
            print(f"💥 {model} - Error: {e}")

if __name__ == "__main__":
    print("🔬 Testing Jina AI API Configuration\n")
    
    # Basic API test
    success = test_jina_api()
    
    if not success:
        print("\n🔍 Testing different model names...")
        test_different_models()
    
    print("\n" + "="*50)
    if success:
        print("✅ Jina AI API is working correctly!")
        print("🚀 You can proceed with PDF ingestion")
    else:
        print("❌ Jina AI API test failed")
        print("🔧 Please check your API key or try again later")
