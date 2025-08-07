#!/usr/bin/env python3
"""
Simple test script to verify Gemini API key is working
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

def test_gemini_chat():
    """Test the Gemini chat model"""
    try:
        print("Testing Gemini Chat API...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )
        
        response = llm.invoke("Hello! Just say 'API is working' to test the connection.")
        print(f"✅ Chat API Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Chat API Error: {e}")
        return False

def test_gemini_embeddings():
    """Test the Gemini embeddings model"""
    try:
        print("\nTesting Gemini Embeddings API...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Test with a simple text
        test_text = "This is a test sentence for embeddings."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embeddings API working! Generated {len(embedding)} dimensional vector")
        print(f"First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings API Error: {e}")
        return False

if __name__ == "__main__":
    print("🔑 Testing Gemini API Key...\n")
    
    # Check if API key is loaded
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables!")
        exit(1)
    
    print(f"API Key loaded: {api_key[:20]}...{api_key[-10:]}")
    
    # Test both services
    chat_ok = test_gemini_chat()
    embeddings_ok = test_gemini_embeddings()
    
    if chat_ok and embeddings_ok:
        print("\n✅ All Gemini API tests passed! Your API key is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check your API key and network connection.")
