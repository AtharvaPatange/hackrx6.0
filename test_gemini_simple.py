#!/usr/bin/env python3
"""
Simple test script to verify Gemini API key using google-generativeai library
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_simple():
    """Test Gemini API with simple google-generativeai library"""
    try:
        import google.generativeai as genai
        
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found!")
            return False
            
        print(f"🔑 API Key: {api_key[:20]}...{api_key[-10:]}")
        
        genai.configure(api_key=api_key)
        
        # Test simple text generation
        print("🧪 Testing Gemini API...")
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content("Hello! Just respond with 'API is working' to test.")
        
        print(f"✅ Response: {response.text}")
        return True
        
    except ImportError:
        print("❌ google-generativeai library not installed. Install with:")
        print("pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

if __name__ == "__main__":
    print("🔑 Testing Gemini API Key (Simple Test)...\n")
    
    success = test_gemini_simple()
    
    if success:
        print("\n✅ Gemini API key is working!")
    else:
        print("\n❌ Gemini API test failed!")
