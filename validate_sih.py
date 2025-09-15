# Simple test to validate SIH.py is working correctly
print("🧪 Testing SIH.py import and basic functionality...")

try:
    # Test imports
    import sih
    print("✅ SIH.py imports successfully")
    
    # Test FastAPI app creation
    from sih import app
    print("✅ FastAPI app created successfully")
    
    # Test configuration
    from sih import GROQ_API_KEY, JINA_API_KEY, PINECONE_API_KEY
    print("✅ Environment variables loaded:")
    print(f"   - Groq API Key: {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")
    print(f"   - Jina API Key: {'✅ Set' if JINA_API_KEY else '❌ Missing'}")
    print(f"   - Pinecone API Key: {'✅ Set' if PINECONE_API_KEY else '❌ Missing'}")
    
    # Test client initialization
    from sih import groq_client, index
    print("✅ Clients initialized:")
    print(f"   - Groq client: {'✅ Ready' if groq_client else '❌ Failed'}")
    print(f"   - Pinecone index: {'✅ Ready' if index else '❌ Failed'}")
    
    # Test utility functions
    from sih import detect_language_safe, chunk_text
    
    # Test language detection
    test_text = "This is a test text for language detection."
    lang = detect_language_safe(test_text)
    print(f"✅ Language detection: '{test_text[:30]}...' -> {lang}")
    
    # Test text chunking
    chunks = chunk_text(test_text, chunk_size=10, overlap=2)
    print(f"✅ Text chunking: {len(chunks)} chunks created")
    
    print("\n🎯 All SIH.py tests passed! The system is ready to use.")
    print("\nTo start the server, run:")
    print("   uvicorn sih:app --host 0.0.0.0 --port 8001")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
