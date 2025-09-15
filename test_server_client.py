#!/usr/bin/env python3
"""
Simple test client for the SIH server
"""

import requests
import json
import time

def test_server_ingestion():
    """Test server ingestion with simple data"""
    
    url = "http://localhost:8001/ingest"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
    }
    
    test_data = {
        "text": "This is a simple test document about groundwater resources in India. It contains basic information for testing the embedding functionality. Groundwater is a crucial resource for agricultural and domestic use. India has vast groundwater reserves that need careful management. The Central Ground Water Board monitors groundwater levels across the country.",
        "title": "Test Document - Groundwater Resources",
        "source_url": "test://localhost/test-doc",
        "region_meta": {
            "state": "Test State",
            "region": "Test Region",
            "category": "educational"
        }
    }
    
    try:
        print("🧪 Testing server ingestion...")
        print(f"📡 Sending request to: {url}")
        
        response = requests.post(url, headers=headers, json=test_data, timeout=120)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📄 Document ID: {result.get('doc_id')}")
            print(f"🧩 Chunks processed: {result.get('chunks_processed')}")
            print(f"📏 Total characters: {result.get('total_characters')}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("🌐 Connection error - is the server running?")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def test_server_query():
    """Test server query functionality"""
    
    url = "http://localhost:8001/query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
    }
    
    test_query = {
        "query": "What is groundwater?",
        "top_k": 3
    }
    
    try:
        print("\n🔍 Testing server query...")
        print(f"📡 Sending query: {test_query['query']}")
        
        response = requests.post(url, headers=headers, json=test_query, timeout=60)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"💬 Answer: {result.get('answer', 'No answer')[:100]}...")
            print(f"📚 Sources found: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ Query error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"💥 Query error: {e}")
        return False

def check_server_health():
    """Check if server is responsive"""
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ Server is healthy!")
            print(f"   - Pinecone: {'✅' if health.get('pinecone') else '❌'}")
            print(f"   - Groq: {'✅' if health.get('groq') else '❌'}")
            print(f"   - Jina: {'✅' if health.get('jina') else '❌'}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"💥 Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 SIH Server Test Client\n")
    
    # Check server health first
    if not check_server_health():
        print("❌ Server is not healthy - cannot proceed with tests")
        exit(1)
    
    print("\n" + "="*50)
    
    # Test ingestion
    ingestion_success = test_server_ingestion()
    
    if ingestion_success:
        # Wait a moment for indexing
        print("⏳ Waiting 3 seconds for indexing...")
        time.sleep(3)
        
        # Test query
        query_success = test_server_query()
        
        if query_success:
            print("\n🎉 All tests passed! Server is working correctly.")
        else:
            print("\n⚠️ Ingestion works, but query failed.")
    else:
        print("\n❌ Ingestion test failed - check server logs")
