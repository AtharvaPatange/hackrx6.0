import requests
import json
import time

# Test the SIH API
def test_sih_api():
    base_url = "http://localhost:8001"
    
    print("🧪 Testing SIH INGRES RAG API...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Services: {health_data.get('services', {})}")
            print(f"   Config: {health_data.get('config', {})}")
        else:
            print(f"❌ Health check failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            root_data = response.json()
            print(f"   Status: {root_data.get('status')}")
            print(f"   Version: {root_data.get('version')}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 3: Sample query (without authentication for now)
    query_data = {
        "query": "What is groundwater in India?",
        "k": 3
    }
    
    try:
        response = requests.post(
            f"{base_url}/query", 
            json=query_data,
            timeout=30
        )
        print(f"✅ Query endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources: {result.get('total_sources', 0)}")
            print(f"   Language detected: {result.get('language_detected', 'unknown')}")
        elif response.status_code == 401:
            print(f"   ℹ️  Authentication required (expected)")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Query endpoint failed: {e}")
    
    print("🎯 SIH API test completed!")

if __name__ == "__main__":
    test_sih_api()
