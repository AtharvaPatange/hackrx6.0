#!/usr/bin/env python3
"""
Direct test of PDF ingestion to isolate the issue
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8001"
AUTH_TOKEN = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"

def test_direct_ingestion():
    """Test direct ingestion with text data"""
    
    url = f"{BASE_URL}/ingest"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test data with actual groundwater content
    payload = {
        "title": "Groundwater Resources of India - Test Document",
        "text": "India has vast groundwater resources that are crucial for agriculture and domestic use. The Central Ground Water Board monitors groundwater levels across different states. Major groundwater-dependent states include Punjab, Haryana, Rajasthan, and Tamil Nadu. Over-extraction has led to declining water tables in many regions. The government has implemented the Atal Bhujal Yojana to promote sustainable groundwater management. Rainwater harvesting is essential for aquifer recharge. Quality issues include contamination from industrial waste and naturally occurring elements like arsenic and fluoride.",
        "region_meta": {
            "state": "All India",
            "year": 2024,
            "report_type": "test_document",
            "agency": "Test Agency",
            "language": "English"
        }
    }
    
    try:
        print("🧪 Testing direct ingestion...")
        print(f"📡 URL: {url}")
        print(f"📝 Payload size: {len(json.dumps(payload))} chars")
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📄 Document ID: {result.get('doc_id')}")
            print(f"🧩 Chunks processed: {result.get('chunks_processed')}")
            print(f"📏 Total characters: {result.get('total_characters')}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                print(f"💥 Detail: {error_data.get('detail', 'No details available')}")
            except:
                pass
            
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

def test_server_status():
    """Test server status"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
            return True
        else:
            print(f"⚠️ Server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Direct Ingestion Test\n")
    
    # Check server first
    if test_server_status():
        print("\n" + "="*50)
        success = test_direct_ingestion()
        
        if success:
            print("\n🎉 Direct ingestion test passed!")
        else:
            print("\n❌ Direct ingestion test failed!")
    else:
        print("❌ Cannot proceed - server is not accessible")
