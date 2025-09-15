#!/usr/bin/env python3
"""
Test script to verify query endpoint works without authentication
"""

import requests
import json

def test_public_query():
    """Test the query endpoint without authentication"""
    
    url = "http://localhost:8001/query"
    
    # No Authorization header needed now
    headers = {
        "Content-Type": "application/json"
    }
    
    test_query = {
        "query": "What is the groundwater situation in India?",
        "k": 3
    }
    
    try:
        print("🧪 Testing /query endpoint without authentication...")
        print(f"📡 URL: {url}")
        print(f"📝 Query: {test_query['query']}")
        
        response = requests.post(url, headers=headers, json=test_query, timeout=60)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Query works without authentication")
            print(f"🆔 Query ID: {result.get('query_id')}")
            print(f"🌐 Language: {result.get('language_detected')}")
            print(f"💬 Answer: {result.get('answer', '')[:100]}...")
            print(f"📚 Sources: {result.get('total_sources')} found")
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def test_network_url():
    """Instructions for testing from friend's laptop"""
    print("\n" + "="*60)
    print("🌐 NETWORK ACCESS INSTRUCTIONS FOR YOUR FRIEND")
    print("="*60)
    print()
    print("Your friend should use this URL from his laptop:")
    print("📡 http://YOUR_IP_ADDRESS:8001/query")
    print()
    print("Example request using curl:")
    print('curl -X POST "http://YOUR_IP_ADDRESS:8001/query" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "What is groundwater in India?", "k": 3}\'')
    print()
    print("Example request using Python:")
    print("```python")
    print("import requests")
    print()
    print('url = "http://YOUR_IP_ADDRESS:8001/query"')
    print("headers = {'Content-Type': 'application/json'}")
    print("data = {'query': 'What is groundwater in India?', 'k': 3}")
    print()
    print("response = requests.post(url, headers=headers, json=data)")
    print("print(response.json())")
    print("```")
    print()
    print("🔧 Make sure to:")
    print("1. Replace YOUR_IP_ADDRESS with your actual IP")
    print("2. Ensure your firewall allows connections on port 8001")
    print("3. Server should be running with --host 0.0.0.0")

if __name__ == "__main__":
    print("🔬 Public Query Endpoint Test\n")
    
    # Test locally first
    success = test_public_query()
    
    if success:
        test_network_url()
        print("\n✅ Authentication disabled successfully!")
        print("🚀 Your friend can now access the /query endpoint!")
    else:
        print("\n❌ Test failed - make sure server is running")
        print("💡 Restart your server after commenting out API_AUTH_TOKEN")