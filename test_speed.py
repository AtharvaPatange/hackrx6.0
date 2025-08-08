#!/usr/bin/env python3
"""
Quick speed test for the optimized HackRx API
"""
import requests
import time
import json

# Test configuration
API_URL = "https://hackrx6-0-xbz8.onrender.com/hackrx/run"
AUTH_TOKEN = "hackrx_secret_auth_token_2024"

# Test payload
test_data = {
    "documents": "https://www.irdai.gov.in/admincms/cms/frmGeneral_Layout.aspx?page=PageNo3149&flag=1",
    "questions": [
        "What is the sum insured mentioned in the document?",
        "What are the coverage details?",
        "What are the exclusions mentioned?"
    ]
}

def test_api_speed():
    print("🚀 Testing optimized HackRx API speed...")
    print(f"URL: {API_URL}")
    print(f"Document: {test_data['documents']}")
    print(f"Questions: {len(test_data['questions'])}")
    print("-" * 50)
    
    start_time = time.time()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    try:
        print("⏳ Sending request...")
        response = requests.post(
            API_URL, 
            json=test_data, 
            headers=headers,
            timeout=35  # 35-second timeout to test our 30s target
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"📊 RESPONSE TIME: {total_time:.2f} seconds")
        
        if total_time <= 30:
            print("✅ SUCCESS: Response time is within 30-second hackathon requirement!")
        else:
            print("❌ FAILED: Response time exceeds 30-second limit")
        
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📝 Answers received: {len(result.get('answers', []))}")
            
            for i, answer in enumerate(result.get('answers', [])[:2]):  # Show first 2
                print(f"\n🔍 Answer {i+1} (first 100 chars):")
                print(f"   {answer[:100]}...")
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        end_time = time.time()
        print(f"⏰ TIMEOUT after {end_time - start_time:.2f} seconds")
        print("❌ API took longer than 35 seconds")
    except Exception as e:
        end_time = time.time()
        print(f"💥 ERROR after {end_time - start_time:.2f} seconds: {e}")

if __name__ == "__main__":
    test_api_speed()
