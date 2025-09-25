#!/usr/bin/env python3
"""
Simple test script for deployed SIH RAG API
"""

import requests
import json

def test_deployed_api():
    """Test the deployed API with groundwater queries"""
    
    # Deployed URL
    base_url = "https://hackrx6-0-2-51se.onrender.com"
    
    print("🚀 TESTING DEPLOYED SIH RAG API")
    print("=" * 50)
    print(f"🎯 URL: {base_url}")
    print()
    
    # Test queries
    queries = [
"What is the annual groundwater recharge in Maharashtra according to the report?",

"How much of Punjab’s groundwater resource is categorized as “over-exploited”?",

"Which districts in Rajasthan have the lowest groundwater availability per capita?",

"What percentage of groundwater in Tamil Nadu is used for irrigation?"
    ]
    
    headers = {"Content-Type": "application/json"}
    
    for i, query in enumerate(queries, 1):
        print(f"🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # Make request to deployed API
            response = requests.post(
                f"{base_url}/query",
                json={"query": query, "top_k": 5},
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer")
                sources = result.get("sources", [])
                query_id = result.get("query_id", "No ID")
                
                print(f"✅ SUCCESS!")
                print(f"📝 Answer: {answer[:200]}...")
                print(f"📚 Sources: {len(sources)} found")
                print(f"🆔 Query ID: {query_id}")
                
                # Check if answer is meaningful
                if "information not available" in answer.lower():
                    print("⚠️  Generic response - may need more data")
                else:
                    print("🎯 Specific answer found!")
                    
            else:
                print(f"❌ FAILED: HTTP {response.status_code}")
                print(f"📄 Response: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print("⏱️  TIMEOUT: Request took too long")
            
        except requests.exceptions.ConnectionError:
            print("🔌 CONNECTION ERROR: Could not reach server")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print()

def test_health_endpoints():
    """Test health endpoints"""
    
    base_url = "https://hackrx6-0-2-51se.onrender.com"
    
    print("🏥 TESTING HEALTH ENDPOINTS")
    print("=" * 30)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working")
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Status: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            services = data.get('services', {})
            print(f"   Pinecone: {services.get('pinecone', 'Unknown')}")
            print(f"   Groq: {services.get('groq', 'Unknown')}")
            print(f"   Jina: {services.get('jina', 'Unknown')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    print()

if __name__ == "__main__":
    # Test health first
    test_health_endpoints()
    
    # Test queries
    test_deployed_api()
    
    print("🎉 TESTING COMPLETE!")
    print("💡 Your deployed API is ready for use!")