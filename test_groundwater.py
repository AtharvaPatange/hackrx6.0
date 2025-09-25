#!/usr/bin/env python3
"""
Quick test for the uploaded groundwater data
"""

import requests

def test_groundwater_queries():
    """Test groundwater-related queries"""
    
    queries = [
        "What is the GEC methodology for groundwater assessment?", 
        "What software is used for groundwater estimation in India?",
        "What is the purpose of INDIA GEC Software?"
    ]
    
    print("🧪 TESTING GROUNDWATER DATA IN RAG SYSTEM")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            response = requests.post(
                'http://localhost:8001/query', 
                json={'query': query},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer')
                sources = result.get('sources', [])
                
                print(f"✅ SUCCESS!")
                print(f"📝 Answer: {answer[:250]}...")
                print(f"📚 Sources: {len(sources)} chunks found")
                
                if "information not available" in answer.lower():
                    print("⚠️  Generic response - data may not be indexed properly")
                else:
                    print("🎯 Specific answer found - data working!")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - SIH server not running")
            print("💡 Start server with: python sih.py")
            break
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_groundwater_queries()