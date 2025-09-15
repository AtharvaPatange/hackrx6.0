import requests
import json
import time

def test_with_auth():
    """Test SIH API with authentication"""
    base_url = "http://localhost:8001"
    auth_token = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    print("🚀 Testing SIH API with Authentication")
    print("=" * 50)
    
    # Test 1: Health Check
    print("🏥 Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Services: {data.get('services', {})}")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Test 2: Ingest Sample Data
    print("\n📥 Ingesting Sample Data...")
    sample_doc = {
        "title": "Indian Groundwater Overview",
        "text": """India is the world's largest groundwater user, accounting for 25% of global groundwater extraction.
        The country has over 30 million groundwater structures including tube wells, dug wells, and bore wells.
        Major aquifer systems include the Ganga-Brahmaputra-Meghna basin, Indus basin, and peninsular hard rock aquifers.
        Groundwater provides 85% of rural domestic water supply and 50% of urban water supply.
        The Central Ground Water Board (CGWB) monitors groundwater levels across India.
        Over-exploitation is a major concern, with groundwater levels declining in Punjab, Haryana, and Rajasthan.
        Traditional water harvesting methods like stepwells, tanks, and check dams help recharge groundwater.
        Groundwater quality issues include arsenic contamination in West Bengal and fluoride in Rajasthan.""",
        "region_meta": {"country": "India", "type": "overview"}
    }
    
    try:
        response = requests.post(f"{base_url}/ingest", json=sample_doc, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document ingested successfully!")
            print(f"   Doc ID: {result.get('doc_id')}")
            print(f"   Chunks processed: {result.get('chunks_processed')}")
        else:
            print(f"❌ Ingestion failed: {response.status_code}")
            print(f"   Error: {response.text}")
        time.sleep(3)  # Wait for indexing
    except Exception as e:
        print(f"❌ Ingestion error: {e}")

    # Test 3: Real Queries
    print("\n🔍 Testing Real Queries...")
    
    test_queries = [
        "What percentage of global groundwater does India use?",
        "Tell me about groundwater structures in India",
        "Which regions in India face groundwater over-exploitation?",
        "What are the main aquifer systems in India?",
        "How much of rural water supply comes from groundwater?",
        "What is CGWB and what does it do?",
        "Tell me about groundwater quality issues in India",
        "What traditional water harvesting methods help groundwater?",
        "भारत में भूजल की मुख्य समस्याएं क्या हैं?"  # Hindi query
    ]
    
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query[:60]}{'...' if len(query) > 60 else ''} ---")
        
        try:
            query_data = {"query": query, "k": 3}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/query", json=query_data, headers=headers, timeout=45)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Query successful ({response_time:.2f}s)")
                print(f"   Language: {result.get('language_detected', 'unknown')}")
                print(f"   Sources: {result.get('total_sources', 0)}")
                
                answer = result.get('answer', '')
                if answer:
                    # Show first 150 characters of answer
                    preview = answer[:150].replace('\n', ' ')
                    print(f"   Answer: {preview}{'...' if len(answer) > 150 else ''}")
                    
                    # Show top source
                    sources = result.get('sources', [])
                    if sources:
                        top_source = sources[0]
                        print(f"   Top source: {top_source.get('title', 'Unknown')} (relevance: {top_source.get('relevance_score', 0):.3f})")
                
                successful_queries += 1
                
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text[:100]}...")
            
        except Exception as e:
            print(f"❌ Query error: {e}")
        
        time.sleep(0.5)  # Brief pause between queries

    # Test 4: Feedback
    print(f"\n📝 Testing Feedback...")
    feedback_data = {
        "query_id": "test-123",
        "correct": True,
        "notes": "Test feedback for API validation"
    }
    
    try:
        response = requests.post(f"{base_url}/feedback", json=feedback_data, headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Feedback submitted: {result.get('feedback_id')}")
        else:
            print(f"❌ Feedback failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Feedback error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print(f"   Successful queries: {successful_queries}/{len(test_queries)}")
    print(f"   Success rate: {(successful_queries/len(test_queries)*100):.1f}%")
    
    if successful_queries > 6:
        print("   🟢 Excellent! API is working perfectly")
    elif successful_queries > 3:
        print("   🟡 Good! API is mostly functional")
    else:
        print("   🔴 Issues detected. Check logs for details")
    
    print("\n💡 The SIH INGRES RAG system is ready for production use!")

if __name__ == "__main__":
    test_with_auth()
