import requests
import json
import time
from typing import Dict, Any

class SIHAPITester:
    def __init__(self, base_url: str = "http://localhost:8001", auth_token: str = None):
        self.base_url = base_url
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def test_health(self):
        """Test the health endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health Status: {data.get('status')}")
                print(f"   Services: {data.get('services', {})}")
                print(f"   Config: {data.get('config', {})}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def ingest_sample_data(self):
        """Ingest some sample groundwater data"""
        print("\n📥 Testing Document Ingestion...")
        
        sample_documents = [
            {
                "title": "Maharashtra Groundwater Status",
                "text": """Maharashtra is one of India's largest states with significant groundwater challenges. 
                The state has over 43 million wells, making it one of the highest well-density regions in India. 
                The average groundwater level in Maharashtra is declining at 0.1-0.3 meters per year. 
                Major districts like Pune, Aurangabad, and Nashik face severe groundwater depletion.
                The state government has implemented rainwater harvesting schemes and watershed management programs.
                Groundwater quality issues include high fluoride content in Chandrapur and Nagpur districts.
                Agricultural consumption accounts for 85% of groundwater usage in the state.""",
                "region_meta": {"state": "Maharashtra", "type": "state_report"}
            },
            {
                "title": "Rajasthan Desert Groundwater",
                "text": """Rajasthan, being largely in the Thar Desert, faces unique groundwater challenges. 
                The state has both scarce groundwater resources and high salinity issues.
                The Indira Gandhi Canal Project has helped improve groundwater recharge in western Rajasthan.
                Traditional water harvesting systems like bawris and tanks are still used in rural areas.
                The groundwater in Rajasthan is found at depths of 30-60 meters on average.
                Districts like Jaisalmer and Barmer have extremely limited freshwater aquifers.
                The state has implemented strict groundwater extraction regulations.""",
                "region_meta": {"state": "Rajasthan", "type": "desert_study"}
            },
            {
                "title": "Kerala Coastal Aquifers",
                "text": """Kerala's groundwater is heavily influenced by its coastal location and monsoon patterns.
                The state receives 3000mm annual rainfall, providing good groundwater recharge.
                However, saltwater intrusion is a major problem in coastal districts like Kollam and Alappuzha.
                The laterite formations in Kerala provide good groundwater storage capacity.
                Groundwater quality is generally good in the midland and highland regions.
                The state has implemented artificial recharge projects in urban areas like Kochi and Thiruvananthapuram.
                Traditional wells called 'Kinaru' are common in rural Kerala.""",
                "region_meta": {"state": "Kerala", "type": "coastal_study"}
            }
        ]
        
        ingested_docs = []
        for doc in sample_documents:
            try:
                response = requests.post(
                    f"{self.base_url}/ingest",
                    json=doc,
                    headers=self.headers,
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Ingested: {doc['title']}")
                    print(f"   Doc ID: {result.get('doc_id')}")
                    print(f"   Chunks: {result.get('chunks_processed')}")
                    ingested_docs.append(result.get('doc_id'))
                else:
                    print(f"❌ Failed to ingest {doc['title']}: {response.status_code}")
                    if response.status_code == 401:
                        print("   Note: Authentication required. Add API_AUTH_TOKEN to test with auth.")
                    else:
                        print(f"   Error: {response.text}")
                        
                time.sleep(2)  # Wait between ingestions
                
            except Exception as e:
                print(f"❌ Ingestion error for {doc['title']}: {e}")
        
        return ingested_docs
    
    def test_queries(self):
        """Test various groundwater-related queries"""
        print("\n🔍 Testing RAG Queries...")
        
        test_queries = [
            {
                "query": "What is the groundwater situation in Maharashtra?",
                "expected_topics": ["Maharashtra", "wells", "depletion", "agriculture"]
            },
            {
                "query": "Tell me about water harvesting systems in Rajasthan",
                "expected_topics": ["Rajasthan", "bawris", "tanks", "traditional", "desert"]
            },
            {
                "query": "What are the groundwater problems in coastal areas of India?",
                "expected_topics": ["coastal", "saltwater intrusion", "Kerala", "salinity"]
            },
            {
                "query": "How deep is groundwater typically found in India?",
                "expected_topics": ["depth", "meters", "aquifers"]
            },
            {
                "query": "What are the main causes of groundwater depletion?",
                "expected_topics": ["agricultural", "consumption", "extraction", "usage"]
            },
            {
                "query": "भारत में भूजल की स्थिति क्या है?",  # Hindi query
                "expected_topics": ["groundwater", "India", "भूजल"]
            }
        ]
        
        successful_queries = 0
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {test['query'][:50]}{'...' if len(test['query']) > 50 else ''} ---")
            
            try:
                query_data = {
                    "query": test['query'],
                    "k": 5
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json=query_data,
                    headers=self.headers,
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Query successful ({response_time:.2f}s)")
                    print(f"   Language detected: {result.get('language_detected', 'unknown')}")
                    print(f"   Answer length: {len(result.get('answer', ''))} characters")
                    print(f"   Sources found: {result.get('total_sources', 0)}")
                    
                    # Show first 200 characters of answer
                    answer = result.get('answer', '')
                    if answer:
                        print(f"   Answer preview: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                    
                    # Check if expected topics are covered
                    answer_lower = answer.lower()
                    found_topics = [topic for topic in test['expected_topics'] 
                                  if topic.lower() in answer_lower]
                    
                    if found_topics:
                        print(f"   ✅ Found relevant topics: {', '.join(found_topics)}")
                    else:
                        print(f"   ⚠️  Expected topics not found: {', '.join(test['expected_topics'])}")
                    
                    # Show top sources
                    sources = result.get('sources', [])
                    if sources:
                        print(f"   Top source: {sources[0].get('title', 'Unknown')} (score: {sources[0].get('relevance_score', 0):.3f})")
                    
                    successful_queries += 1
                    
                elif response.status_code == 401:
                    print("❌ Authentication required")
                    print("   Add API_AUTH_TOKEN to .env file or remove authentication requirement")
                else:
                    print(f"❌ Query failed: {response.status_code}")
                    print(f"   Error: {response.text[:200]}...")
                
            except Exception as e:
                print(f"❌ Query error: {e}")
            
            time.sleep(1)  # Brief pause between queries
        
        print(f"\n📊 Query Results: {successful_queries}/{len(test_queries)} successful")
        return successful_queries
    
    def test_feedback(self, query_id: str = "test-query-123"):
        """Test the feedback endpoint"""
        print(f"\n📝 Testing Feedback Endpoint...")
        
        feedback_data = {
            "query_id": query_id,
            "correct": True,
            "notes": "Test feedback - response was accurate and helpful"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/feedback",
                json=feedback_data,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Feedback submitted successfully")
                print(f"   Feedback ID: {result.get('feedback_id')}")
                return True
            elif response.status_code == 401:
                print("❌ Authentication required for feedback")
                return False
            else:
                print(f"❌ Feedback failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Feedback error: {e}")
            return False

def main():
    print("🚀 SIH INGRES RAG System - Real Query Testing")
    print("=" * 60)
    
    # Initialize tester
    # Note: Remove auth_token parameter if authentication is not required
    tester = SIHAPITester(
        base_url="http://localhost:8001",
        # auth_token="b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"  # Uncomment if auth required
    )
    
    # Test sequence
    if not tester.test_health():
        print("❌ Health check failed. Make sure the server is running on port 8001")
        return
    
    print("\nℹ️  Note: If ingestion fails due to authentication, the system will still test queries with existing data.")
    ingested_docs = tester.ingest_sample_data()
    
    print("\n⏳ Waiting 5 seconds for vector indexing...")
    time.sleep(5)
    
    successful_queries = tester.test_queries()
    
    # Test feedback (optional)
    tester.test_feedback()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"   Health Check: ✅")
    print(f"   Documents Ingested: {len(ingested_docs)}")
    print(f"   Successful Queries: {successful_queries}")
    print(f"   System Status: {'🟢 All Good!' if successful_queries > 0 else '🟡 Needs Review'}")
    
    print("\n💡 Next Steps:")
    print("   1. Try more specific groundwater queries")
    print("   2. Test with different languages")
    print("   3. Ingest more comprehensive groundwater documents")
    print("   4. Monitor response times and accuracy")

if __name__ == "__main__":
    main()
