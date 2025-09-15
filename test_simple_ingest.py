import requests
import json

def ingest_simple_test():
    """Test simple document ingestion"""
    base_url = "http://localhost:8001"
    auth_token = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    print("📥 Testing Simple Document Ingestion...")
    
    # Smaller test document
    simple_doc = {
        "title": "Groundwater Test",
        "text": """Groundwater is water located underground in aquifers. 
        India uses 25% of global groundwater. 
        Punjab and Haryana have groundwater depletion issues. 
        CGWB monitors groundwater levels across India.""",
        "region_meta": {"type": "test"}
    }
    
    try:
        print("🔄 Ingesting simple document...")
        response = requests.post(f"{base_url}/ingest", json=simple_doc, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document ingested successfully!")
            print(f"   Doc ID: {result.get('doc_id')}")
            print(f"   Chunks processed: {result.get('chunks_processed')}")
            
            # Wait for indexing
            print("⏳ Waiting for indexing...")
            import time
            time.sleep(5)
            
            # Test query
            print("🔍 Testing query with new data...")
            query_data = {"query": "What percentage of global groundwater does India use?", "k": 2}
            
            response = requests.post(f"{base_url}/query", json=query_data, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                print(f"✅ Query successful!")
                print(f"   Answer: {answer[:200]}...")
                print(f"   Sources: {result.get('total_sources', 0)}")
            else:
                print(f"❌ Query failed: {response.status_code}")
        else:
            print(f"❌ Ingestion failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    ingest_simple_test()
