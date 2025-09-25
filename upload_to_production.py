#!/usr/bin/env python3
"""
Upload local groundwater data to production Render deployment
"""

import requests
import json

# Production URL
PRODUCTION_URL = "https://hackrx6-0-2-51se.onrender.com"

def upload_groundwater_data():
    """Upload your local groundwater document to production"""
    
    print("🚀 UPLOADING GROUNDWATER DATA TO PRODUCTION")
    print("=" * 60)
    
    # Read your local groundwater document
    file_path = "D:\\HackrX\\Ground Water Formula related doc.pdf"
    
    try:
        # You can either:
        # Option 1: Use the clean text we extracted earlier
        print("📄 Preparing groundwater content...")
        
        # Sample groundwater content (you can replace with full extracted text)
        groundwater_content = """
        INDIA GEC Software - User Manual
        
        This manual covers the automation of estimation of Dynamic Ground Water Resources 
        using GEC-2015 methodology and related research work to improve GEC Assessment.
        
        The GEC-2015 methodology is used for groundwater resource assessment in India.
        It involves systematic evaluation of aquifer properties, recharge rates, and 
        discharge patterns to determine sustainable groundwater extraction limits.
        
        Key components of GEC methodology:
        1. Assessment of groundwater occurrence
        2. Evaluation of aquifer characteristics  
        3. Estimation of recharge and discharge
        4. Determination of net groundwater availability
        5. Sustainable extraction limits
        
        The INDIA GEC Software automates these calculations and provides 
        standardized reporting for groundwater resource management.
        """
        
        # Upload to production
        payload = {
            "text": groundwater_content,
            "title": "India GEC Software Manual - Groundwater Assessment",
            "region_meta": {
                "document_type": "technical_manual",
                "agency": "Central Ground Water Board",
                "methodology": "GEC-2015",
                "country": "India",
                "topic": "groundwater_assessment"
            }
        }
        
        print("📤 Uploading to production...")
        response = requests.post(
            f"{PRODUCTION_URL}/ingest",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ UPLOAD SUCCESSFUL!")
            print(f"   📄 Document ID: {result.get('doc_id')}")
            print(f"   📦 Chunks created: {result.get('chunks_processed')}")
            print(f"   📏 Characters: {result.get('total_characters')}")
            
            # Test the uploaded data
            print("\n🧪 Testing uploaded data...")
            test_query = {
                "query": "What is the GEC methodology for groundwater assessment?",
                "mode": "text"
            }
            
            test_response = requests.post(
                f"{PRODUCTION_URL}/query",
                json=test_query,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if test_response.status_code == 200:
                test_result = test_response.json()
                answer = test_result.get('answer', '')
                
                if "information not available" not in answer.lower():
                    print("✅ TEST SUCCESSFUL!")
                    print(f"   💬 Answer: {answer[:200]}...")
                    print(f"   📚 Sources: {len(test_result.get('sources', []))}")
                else:
                    print("⚠️  Test shows generic response - may need more data")
            else:
                print("❌ Test query failed")
                
        else:
            print(f"❌ Upload failed: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Upload error: {e}")

def main():
    print("🌐 PRODUCTION DATA UPLOAD")
    print(f"🎯 Target: {PRODUCTION_URL}")
    print()
    
    upload_groundwater_data()
    
    print("\n💡 NEXT STEPS:")
    print("   1. Test queries about GEC methodology")
    print("   2. Upload more groundwater documents if needed")
    print("   3. Your production API is ready!")

if __name__ == "__main__":
    main()