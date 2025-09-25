#!/usr/bin/env python3
"""
Test script for deployed SIH RAG API on Render
Tests all endpoints and functionality
"""

import requests
import json
import time
from typing import Dict, Any

# Deployed URL
BASE_URL = "https://hackrx6-0-2-51se.onrender.com"

class RenderAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SIH-RAG-Tester/1.0'
        })
        
    def test_endpoint(self, method: str, endpoint: str, payload: Dict = None, timeout: int = 30) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            print(f"🔍 Testing {method.upper()} {endpoint}")
            
            start_time = time.time()
            
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=payload, timeout=timeout)
            else:
                return {"status": "error", "message": f"Unsupported method: {method}"}
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text[:500]}
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response_time": f"{response_time}s",
                "response_size": len(response.content),
                "data": response_data
            }
            
            # Print result
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: {response.status_code} ({response_time}s)")
                print(f"   📊 Response: {json.dumps(response_data, indent=2)[:200]}...")
            else:
                print(f"   ❌ FAILED: {response.status_code} ({response_time}s)")
                print(f"   📄 Error: {response.text[:200]}...")
            
            return result
            
        except requests.exceptions.Timeout:
            print(f"   ⏱️  TIMEOUT: Request took longer than {timeout}s")
            return {"status": "timeout", "message": f"Request timeout after {timeout}s"}
            
        except requests.exceptions.ConnectionError:
            print(f"   🔌 CONNECTION ERROR: Could not connect to {url}")
            return {"status": "connection_error", "message": "Could not connect to server"}
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_comprehensive_test(self):
        """Run comprehensive API tests"""
        print("🚀 TESTING SIH RAG API ON RENDER")
        print("=" * 60)
        print(f"🎯 Target URL: {self.base_url}")
        print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        test_results = {}
        
        # Test 1: Root endpoint
        print("📍 TEST 1: Root Health Check")
        print("-" * 30)
        test_results["root"] = self.test_endpoint("GET", "/")
        print()
        
        # Test 2: Health endpoint
        print("📍 TEST 2: Detailed Health Check")
        print("-" * 30)
        test_results["health"] = self.test_endpoint("GET", "/health")
        print()
        
        # Test 3: Query endpoint (most important)
        print("📍 TEST 3: Query Endpoint")
        print("-" * 30)
        query_payload = {
            "query": "What is groundwater?",
            "mode": "text"
        }
        test_results["query"] = self.test_endpoint("POST", "/query", query_payload, timeout=60)
        print()
        
        # Test 4: Specific groundwater query
        print("📍 TEST 4: Groundwater Specific Query")
        print("-" * 30)
        groundwater_payload = {
            "query": "What is the GEC methodology for groundwater assessment?",
            "mode": "text"
        }
        test_results["groundwater_query"] = self.test_endpoint("POST", "/query", groundwater_payload, timeout=60)
        print()
        
        # Test 5: Ingest endpoint (optional)
        print("📍 TEST 5: Ingest Endpoint")
        print("-" * 30)
        ingest_payload = {
            "text": "Test groundwater data for ingestion",
            "title": "Test Document",
            "region_meta": {"test": True}
        }
        test_results["ingest"] = self.test_endpoint("POST", "/ingest", ingest_payload, timeout=60)
        print()
        
        # Print comprehensive summary
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, Dict]):
        """Print comprehensive test summary"""
        print("=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get("status") == "success")
        failed_tests = total_tests - successful_tests
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Successful: {successful_tests}/{total_tests}")
        print(f"   ❌ Failed: {failed_tests}/{total_tests}")
        print(f"   📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        print()
        
        print(f"📋 DETAILED RESULTS:")
        for test_name, result in results.items():
            status_emoji = "✅" if result.get("status") == "success" else "❌"
            status_code = result.get("status_code", "N/A")
            response_time = result.get("response_time", "N/A")
            
            print(f"   {status_emoji} {test_name.upper()}: HTTP {status_code} ({response_time})")
            
            if result.get("status") == "success":
                # Show key data points
                data = result.get("data", {})
                if "service" in data:
                    print(f"      🔧 Service: {data['service']}")
                if "answer" in data:
                    print(f"      💬 Answer: {data['answer'][:100]}...")
                if "sources" in data:
                    print(f"      📚 Sources: {len(data['sources'])} found")
            else:
                print(f"      ❌ Error: {result.get('message', 'Unknown error')}")
        
        print()
        
        # Deployment status
        if successful_tests >= 3:
            print("🏆 DEPLOYMENT STATUS: EXCELLENT")
            print("✅ Your API is fully functional on Render!")
            print("🌐 Ready for production use")
        elif successful_tests >= 2:
            print("🎉 DEPLOYMENT STATUS: GOOD")
            print("✅ Core functionality working")
            print("🔧 Minor issues may need attention")
        elif successful_tests >= 1:
            print("⚠️  DEPLOYMENT STATUS: PARTIAL")
            print("🔧 Basic connectivity working but features failing")
            print("🛠️  Check environment variables and logs")
        else:
            print("❌ DEPLOYMENT STATUS: FAILED")
            print("🔧 API not responding properly")
            print("🛠️  Check Render logs and configuration")
        
        print()
        print("💡 NEXT STEPS:")
        if failed_tests > 0:
            print("   1. Check Render logs for error details")
            print("   2. Verify environment variables are set")
            print("   3. Ensure all dependencies are installed")
            print("   4. Check API quotas (Groq, Pinecone, Jina)")
        else:
            print("   1. Test with your frontend application")
            print("   2. Monitor performance and logs")
            print("   3. Set up monitoring and alerts")

def main():
    """Run the comprehensive test"""
    tester = RenderAPITester(BASE_URL)
    results = tester.run_comprehensive_test()
    
    # Save results to file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"render_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "base_url": BASE_URL,
            "results": results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()