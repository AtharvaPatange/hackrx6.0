#!/usr/bin/env python3
"""
Test Ingested Data - SIH INGRES RAG System
==========================================

This script tests queries against your newly ingested groundwater documents
to verify that the bulk ingestion was successful and the system is working properly.

Usage: python test_ingested_data.py

Author: SIH INGRES Team  
Date: September 2025
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8001"
AUTH_TOKEN = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

class IngestedDataTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = HEADERS
        self.test_results = []
    
    def check_system_health(self):
        """Check if the SIH system is healthy"""
        try:
            print("🏥 Checking system health...")
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                services = health_data.get('services', {})
                config = health_data.get('config', {})
                
                print("✅ System is healthy!")
                print(f"   📊 Pinecone: {'✅' if services.get('pinecone') else '❌'}")
                print(f"   🤖 Groq LLM: {'✅' if services.get('groq') else '❌'}")
                print(f"   🔤 Jina Embeddings: {'✅' if services.get('jina') else '❌'}")
                print(f"   📝 Vector Dimension: {config.get('vector_dim', 'unknown')}")
                print(f"   🧠 LLM Model: {config.get('llm_model', 'unknown')}")
                return True
            else:
                print(f"❌ System health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to SIH system. Is it running on port 8001?")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_query(self, query: str, category: str = "General") -> dict:
        """Test a single query against the ingested data"""
        try:
            print(f"\n🔍 Testing Query ({category}):")
            print(f"   Q: {query}")
            
            # Prepare query request
            query_data = {
                "query": query,
                "k": 5  # Get top 5 most relevant chunks
            }
            
            start_time = time.time()
            
            # Send query to /query endpoint
            response = requests.post(
                f"{self.base_url}/query",
                json=query_data,
                headers=self.headers,
                timeout=60  # 1 minute timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response details
                answer = result.get('answer', '')
                language_detected = result.get('language_detected', 'unknown')
                sources = result.get('sources', [])
                total_sources = result.get('total_sources', 0)
                query_id = result.get('query_id', 'unknown')
                
                print(f"   ✅ Success ({response_time:.2f}s)")
                print(f"   🌐 Language: {language_detected}")
                print(f"   📊 Sources found: {total_sources}")
                
                # Show answer preview
                if answer:
                    answer_preview = answer[:200].replace('\n', ' ')
                    print(f"   💬 Answer: {answer_preview}{'...' if len(answer) > 200 else ''}")
                
                # Show top source
                if sources:
                    top_source = sources[0]
                    print(f"   📄 Top source: {top_source.get('title', 'Unknown')} (relevance: {top_source.get('relevance_score', 0):.3f})")
                
                # Store test result
                test_result = {
                    "query": query,
                    "category": category,
                    "success": True,
                    "response_time": response_time,
                    "language_detected": language_detected,
                    "total_sources": total_sources,
                    "answer_length": len(answer),
                    "query_id": query_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.test_results.append(test_result)
                return test_result
                
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                error_msg = response.text[:200] + "..." if len(response.text) > 200 else response.text
                print(f"   Error: {error_msg}")
                
                test_result = {
                    "query": query,
                    "category": category,
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_msg}",
                    "timestamp": datetime.now().isoformat()
                }
                
                self.test_results.append(test_result)
                return test_result
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Query timed out")
            test_result = {
                "query": query,
                "category": category,
                "success": False,
                "error": "Request timeout",
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            test_result = {
                "query": query,
                "category": category,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(test_result)
            return test_result
    
    def run_comprehensive_tests(self):
        """Run a comprehensive set of test queries"""
        
        # Comprehensive test queries covering different aspects
        test_queries = [
            # Factual queries
            {
                "query": "Which state in India has the highest number of groundwater wells?",
                "category": "Factual - Wells"
            },
            {
                "query": "What percentage of global groundwater does India consume?",
                "category": "Factual - Statistics"
            },
            {
                "query": "What is CGWB and what are its main responsibilities?",
                "category": "Factual - Organizations"
            },
            
            # Regional queries
            {
                "query": "What are the main groundwater problems in Maharashtra?",
                "category": "Regional - Maharashtra"
            },
            {
                "query": "Describe the groundwater situation in Rajasthan desert regions",
                "category": "Regional - Rajasthan"
            },
            {
                "query": "What water conservation techniques are used in Gujarat?",
                "category": "Regional - Gujarat"
            },
            {
                "query": "Which regions face the most severe groundwater over-exploitation?",
                "category": "Regional - Crisis Areas"
            },
            
            # Technical queries
            {
                "query": "What are the different types of aquifers found in India?",
                "category": "Technical - Aquifers"
            },
            {
                "query": "How deep is groundwater typically found in different regions?",
                "category": "Technical - Depth"
            },
            {
                "query": "What are the main groundwater quality issues in India?",
                "category": "Technical - Quality"
            },
            
            # Policy and management
            {
                "query": "What government schemes exist for groundwater management?",
                "category": "Policy - Schemes"
            },
            {
                "query": "What are the traditional water harvesting methods in India?",
                "category": "Policy - Traditional Methods"
            },
            {
                "query": "What regulations exist for groundwater extraction?",
                "category": "Policy - Regulations"
            },
            
            # Multi-language
            {
                "query": "भारत में भूजल की मुख्य समस्याएं क्या हैं?",
                "category": "Hindi - Problems"
            },
            {
                "query": "भारत में पारंपरिक जल संचयन के तरीके क्या हैं?",
                "category": "Hindi - Traditional Methods"
            }
        ]
        
        print(f"🚀 Starting comprehensive test of ingested data...")
        print(f"📊 Running {len(test_queries)} test queries...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test queries
        for i, test_config in enumerate(test_queries, 1):
            print(f"\n📈 Progress: {i}/{len(test_queries)} ({(i/len(test_queries))*100:.1f}%)")
            print("-" * 40)
            
            self.test_query(
                query=test_config["query"],
                category=test_config["category"]
            )
            
            # Brief pause between queries
            time.sleep(1)
        
        # Generate final report
        self.generate_test_report(start_time)
    
    def generate_test_report(self, start_time: float):
        """Generate comprehensive test report"""
        total_time = time.time() - start_time
        successful_tests = [t for t in self.test_results if t.get('success')]
        failed_tests = [t for t in self.test_results if not t.get('success')]
        
        print("\n" + "="*60)
        print("🎯 INGESTED DATA TEST REPORT")
        print("="*60)
        
        # Overall statistics
        total_tests = len(self.test_results)
        success_rate = (len(successful_tests) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Successful queries: {len(successful_tests)}/{total_tests}")
        print(f"   ❌ Failed queries: {len(failed_tests)}/{total_tests}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   ⏱️ Total test time: {total_time:.1f} seconds")
        
        if successful_tests:
            avg_response_time = sum(t.get('response_time', 0) for t in successful_tests) / len(successful_tests)
            avg_sources = sum(t.get('total_sources', 0) for t in successful_tests) / len(successful_tests)
            avg_answer_length = sum(t.get('answer_length', 0) for t in successful_tests) / len(successful_tests)
            
            print(f"   ⚡ Average response time: {avg_response_time:.2f} seconds")
            print(f"   📄 Average sources per query: {avg_sources:.1f}")
            print(f"   📝 Average answer length: {avg_answer_length:.0f} characters")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        if avg_response_time < 15:
            print(f"   🟢 Excellent response speed!")
        elif avg_response_time < 30:
            print(f"   🟡 Good response speed")
        else:
            print(f"   🔴 Response time could be improved")
        
        if success_rate >= 90:
            print(f"   🟢 Excellent success rate!")
        elif success_rate >= 70:
            print(f"   🟡 Good success rate")
        else:
            print(f"   🔴 Success rate needs improvement")
        
        if avg_sources >= 3:
            print(f"   🟢 Good source retrieval!")
        else:
            print(f"   🟡 Consider adding more documents for better coverage")
        
        # Category breakdown
        if successful_tests:
            print(f"\n📊 SUCCESS BY CATEGORY:")
            categories = {}
            for test in successful_tests:
                category = test.get('category', 'Unknown')
                categories[category] = categories.get(category, 0) + 1
            
            for category, count in categories.items():
                print(f"   ✅ {category}: {count} successful queries")
        
        # Language support
        languages = set()
        for test in successful_tests:
            lang = test.get('language_detected', 'unknown')
            languages.add(lang)
        
        if languages:
            print(f"\n🌐 LANGUAGE SUPPORT:")
            for lang in sorted(languages):
                lang_name = {"en": "English", "hi": "Hindi", "unknown": "Unknown"}.get(lang, lang)
                print(f"   🗣️ {lang_name} ({lang})")
        
        # Failed queries
        if failed_tests:
            print(f"\n❌ FAILED QUERIES:")
            for test in failed_tests:
                print(f"   📄 {test['category']}: {test.get('error', 'Unknown error')}")
        
        # System status
        print(f"\n🎯 SYSTEM STATUS:")
        if success_rate >= 80 and avg_response_time < 30:
            print(f"   🟢 Your SIH INGRES RAG system is working excellently!")
            print(f"   🚀 Ready for production use!")
        elif success_rate >= 60:
            print(f"   🟡 System is functional but could be improved")
            print(f"   🔧 Consider adding more documents or optimizing")
        else:
            print(f"   🔴 System needs attention")
            print(f"   🔧 Check ingested data and system configuration")
        
        # Next steps
        print(f"\n💡 NEXT STEPS:")
        if success_rate >= 80:
            print(f"   1. 🎉 Congratulations! Your system is working well")
            print(f"   2. 🚀 Deploy to production environment")
            print(f"   3. 📊 Monitor real user queries and performance")
            print(f"   4. 📈 Consider adding more specialized documents")
        else:
            print(f"   1. 🔍 Review failed queries and error messages")
            print(f"   2. 📄 Verify document ingestion was successful")
            print(f"   3. 🔧 Check system logs for any issues")
            print(f"   4. 🔄 Re-run bulk ingestion if needed")
        
        # Save test report
        self.save_test_report()
    
    def save_test_report(self):
        """Save detailed test report to file"""
        try:
            report = {
                "test_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_tests": len(self.test_results),
                    "successful_tests": len([t for t in self.test_results if t.get('success')]),
                    "failed_tests": len([t for t in self.test_results if not t.get('success')]),
                    "success_rate": (len([t for t in self.test_results if t.get('success')]) / len(self.test_results) * 100) if self.test_results else 0
                },
                "test_results": self.test_results
            }
            
            report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"   📋 Detailed test report saved: {report_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not save test report: {e}")

def main():
    """Main execution function"""
    print("🧪 SIH INGRES RAG - Ingested Data Testing Tool")
    print("=" * 60)
    print("This tool tests queries against your ingested groundwater documents")
    print("to verify that the bulk ingestion was successful.\n")
    
    # Initialize tester
    tester = IngestedDataTester()
    
    # Check system health first
    if not tester.check_system_health():
        print("\n❌ System health check failed!")
        print("   Please ensure your SIH server is running on port 8001")
        return
    
    print("\n⏳ Waiting 3 seconds before starting tests...")
    time.sleep(3)
    
    # Run comprehensive tests
    tester.run_comprehensive_tests()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
