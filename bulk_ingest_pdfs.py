#!/usr/bin/env python3
"""
Bulk PDF Ingestion for SIH INGRES RAG System
============================================

This script allows you to ingest multiple PDF documents at once into your SIH system.
It uses the existing /ingest endpoint and processes documents one by one efficiently.

Usage:
1. Edit pdf_sources.json with your PDF URLs/metadata
2. Run: python bulk_ingest_pdfs.py
3. Monitor progress and results

Author: SIH INGRES Team
Date: September 2025
"""

import os
import time
import requests
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:8001"
AUTH_TOKEN = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

class BulkPDFIngestor:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = HEADERS
        self.ingested_docs = []
        self.failed_docs = []
        self.total_chunks = 0
        self.total_chars = 0
        
    def check_server_health(self):
        """Check if SIH server is running and healthy"""
        try:
            print("🏥 Checking server health...")
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                services = health_data.get('services', {})
                
                print("✅ Server is healthy!")
                print(f"   - Pinecone: {'✅' if services.get('pinecone') else '❌'}")
                print(f"   - Groq LLM: {'✅' if services.get('groq') else '❌'}")
                print(f"   - Jina Embeddings: {'✅' if services.get('jina') else '❌'}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Is it running on port 8001?")
            print("   Start server with: uvicorn sih:app --host 0.0.0.0 --port 8001")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def ingest_single_document(self, doc_config: Dict[str, Any]) -> bool:
        """Ingest a single document using /ingest endpoint"""
        try:
            # Prepare request payload
            payload = {
                "source_url": doc_config.get("url"),
                "title": doc_config.get("title", "Unknown Document"),
                "text": doc_config.get("text"),  # For direct text input
                "region_meta": doc_config.get("region_meta", {})
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            print(f"📄 Ingesting: {payload['title']}")
            if payload.get('source_url'):
                print(f"   📂 Source: {payload['source_url']}")
            
            # Send request to /ingest endpoint
            response = requests.post(
                f"{self.base_url}/ingest",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minutes for large documents
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Track success metrics
                chunks_processed = result.get('chunks_processed', 0)
                total_characters = result.get('total_characters', 0)
                doc_id = result.get('doc_id', 'unknown')
                
                print(f"✅ Success!")
                print(f"   📄 Document ID: {doc_id}")
                print(f"   🧩 Chunks created: {chunks_processed}")
                print(f"   📝 Characters processed: {total_characters:,}")
                
                # Store success info
                self.ingested_docs.append({
                    "title": payload['title'],
                    "url": payload.get('source_url', ''),
                    "doc_id": doc_id,
                    "chunks": chunks_processed,
                    "characters": total_characters,
                    "region_meta": doc_config.get("region_meta", {}),
                    "ingested_at": datetime.now().isoformat()
                })
                
                self.total_chunks += chunks_processed
                self.total_chars += total_characters
                
                return True
                
            else:
                error_msg = response.text[:200] + "..." if len(response.text) > 200 else response.text
                print(f"❌ Failed: HTTP {response.status_code}")
                print(f"   Error: {error_msg}")
                
                self.failed_docs.append({
                    "title": payload['title'],
                    "url": payload.get('source_url', ''),
                    "error": f"HTTP {response.status_code}: {error_msg}",
                    "failed_at": datetime.now().isoformat()
                })
                
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out (document might be too large)")
            self.failed_docs.append({
                "title": doc_config.get("title", "Unknown"),
                "url": doc_config.get("url", ""),
                "error": "Request timeout",
                "failed_at": datetime.now().isoformat()
            })
            return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.failed_docs.append({
                "title": doc_config.get("title", "Unknown"),
                "url": doc_config.get("url", ""),
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
            return False
    
    def ingest_from_config(self, config_file: str = "pdf_sources.json"):
        """Ingest all documents from configuration file"""
        try:
            # Load configuration
            if not os.path.exists(config_file):
                print(f"❌ Configuration file '{config_file}' not found!")
                print("   Please create this file with your PDF sources.")
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            documents = config.get('documents', [])
            if not documents:
                print("❌ No documents found in configuration file!")
                return False
            
            total_docs = len(documents)
            print(f"📚 Found {total_docs} documents to ingest")
            
            # Check server health before starting
            if not self.check_server_health():
                return False
            
            print(f"\n🚀 Starting bulk ingestion...")
            print("=" * 60)
            
            start_time = time.time()
            
            # Process each document
            for i, doc in enumerate(documents, 1):
                print(f"\n📊 Progress: {i}/{total_docs} ({(i/total_docs)*100:.1f}%)")
                print("-" * 40)
                
                success = self.ingest_single_document(doc)
                
                if success:
                    print(f"   ⏱️ Elapsed time: {time.time() - start_time:.1f}s")
                
                # Small delay between documents to avoid overwhelming the system
                if i < total_docs:  # Don't wait after the last document
                    print("   ⏳ Waiting 3 seconds before next document...")
                    time.sleep(3)
            
            # Final summary
            self.print_final_summary(start_time)
            
            # Save detailed results
            self.save_ingestion_report()
            
            return len(self.ingested_docs) > 0
            
        except FileNotFoundError:
            print(f"❌ Configuration file '{config_file}' not found!")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            return False
        except Exception as e:
            print(f"❌ Bulk ingestion failed: {e}")
            return False
    
    def print_final_summary(self, start_time: float):
        """Print comprehensive final summary"""
        total_time = time.time() - start_time
        successful_docs = len(self.ingested_docs)
        failed_docs = len(self.failed_docs)
        total_docs = successful_docs + failed_docs
        
        print("\n" + "="*60)
        print("🎯 BULK INGESTION COMPLETE")
        print("="*60)
        
        # Overall statistics
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Successfully ingested: {successful_docs}/{total_docs} documents")
        print(f"   ❌ Failed to ingest: {failed_docs}/{total_docs} documents")
        print(f"   📈 Success rate: {(successful_docs/total_docs)*100:.1f}%" if total_docs > 0 else "   📈 Success rate: 0%")
        print(f"   ⏱️ Total processing time: {total_time/60:.1f} minutes")
        print(f"   ⚡ Average time per document: {total_time/total_docs:.1f}s" if total_docs > 0 else "   ⚡ Average time per document: 0s")
        
        # Content statistics
        if successful_docs > 0:
            print(f"\n📄 CONTENT PROCESSED:")
            print(f"   🧩 Total chunks created: {self.total_chunks:,}")
            print(f"   📝 Total characters processed: {self.total_chars:,}")
            print(f"   📊 Average chunks per document: {self.total_chunks/successful_docs:.1f}")
            print(f"   📊 Average characters per document: {self.total_chars/successful_docs:,}")
        
        # Successful documents
        if self.ingested_docs:
            print(f"\n✅ SUCCESSFULLY INGESTED DOCUMENTS:")
            for doc in self.ingested_docs:
                region_info = []
                for key, value in doc.get('region_meta', {}).items():
                    if key != 'ingested_at':
                        region_info.append(f"{key}: {value}")
                region_str = f" ({', '.join(region_info)})" if region_info else ""
                print(f"   📄 {doc['title']}: {doc['chunks']} chunks{region_str}")
        
        # Failed documents
        if self.failed_docs:
            print(f"\n❌ FAILED DOCUMENTS:")
            for doc in self.failed_docs:
                print(f"   📄 {doc['title']}: {doc['error']}")
        
        # Next steps
        print(f"\n💡 NEXT STEPS:")
        if successful_docs > 0:
            print(f"   1. 🔍 Test queries against your data:")
            print(f"      python test_ingested_data.py")
            print(f"   2. 🌐 Your knowledge base now has {self.total_chunks:,} searchable chunks")
            print(f"   3. 🚀 Start querying: POST /query with your questions")
            
            if failed_docs > 0:
                print(f"   4. 🔧 Review and retry failed documents")
        else:
            print(f"   1. 🔧 Check server logs and fix any issues")
            print(f"   2. 🔍 Verify PDF URLs are accessible")
            print(f"   3. 🔄 Retry ingestion after fixing issues")
        
        # System status
        print(f"\n🎯 SYSTEM STATUS:")
        if successful_docs > 0:
            print(f"   🟢 Your SIH INGRES RAG system is ready for queries!")
        elif failed_docs == total_docs:
            print(f"   🔴 All documents failed - check configuration and server")
        else:
            print(f"   🟡 Partial success - review failed documents")
    
    def save_ingestion_report(self):
        """Save detailed ingestion report to file"""
        try:
            report = {
                "ingestion_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_documents": len(self.ingested_docs) + len(self.failed_docs),
                    "successful_documents": len(self.ingested_docs),
                    "failed_documents": len(self.failed_docs),
                    "total_chunks": self.total_chunks,
                    "total_characters": self.total_chars
                },
                "successful_documents": self.ingested_docs,
                "failed_documents": self.failed_docs
            }
            
            report_file = f"ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"   📋 Detailed report saved: {report_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not save report: {e}")

def main():
    """Main execution function"""
    print("🚀 SIH INGRES RAG - Bulk PDF Ingestion Tool")
    print("=" * 60)
    print("This tool will ingest multiple PDF documents into your SIH system.")
    print("Each document will be processed using the /ingest endpoint.\n")
    
    # Initialize ingestor
    ingestor = BulkPDFIngestor()
    
    # Check if configuration file exists
    config_file = "pdf_sources.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found!")
        print("\n🔧 Creating sample configuration file...")
        
        # Create sample configuration
        sample_config = {
            "documents": [
                {
                    "url": "https://example.com/maharashtra-groundwater-report-2023.pdf",
                    "title": "Maharashtra Groundwater Assessment 2023",
                    "region_meta": {
                        "state": "Maharashtra",
                        "year": 2023,
                        "report_type": "groundwater_assessment",
                        "agency": "CGWB"
                    }
                },
                {
                    "url": "https://example.com/rajasthan-aquifer-study.pdf",
                    "title": "Rajasthan Aquifer Management Study",
                    "region_meta": {
                        "state": "Rajasthan",
                        "year": 2023,
                        "report_type": "aquifer_study",
                        "agency": "State Water Board"
                    }
                }
            ]
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Sample configuration created: {config_file}")
        print("📝 Please edit this file with your actual PDF URLs and run again.")
        return
    
    # Start bulk ingestion
    success = ingestor.ingest_from_config(config_file)
    
    if success:
        print("\n🎉 Bulk ingestion completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Bulk ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
