#!/usr/bin/env python3
"""
Excel to SIH Ingestion Tool
===========================

This tool processes Excel (.xlsx) files and ingests them into your SIH INGRES RAG system.
It extracts text from all sheets and converts them into a format suitable for ingestion.

Usage: 
1. Add your Excel files to excel_sources.json
2. Run: python ingest_excel_files.py

Author: SIH INGRES Team
Date: September 2025
"""

import os
import time
import requests
import json
import pandas as pd
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

class ExcelToSIHIngestor:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = HEADERS
        self.ingested_docs = []
        self.failed_docs = []
        self.total_chunks = 0
        self.total_chars = 0
        
    def extract_text_from_excel_url(self, excel_url: str) -> str:
        """Download and extract text from Excel URL"""
        try:
            print(f"📥 Downloading Excel file from: {excel_url}")
            
            # Download the Excel file
            response = requests.get(excel_url, timeout=60)
            response.raise_for_status()
            
            # Save temporarily
            temp_file = "temp_excel.xlsx"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            text = self.extract_text_from_excel_file(temp_file)
            
            # Clean up
            os.remove(temp_file)
            
            return text
            
        except Exception as e:
            print(f"❌ Error downloading/processing Excel from URL: {e}")
            return None
    
    def extract_text_from_excel_file(self, excel_path: str) -> str:
        """Extract text content from Excel file"""
        try:
            print(f"📊 Extracting data from Excel file...")
            
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(excel_path)
            all_text_parts = []
            
            print(f"   📄 Found {len(excel_file.sheet_names)} sheets: {', '.join(excel_file.sheet_names)}")
            
            for sheet_name in excel_file.sheet_names:
                print(f"   🔍 Processing sheet: {sheet_name}")
                
                # Read the sheet
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                # Add sheet header
                all_text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")
                
                # Convert DataFrame to text
                sheet_text = self.dataframe_to_text(df, sheet_name)
                all_text_parts.append(sheet_text)
                
                print(f"      ✅ Extracted {len(sheet_text)} characters from {sheet_name}")
            
            # Combine all text
            full_text = "\n".join(all_text_parts)
            
            print(f"   📊 Total extracted text: {len(full_text):,} characters")
            return full_text
            
        except Exception as e:
            print(f"❌ Error extracting text from Excel: {e}")
            return None
    
    def dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convert pandas DataFrame to readable text"""
        try:
            text_parts = []
            
            # Add basic info about the data
            text_parts.append(f"Data from sheet '{sheet_name}':")
            text_parts.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            text_parts.append(f"Column names: {', '.join(df.columns.astype(str))}")
            text_parts.append("")
            
            # Convert data to text format
            if len(df) > 0:
                # Method 1: Row by row description (good for structured data)
                if len(df) <= 1000:  # For smaller datasets, include all rows
                    for idx, row in df.iterrows():
                        row_text = []
                        for col in df.columns:
                            value = row[col]
                            if pd.notna(value):  # Skip NaN values
                                row_text.append(f"{col}: {value}")
                        
                        if row_text:  # Only add if there's content
                            text_parts.append(f"Row {idx + 1}: {', '.join(row_text)}")
                
                else:  # For larger datasets, provide summary
                    # Add first 100 rows
                    text_parts.append("First 100 rows of data:")
                    for idx, row in df.head(100).iterrows():
                        row_text = []
                        for col in df.columns:
                            value = row[col]
                            if pd.notna(value):
                                row_text.append(f"{col}: {value}")
                        
                        if row_text:
                            text_parts.append(f"Row {idx + 1}: {', '.join(row_text)}")
                    
                    # Add summary statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        text_parts.append("\nSummary Statistics for Numeric Columns:")
                        for col in numeric_cols:
                            stats = df[col].describe()
                            text_parts.append(f"{col}: Mean={stats['mean']:.2f}, Min={stats['min']}, Max={stats['max']}, Count={stats['count']}")
                    
                    # Add value counts for categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        if df[col].nunique() < 20:  # Only for columns with few unique values
                            value_counts = df[col].value_counts().head(10)
                            text_parts.append(f"\nTop values in {col}:")
                            for value, count in value_counts.items():
                                text_parts.append(f"  {value}: {count} occurrences")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"❌ Error converting DataFrame to text: {e}")
            return f"Error processing sheet {sheet_name}: {str(e)}"
    
    def ingest_excel_document(self, doc_config: Dict[str, Any]) -> bool:
        """Process and ingest an Excel document"""
        try:
            title = doc_config.get("title", "Unknown Excel Document")
            print(f"\n📊 Processing Excel Document: {title}")
            
            # Extract text from Excel
            if doc_config.get("url"):
                # Download from URL
                text = self.extract_text_from_excel_url(doc_config["url"])
            elif doc_config.get("local_path"):
                # Load from local file
                text = self.extract_text_from_excel_file(doc_config["local_path"])
            else:
                print("❌ No URL or local_path specified")
                return False
            
            if not text:
                print("❌ Failed to extract text from Excel file")
                return False
            
            # Prepare payload for SIH /ingest endpoint
            payload = {
                "text": text,  # Use extracted text directly
                "title": title,
                "region_meta": doc_config.get("region_meta", {})
            }
            
            print(f"📤 Sending to SIH system...")
            print(f"   📝 Text length: {len(text):,} characters")
            
            # Send to /ingest endpoint
            response = requests.post(
                f"{self.base_url}/ingest",
                json=payload,
                headers=self.headers,
                timeout=300  # 5 minutes for large Excel files
            )
            
            if response.status_code == 200:
                result = response.json()
                
                chunks_processed = result.get('chunks_processed', 0)
                total_characters = result.get('total_characters', 0)
                doc_id = result.get('doc_id', 'unknown')
                
                print(f"✅ Excel file successfully ingested!")
                print(f"   📄 Document ID: {doc_id}")
                print(f"   🧩 Chunks created: {chunks_processed}")
                print(f"   📝 Characters processed: {total_characters:,}")
                
                # Store success info
                self.ingested_docs.append({
                    "title": title,
                    "source": doc_config.get("url") or doc_config.get("local_path", ""),
                    "doc_id": doc_id,
                    "chunks": chunks_processed,
                    "characters": total_characters,
                    "region_meta": doc_config.get("region_meta", {}),
                    "file_type": "xlsx",
                    "ingested_at": datetime.now().isoformat()
                })
                
                self.total_chunks += chunks_processed
                self.total_chars += total_characters
                
                return True
            else:
                error_msg = response.text[:200] + "..." if len(response.text) > 200 else response.text
                print(f"❌ SIH ingestion failed: HTTP {response.status_code}")
                print(f"   Error: {error_msg}")
                
                self.failed_docs.append({
                    "title": title,
                    "source": doc_config.get("url") or doc_config.get("local_path", ""),
                    "error": f"HTTP {response.status_code}: {error_msg}",
                    "file_type": "xlsx",
                    "failed_at": datetime.now().isoformat()
                })
                
                return False
                
        except Exception as e:
            print(f"❌ Error processing Excel document: {e}")
            self.failed_docs.append({
                "title": doc_config.get("title", "Unknown"),
                "source": doc_config.get("url") or doc_config.get("local_path", ""),
                "error": str(e),
                "file_type": "xlsx",
                "failed_at": datetime.now().isoformat()
            })
            return False
    
    def ingest_from_config(self, config_file: str = "excel_sources.json"):
        """Ingest all Excel files from configuration"""
        try:
            if not os.path.exists(config_file):
                print(f"❌ Configuration file '{config_file}' not found!")
                self.create_sample_config(config_file)
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            documents = config.get('documents', [])
            if not documents:
                print("❌ No documents found in configuration file!")
                return False
            
            total_docs = len(documents)
            print(f"📊 Found {total_docs} Excel documents to process")
            
            # Check SIH system health
            if not self.check_sih_health():
                return False
            
            print(f"\n🚀 Starting Excel file ingestion...")
            print("=" * 60)
            
            start_time = time.time()
            
            for i, doc in enumerate(documents, 1):
                print(f"\n📈 Progress: {i}/{total_docs} ({(i/total_docs)*100:.1f}%)")
                print("-" * 40)
                
                success = self.ingest_excel_document(doc)
                
                if success:
                    print(f"   ⏱️ Elapsed time: {time.time() - start_time:.1f}s")
                
                # Wait between documents
                if i < total_docs:
                    print("   ⏳ Waiting 3 seconds before next document...")
                    time.sleep(3)
            
            # Final summary
            self.print_summary(start_time)
            self.save_report()
            
            return len(self.ingested_docs) > 0
            
        except Exception as e:
            print(f"❌ Excel ingestion failed: {e}")
            return False
    
    def check_sih_health(self):
        """Check if SIH system is healthy"""
        try:
            print("🏥 Checking SIH system health...")
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                print("✅ SIH system is healthy!")
                return True
            else:
                print(f"❌ SIH health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to SIH system. Is it running on port 8001?")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def create_sample_config(self, config_file: str):
        """Create sample configuration file for Excel ingestion"""
        sample_config = {
            "documents": [
                {
                    "url": "https://example.com/maharashtra-groundwater-data.xlsx",
                    "title": "Maharashtra Groundwater Data 2023",
                    "region_meta": {
                        "state": "Maharashtra",
                        "year": 2023,
                        "data_type": "groundwater_measurements",
                        "file_format": "xlsx"
                    }
                },
                {
                    "local_path": "./data/rajasthan-well-data.xlsx",
                    "title": "Rajasthan Well Survey Data",
                    "region_meta": {
                        "state": "Rajasthan",
                        "year": 2023,
                        "data_type": "well_survey",
                        "file_format": "xlsx"
                    }
                }
            ],
            "settings": {
                "description": "Configuration for Excel file ingestion into SIH system",
                "supported_formats": [".xlsx", ".xls"],
                "instructions": [
                    "Add either 'url' for online Excel files or 'local_path' for local files",
                    "Update titles and region_meta with accurate information",
                    "Ensure URLs are publicly accessible or files exist locally",
                    "Run: python ingest_excel_files.py"
                ]
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Sample configuration created: {config_file}")
        print("📝 Please edit this file with your actual Excel files and run again.")
    
    def print_summary(self, start_time: float):
        """Print ingestion summary"""
        total_time = time.time() - start_time
        successful_docs = len(self.ingested_docs)
        failed_docs = len(self.failed_docs)
        total_docs = successful_docs + failed_docs
        
        print("\n" + "="*60)
        print("📊 EXCEL INGESTION SUMMARY")
        print("="*60)
        
        print(f"✅ Successfully processed: {successful_docs}/{total_docs} Excel files")
        print(f"❌ Failed to process: {failed_docs}/{total_docs} Excel files")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        
        if successful_docs > 0:
            print(f"🧩 Total chunks created: {self.total_chunks:,}")
            print(f"📝 Total characters processed: {self.total_chars:,}")
            
            print(f"\n✅ Successfully processed Excel files:")
            for doc in self.ingested_docs:
                print(f"   📊 {doc['title']}: {doc['chunks']} chunks")
        
        if failed_docs > 0:
            print(f"\n❌ Failed Excel files:")
            for doc in self.failed_docs:
                print(f"   📊 {doc['title']}: {doc['error']}")
    
    def save_report(self):
        """Save detailed report"""
        try:
            report = {
                "summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_files": len(self.ingested_docs) + len(self.failed_docs),
                    "successful_files": len(self.ingested_docs),
                    "failed_files": len(self.failed_docs),
                    "total_chunks": self.total_chunks,
                    "total_characters": self.total_chars
                },
                "successful_files": self.ingested_docs,
                "failed_files": self.failed_docs
            }
            
            report_file = f"excel_ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Detailed report saved: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")

def main():
    """Main execution function"""
    print("📊 SIH INGRES RAG - Excel File Ingestion Tool")
    print("=" * 60)
    print("This tool processes Excel (.xlsx) files and ingests them into your SIH system.\n")
    
    # Check if pandas is installed
    try:
        import pandas as pd
        print("✅ pandas library available")
    except ImportError:
        print("❌ pandas library not found!")
        print("   Install with: pip install pandas openpyxl")
        return
    
    # Initialize ingestor
    ingestor = ExcelToSIHIngestor()
    
    # Start ingestion
    success = ingestor.ingest_from_config("excel_sources.json")
    
    if success:
        print("\n🎉 Excel ingestion completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Excel ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
