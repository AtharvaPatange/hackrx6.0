#!/usr/bin/env python3
"""
Excel-Only Uploader for SIH RAG System using Docling
Specialized for Excel files (.xlsx, .xls) with advanced table extraction
"""

import os
import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Docling imports for advanced Excel processing
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
    print("✅ Docling available for advanced Excel processing")
except ImportError:
    DOCLING_AVAILABLE = False
    print("❌ Docling not available - install with: pip install docling")
    exit(1)

# Configuration
BASE_URL = "http://localhost:8001"
HEADERS = {"Content-Type": "application/json"}

class ExcelDoclingUploader:
    def __init__(self):
        self.converter = DocumentConverter()
        self.processed_files = []
        self.failed_files = []
        self.total_chunks_created = 0
        
    def extract_excel_with_docling(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract structured data from Excel using Docling"""
        try:
            print(f"🔬 Processing Excel with Docling: {Path(file_path).name}")
            
            # Convert Excel document
            result = self.converter.convert(file_path)
            doc = result.document
            
            extracted_items = []
            table_count = 0
            text_count = 0
            
            # Iterate through document items
            for item, level in doc.iterate_items():
                
                if item.label == "table":
                    table_count += 1
                    # Convert table to markdown for better readability
                    table_md = item.export_to_markdown()
                    
                    # Also try to get structured table data
                    table_data = self.process_table_item(item, table_count)
                    
                    extracted_items.append({
                        "content": table_md,
                        "structured_data": table_data,
                        "metadata": {
                            "type": "table",
                            "table_number": table_count,
                            "level": level,
                            "source": "docling",
                            "format": "markdown"
                        }
                    })
                    
                    print(f"   📊 Table {table_count}: {len(table_md)} chars, {table_data.get('rows', 0)} rows")
                
                elif item.label in ["text", "paragraph"]:
                    if len(item.text.strip()) > 30:  # Only meaningful text
                        text_count += 1
                        extracted_items.append({
                            "content": item.text.strip(),
                            "metadata": {
                                "type": "text",
                                "text_number": text_count,
                                "level": level,
                                "source": "docling"
                            }
                        })
                
                elif item.label in ["heading", "title"]:
                    extracted_items.append({
                        "content": f"# {item.text}",
                        "metadata": {
                            "type": "heading",
                            "level": level,
                            "source": "docling"
                        }
                    })
            
            print(f"   ✅ Docling extracted: {table_count} tables, {text_count} text blocks")
            return extracted_items
            
        except Exception as e:
            print(f"   ❌ Docling extraction failed: {e}")
            return []
    
    def process_table_item(self, table_item, table_number: int) -> Dict[str, Any]:
        """Extract structured information from table item"""
        try:
            # Get table as markdown first
            table_md = table_item.export_to_markdown()
            
            # Parse basic table structure
            lines = table_md.split('\n')
            data_lines = [line for line in lines if line.strip() and not line.strip().startswith('|--')]
            
            if len(data_lines) > 1:
                # Extract headers
                header_line = data_lines[0]
                headers = [h.strip() for h in header_line.split('|') if h.strip()]
                
                # Count data rows
                data_rows = len(data_lines) - 1
                
                return {
                    "table_id": f"table_{table_number}",
                    "headers": headers,
                    "rows": data_rows,
                    "columns": len(headers),
                    "markdown": table_md
                }
            
        except Exception as e:
            print(f"      ⚠️  Table processing error: {e}")
        
        return {"table_id": f"table_{table_number}", "markdown": table_item.export_to_markdown()}
    
    def create_enhanced_chunks(self, extracted_items: List[Dict], file_info: Dict) -> List[Dict]:
        """Create enhanced chunks from extracted items"""
        chunks = []
        
        for i, item in enumerate(extracted_items, 1):
            content = item["content"]
            metadata = item["metadata"]
            
            # For tables, create comprehensive searchable text
            if metadata["type"] == "table":
                structured_data = item.get("structured_data", {})
                
                # Create searchable text version
                searchable_text = f"""
Table {metadata.get('table_number', i)} from {file_info['filename']}:

{content}

Table Summary:
- Columns: {structured_data.get('columns', 'Unknown')}
- Rows: {structured_data.get('rows', 'Unknown')}
- Headers: {', '.join(structured_data.get('headers', []))}

This table contains structured data that can be queried for specific values, comparisons, and analysis.
"""
                
                # Split large tables if needed
                if len(searchable_text) > 800:
                    table_chunks = self.chunk_large_content(searchable_text, f"Table {metadata.get('table_number', i)}")
                    for chunk_idx, chunk_text in enumerate(table_chunks, 1):
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "filename": file_info["filename"],
                                "file_type": file_info["file_type"],
                                "chunk_part": chunk_idx,
                                "total_parts": len(table_chunks),
                                "processed_at": datetime.now().isoformat(),
                                **structured_data
                            }
                        })
                else:
                    chunks.append({
                        "text": searchable_text,
                        "metadata": {
                            **metadata,
                            "filename": file_info["filename"],
                            "file_type": file_info["file_type"],
                            "processed_at": datetime.now().isoformat(),
                            **structured_data
                        }
                    })
            
            else:
                # For text content
                if len(content) > 800:
                    text_chunks = self.chunk_large_content(content, f"Text section {i}")
                    for chunk_idx, chunk_text in enumerate(text_chunks, 1):
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "filename": file_info["filename"],
                                "file_type": file_info["file_type"],
                                "chunk_part": chunk_idx,
                                "total_parts": len(text_chunks),
                                "processed_at": datetime.now().isoformat()
                            }
                        })
                else:
                    chunks.append({
                        "text": content,
                        "metadata": {
                            **metadata,
                            "filename": file_info["filename"],
                            "file_type": file_info["file_type"],
                            "processed_at": datetime.now().isoformat()
                        }
                    })
        
        return chunks
    
    def chunk_large_content(self, text: str, content_name: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Chunk large content intelligently"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        chunk_num = 1
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at natural boundaries
            if end < len(text):
                # Look for sentence endings
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size * 0.6:  # Don't break too early
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            # Add chunk metadata
            chunk_with_context = f"{content_name} (Part {chunk_num}):\n\n{chunk.strip()}"
            chunks.append(chunk_with_context)
            
            start = end - overlap
            chunk_num += 1
        
        return chunks
    
    def upload_chunks_to_sih(self, chunks: List[Dict], doc_title: str) -> bool:
        """Upload processed chunks to SIH system"""
        print(f"   📤 Uploading {len(chunks)} chunks to SIH...")
        
        successful_uploads = 0
        total_vectors_created = 0
        
        for i, chunk_data in enumerate(chunks, 1):
            print(f"      📋 Chunk {i}/{len(chunks)}...", end=' ')
            
            # Prepare payload for SIH API
            payload = {
                "text": chunk_data["text"],
                "title": f"{doc_title} - Section {i}",
                "region_meta": chunk_data["metadata"]
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/ingest",
                    json=payload,
                    headers=HEADERS,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    vectors_created = result.get('chunks_processed', 0)
                    successful_uploads += 1
                    total_vectors_created += vectors_created
                    print(f"✅ {vectors_created} vectors")
                else:
                    print(f"❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
        
        success_rate = successful_uploads / len(chunks)
        print(f"   📊 Upload complete: {successful_uploads}/{len(chunks)} chunks ({success_rate*100:.1f}%)")
        print(f"   🧩 Total vectors created: {total_vectors_created}")
        
        self.total_chunks_created += total_vectors_created
        return success_rate >= 0.8
    
    def process_excel_file(self, file_path: str, title: str = None, metadata: Dict = None) -> bool:
        """Process a single Excel file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            print(f"❌ Not an Excel file: {file_path}")
            return False
        
        print(f"\n📊 Processing Excel file: {file_path.name}")
        print(f"   📁 Path: {file_path}")
        print(f"   📏 Size: {file_path.stat().st_size / 1024:.1f} KB")
        
        # Extract data using Docling
        extracted_items = self.extract_excel_with_docling(str(file_path))
        
        if not extracted_items:
            print(f"   ❌ No data extracted from {file_path.name}")
            self.failed_files.append(str(file_path))
            return False
        
        # Prepare file information
        file_info = {
            "filename": file_path.name,
            "file_type": "excel",
            "file_size": file_path.stat().st_size,
            "processed_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Create enhanced chunks
        chunks = self.create_enhanced_chunks(extracted_items, file_info)
        print(f"   📦 Created {len(chunks)} uploadable chunks")
        
        # Upload to SIH
        doc_title = title or file_path.stem
        success = self.upload_chunks_to_sih(chunks, doc_title)
        
        if success:
            print(f"   ✅ Successfully processed: {file_path.name}")
            self.processed_files.append(str(file_path))
            return True
        else:
            print(f"   ❌ Failed to process: {file_path.name}")
            self.failed_files.append(str(file_path))
            return False
    
    def process_multiple_excel_files(self, file_config: List[Dict]):
        """Process multiple Excel files from configuration"""
        print("📊 EXCEL DOCLING UPLOADER FOR SIH RAG")
        print("=" * 60)
        print(f"📚 Processing {len(file_config)} Excel files")
        print("🔬 Using Docling for advanced table extraction")
        
        successful_files = 0
        
        for i, config in enumerate(file_config, 1):
            print(f"\n🎯 Progress: {i}/{len(file_config)} ({i/len(file_config)*100:.1f}%)")
            print("-" * 40)
            
            success = self.process_excel_file(
                config["path"],
                config.get("title"),
                config.get("metadata", {})
            )
            
            if success:
                successful_files += 1
        
        # Print final summary
        self.print_final_summary(successful_files, len(file_config))
    
    def print_final_summary(self, successful: int, total: int):
        """Print comprehensive processing summary"""
        print("\n" + "=" * 60)
        print("🎯 EXCEL PROCESSING COMPLETE")
        print("=" * 60)
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Successfully processed: {successful}/{total} files")
        print(f"   ❌ Failed: {len(self.failed_files)} files")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   🧩 Total vector chunks created: {self.total_chunks_created}")
        
        if self.processed_files:
            print(f"\n✅ SUCCESSFULLY PROCESSED FILES:")
            for file_path in self.processed_files:
                print(f"   📊 {Path(file_path).name}")
        
        if self.failed_files:
            print(f"\n❌ FAILED FILES:")
            for file_path in self.failed_files:
                print(f"   📊 {Path(file_path).name}")
        
        if success_rate >= 90:
            print(f"\n🏆 EXCELLENT: All Excel data successfully uploaded to RAG!")
        elif success_rate >= 70:
            print(f"\n🎉 GOOD: Most Excel data uploaded successfully!")
        else:
            print(f"\n🔧 NEEDS ATTENTION: Check failed files and retry")
        
        print(f"\n🧪 TESTING SUGGESTIONS:")
        print(f"   1. Query specific table data and values")
        print(f"   2. Ask for comparisons across different sheets")
        print(f"   3. Search for specific column headers or data points")
        print(f"   4. Request summaries of tabular information")

def main():
    """Main execution with sample Excel file configuration"""
    
    if not DOCLING_AVAILABLE:
        print("❌ Please install Docling first: pip install docling")
        return
    
    uploader = ExcelDoclingUploader()
    
    # Sample Excel files configuration
    excel_files_config = [
        {
            "path": r"D:\HackrX\2024-25.xlsx",
            "title": "Groundwater Monitoring Data 2023",
            "metadata": {
                "data_type": "measurements",
                "year": 2023,
                "agency": "CGWB",
                "coverage": "multi_state",
                "parameters": ["water_level", "water_quality", "aquifer_data"]
            }
        },
        {
            "path": r"D:\HackrX\2023-24.xlsx",
            "title": "Andhra Pradesh Groundwater Resources",
            "metadata": {
                "state": "Andhra Pradesh",
                "data_type": "resource_assessment", 
                "year": 2023,
                "includes_districts": True
            }
        }
    ]
    
    print("📋 CONFIGURED EXCEL FILES:")
    for i, config in enumerate(excel_files_config, 1):
        print(f"   {i}. {config['title']}")
        print(f"      📁 Path: {config['path']}")
        print(f"      📊 Type: {config['metadata'].get('data_type', 'data')}")
        if not Path(config['path']).exists():
            print(f"      ⚠️  File not found - will be skipped")
    
    # Filter out non-existent files
    existing_files = [config for config in excel_files_config if Path(config['path']).exists()]
    
    if not existing_files:
        print("\n❌ No Excel files found!")
        print("\n📝 TO USE THIS UPLOADER:")
        print("   1. Place your Excel files in the ./data/ directory")
        print("   2. Update the excel_files_config list above")
        print("   3. Run the script again")
        return
    
    print(f"\n🚀 Found {len(existing_files)} Excel files to process")
    
    # Process the Excel files
    uploader.process_multiple_excel_files(existing_files)
    
    print(f"\n🎉 Excel processing complete!")
    print(f"📊 Your RAG system now has enhanced Excel data with proper table structure!")

if __name__ == "__main__":
    main()