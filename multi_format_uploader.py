#!/usr/bin/env python3
"""
Multi-Format Data Uploader for SIH Groundwater RAG System
Supports PDF, DOCX, XLSX files and Google Drive URLs
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

# Check if optional libraries are available
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Configuration
BASE_URL = "http://localhost:8001"  # Your SIH server
AUTH_TOKEN = "b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
HEADERS = {
    "Content-Type": "application/json"
}

class MultiFormatDataUploader:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.headers = HEADERS
        self.upload_results = []
        self.failed_uploads = []
        
    def convert_google_drive_url(self, drive_url: str) -> str:
        """Convert Google Drive sharing URL to direct download URL"""
        # Extract file ID from Google Drive URL
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                file_id = match.group(1)
                direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                print(f"   🔄 Converted Google Drive URL")
                print(f"      Original: {drive_url[:50]}...")
                print(f"      Direct: {direct_url}")
                return direct_url
        
        print(f"   ⚠️  Could not convert Google Drive URL, using original")
        return drive_url
    
    def convert_google_sheets_url(self, sheets_url: str) -> str:
        """Convert Google Sheets URL to CSV export URL"""
        # Extract spreadsheet ID
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheets_url)
        if match:
            sheet_id = match.group(1)
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            print(f"   🔄 Converted Google Sheets to CSV export URL")
            return csv_url
        
        print(f"   ⚠️  Could not convert Google Sheets URL, using original")
        return sheets_url
    
    def download_from_url(self, url: str, title: str) -> Optional[str]:
        """Download content from URL and extract text"""
        try:
            print(f"   📡 Downloading from URL...")
            
            # Convert Google URLs
            if "drive.google.com" in url:
                url = self.convert_google_drive_url(url)
            elif "docs.google.com/spreadsheets" in url:
                url = self.convert_google_sheets_url(url)
            
            # Download with appropriate headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                print(f"   📄 Downloaded {len(response.content)} bytes")
                print(f"   📋 Content type: {content_type}")
                
                # Handle different content types
                if 'pdf' in content_type or url.endswith('.pdf'):
                    return self.extract_pdf_from_bytes(response.content, title)
                elif 'csv' in content_type or 'spreadsheet' in content_type:
                    return self.extract_csv_from_text(response.text, title)
                elif 'html' in content_type:
                    # Sometimes Google Drive returns HTML, try to extract
                    return self.extract_text_from_html(response.text, title)
                else:
                    # Try as plain text
                    return response.text
            else:
                print(f"   ❌ Download failed: HTTP {response.status_code}")
                print(f"      Response: {response.text[:200]}...")
                return None
                
        except Exception as e:
            print(f"   ❌ Download error: {e}")
            return None
    
    def extract_pdf_from_bytes(self, pdf_bytes: bytes, title: str) -> Optional[str]:
        """Extract text from PDF bytes with better cleaning"""
        if not PDF_SUPPORT:
            print(f"   ⚠️  PyPDF2 not installed, cannot extract PDF")
            return None
            
        try:
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = f"Document: {title}\n\n"
            total_pages = len(pdf_reader.pages)
            extracted_chars = 0
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    # Clean the extracted text
                    if page_text:
                        # Remove PDF artifacts and binary content
                        cleaned_text = self.clean_pdf_text(page_text)
                        
                        if cleaned_text and len(cleaned_text.strip()) > 50:  # Only meaningful text
                            text += f"\n--- Page {page_num}/{total_pages} ---\n"
                            text += cleaned_text + "\n"
                            extracted_chars += len(cleaned_text)
                    
                except Exception as page_error:
                    print(f"      ⚠️  Error on page {page_num}: {page_error}")
                    continue
            
            if extracted_chars > 1000:  # Only return if we got meaningful content
                print(f"   📄 Extracted {extracted_chars:,} characters from {total_pages} pages")
                return text
            else:
                print(f"   ⚠️  No meaningful text extracted from PDF")
                return None
            
        except Exception as e:
            print(f"   ❌ PDF extraction error: {e}")
            return None
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean PDF extracted text from artifacts"""
        if not text:
            return ""
        
        # Remove PDF object references
        text = re.sub(r'/\w+\s+\d+\s+\d+\s+R', '', text)
        
        # Remove PDF stream markers
        text = re.sub(r'stream\s*\n.*?endstream', '', text, flags=re.DOTALL)
        
        # Remove PDF object definitions
        text = re.sub(r'\d+\s+\d+\s+obj\s*<<.*?>>', '', text, flags=re.DOTALL)
        text = re.sub(r'endobj', '', text)
        
        # Remove binary/encoded content (non-printable characters)
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove lines that are mostly numbers/symbols (likely PDF metadata)
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Keep short lines
                clean_lines.append(line)
                continue
                
            # Check if line has meaningful content (letters and reasonable structure)
            letter_count = sum(1 for c in line if c.isalpha())
            total_chars = len(line)
            
            if total_chars > 0 and (letter_count / total_chars) > 0.3:  # At least 30% letters
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def extract_csv_from_text(self, csv_text: str, title: str) -> Optional[str]:
        """Extract and format CSV data"""
        try:
            import io
            csv_io = io.StringIO(csv_text)
            df = pd.read_csv(csv_io)
            
            text = f"Data from: {title}\n\n"
            text += f"Dataset Information:\n"
            text += f"- Rows: {len(df)}\n"
            text += f"- Columns: {len(df.columns)}\n"
            text += f"- Column names: {', '.join(df.columns)}\n\n"
            
            # Convert data to readable format
            for idx, row in df.iterrows():
                if idx >= 1000:  # Limit to first 1000 rows for large datasets
                    text += f"\n... (showing first 1000 rows of {len(df)} total rows)\n"
                    break
                    
                row_data = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        row_data.append(f"{col}: {row[col]}")
                
                if row_data:
                    text += f"Record {idx + 1}: {', '.join(row_data)}\n"
            
            # Add summary statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text += f"\nNumerical Summary:\n"
                for col in numeric_cols:
                    if col in df.columns:
                        stats = df[col].describe()
                        text += f"{col}: Mean={stats['mean']:.2f}, Min={stats['min']}, Max={stats['max']}\n"
            
            print(f"   📊 Processed CSV with {len(df)} rows and {len(df.columns)} columns")
            return text
            
        except Exception as e:
            print(f"   ❌ CSV processing error: {e}")
            # Return raw text as fallback
            return f"Raw data from: {title}\n\n{csv_text}"
    
    def extract_text_from_html(self, html_text: str, title: str) -> Optional[str]:
        """Extract text from HTML (fallback for some Google Drive responses)"""
        try:
            # Simple HTML text extraction
            import re
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html_text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) > 100:  # Only if we got meaningful text
                return f"Content from: {title}\n\n{text}"
            else:
                return None
                
        except Exception as e:
            print(f"   ❌ HTML extraction error: {e}")
            return None
    
    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from local PDF file"""
        if not PDF_SUPPORT:
            print(f"   ⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
            
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num}/{total_pages} ---\n"
                    text += page_text
                    
            print(f"   📄 Extracted {len(text)} characters from {total_pages} pages")
            return text
        except Exception as e:
            print(f"   ❌ Error extracting PDF: {e}")
            return None
    
    def extract_docx_text(self, docx_path: str) -> Optional[str]:
        """Extract text from local DOCX file"""
        if not DOCX_SUPPORT:
            print(f"   ⚠️  python-docx not installed. Install with: pip install python-docx")
            return None
            
        try:
            doc = Document(docx_path)
            text = ""
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                text += "\n--- Table Data ---\n"
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    text += row_text + "\n"
            
            print(f"   📄 Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            print(f"   ❌ Error extracting DOCX: {e}")
            return None
    
    def extract_excel_text(self, excel_path: str) -> Optional[str]:
        """Extract text from local Excel file"""
        try:
            text = ""
            xl_file = pd.ExcelFile(excel_path)
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                text += f"Columns: {', '.join(df.columns)}\n\n"
                
                # Convert DataFrame to text
                for idx, row in df.iterrows():
                    if idx >= 500:  # Limit for large sheets
                        text += f"... (showing first 500 rows)\n"
                        break
                        
                    row_data = []
                    for col in df.columns:
                        if pd.notna(row[col]):
                            row_data.append(f"{col}: {row[col]}")
                    if row_data:
                        text += f"Row {idx + 1}: {', '.join(row_data)}\n"
                
                # Add statistics
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    text += f"\nSummary Statistics:\n"
                    for col in numeric_cols:
                        stats = df[col].describe()
                        text += f"{col}: Mean={stats['mean']:.2f}, Min={stats['min']}, Max={stats['max']}\n"
            
            print(f"   📊 Extracted {len(text)} characters from {len(xl_file.sheet_names)} sheets")
            return text
        except Exception as e:
            print(f"   ❌ Error extracting Excel: {e}")
            return None
    
    def upload_to_sih(self, text: str, title: str, metadata: Dict = None, source_url: str = None):
        """Upload extracted text to SIH system with chunked processing for large documents"""
        try:
            if not text or len(text.strip()) < 50:
                print(f"   ❌ Text too short or empty")
                return False
            
            text_length = len(text)
            print(f"   📡 Uploading to SIH system...")
            print(f"      📏 Text length: {text_length:,} characters")
            
            # For large documents (>200KB of text), split into smaller parts
            MAX_CHUNK_SIZE = 200_000  # 200KB of text to respect API limits
            
            if text_length > MAX_CHUNK_SIZE:
                print(f"   🔄 Large document detected, splitting into manageable parts...")
                return self.upload_large_document(text, title, metadata, source_url)
            
            # Prepare metadata (minimal to avoid size limits)
            upload_metadata = {
                "processed_at": datetime.now().isoformat()[:19],  # Just date without microseconds
                "text_length": min(text_length, 999999),  # Cap the number
                "source_type": "direct_upload"
            }
            
            # Add only essential metadata with size limits
            if source_url:
                upload_metadata["source_url"] = source_url[:200]  # Limit URL length
                
            if metadata:
                for key, value in metadata.items():
                    if key in ['state', 'year', 'report_type', 'agency', 'document_type', 'filename', 'file_type']:
                        if isinstance(value, str):
                            upload_metadata[key] = str(value)[:50]  # Limit string values
                        elif isinstance(value, (int, float)):
                            upload_metadata[key] = value
                        # Skip other types to keep metadata small
            
            # Prepare payload
            payload = {
                "text": text,
                "title": title,
                "region_meta": upload_metadata
            }
            
            # Send to SIH API with increased timeout for large documents
            timeout = min(600, max(300, text_length // 20000))  # Dynamic timeout based on size
            print(f"      ⏱️  Timeout set to: {timeout} seconds")
            
            response = requests.post(
                f"{self.base_url}/ingest",
                json=payload,
                headers=self.headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Upload successful!")
                print(f"      📄 Document ID: {result['doc_id']}")
                print(f"      🧩 Chunks created: {result['chunks_processed']}")
                print(f"      📏 Total characters: {result['total_characters']}")
                
                self.upload_results.append({
                    "title": title,
                    "doc_id": result['doc_id'],
                    "chunks": result['chunks_processed'],
                    "characters": result['total_characters'],
                    "success": True,
                    "source_url": source_url
                })
                return True
            else:
                error_msg = response.text
                print(f"   ❌ Upload failed: HTTP {response.status_code}")
                print(f"      Error: {error_msg[:200]}...")
                
                self.failed_uploads.append({
                    "title": title,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "source_url": source_url
                })
                return False
                
        except Exception as e:
            print(f"   ❌ Upload exception: {e}")
            self.failed_uploads.append({
                "title": title,
                "error": str(e),
                "source_url": source_url
            })
            return False
    
    def upload_large_document(self, text: str, title: str, metadata: Dict = None, source_url: str = None):
        """Handle upload of very large documents by splitting them"""
        try:
            # Split document into logical sections
            sections = self.split_document_intelligently(text, title)
            
            print(f"   📄 Split into {len(sections)} sections")
            
            successful_sections = 0
            total_chunks = 0
            total_chars = 0
            
            for i, (section_title, section_text) in enumerate(sections, 1):
                print(f"      📋 Uploading section {i}/{len(sections)}: {section_title}")
                
                # Prepare section metadata (minimal to avoid size limits)
                section_metadata = {
                    "parent_document": title[:100],  # Limit title size
                    "section_number": i,
                    "total_sections": len(sections),
                    "section_title": section_title[:100],  # Limit section title
                    "processed_at": datetime.now().isoformat()[:19],  # Just date without microseconds
                    "source_type": "large_document_section"
                }
                
                # Add only essential metadata from original, with size limits
                if metadata:
                    for key, value in metadata.items():
                        if key in ['state', 'year', 'report_type', 'agency', 'document_type']:
                            if isinstance(value, str):
                                section_metadata[key] = str(value)[:50]  # Limit string values
                            else:
                                section_metadata[key] = value
                
                # Upload this section
                section_payload = {
                    "text": section_text,
                    "title": f"{title} - {section_title}",
                    "region_meta": section_metadata
                }
                
                try:
                    response = requests.post(
                        f"{self.base_url}/ingest",
                        json=section_payload,
                        headers=self.headers,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        successful_sections += 1
                        total_chunks += result['chunks_processed']
                        total_chars += result['total_characters']
                        print(f"         ✅ Section uploaded: {result['chunks_processed']} chunks")
                    else:
                        print(f"         ❌ Section failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"         ❌ Section error: {e}")
                
                # Small delay between sections
                time.sleep(1)
            
            # Record overall success
            if successful_sections > 0:
                self.upload_results.append({
                    "title": title,
                    "doc_id": f"multi_section_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "chunks": total_chunks,
                    "characters": total_chars,
                    "success": True,
                    "source_url": source_url,
                    "sections": successful_sections,
                    "total_sections": len(sections)
                })
                
                print(f"   ✅ Large document processed!")
                print(f"      📊 {successful_sections}/{len(sections)} sections uploaded")
                print(f"      🧩 Total chunks: {total_chunks}")
                print(f"      📏 Total characters: {total_chars:,}")
                return True
            else:
                print(f"   ❌ All sections failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Large document processing error: {e}")
            return False
    
    def split_document_intelligently(self, text: str, title: str):
        """Split large document into API-friendly chunks"""
        sections = []
        
        # First, try logical splits with size constraints
        split_patterns = [
            r'\n\s*CHAPTER\s+\d+.*?\n',
            r'\n\s*Chapter\s+\d+.*?\n', 
            r'\n\s*SECTION\s+\d+.*?\n',
            r'\n\s*Section\s+\d+.*?\n',
            r'\n\s*---\s*Page\s+\d+.*?\n',
            r'\n\s*Page\s+\d+.*?\n'
        ]
        
        # Maximum chunk size in characters (very conservative for API limits)
        MAX_CHUNK_CHARS = 200_000  # ~200KB in characters to ensure small metadata
        
        best_split = None
        for pattern in split_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if len(matches) > 1 and len(matches) < 100:  # Reasonable number of sections
                best_split = matches
                print(f"      🔍 Found {len(best_split)} logical sections using pattern")
                break
        
        if best_split:
            # Split by the found pattern but respect size limits
            last_pos = 0
            for i, match in enumerate(best_split):
                if i > 0:  # Skip first match, use it as start of first section
                    section_text = text[last_pos:match.start()].strip()
                    if len(section_text) > 100:  # Only meaningful sections
                        # If section is too large, split it further
                        if len(section_text) > MAX_CHUNK_CHARS:
                            sub_sections = self.split_by_size(section_text, f"Section {i}", MAX_CHUNK_CHARS)
                            sections.extend(sub_sections)
                        else:
                            section_title = f"Section {i}"
                            # Try to extract title from the section
                            lines = section_text[:500].split('\n')
                            for line in lines[:10]:
                                if len(line.strip()) > 10 and len(line.strip()) < 100:
                                    section_title = line.strip()
                                    break
                            sections.append((section_title, section_text))
                last_pos = match.start()
            
            # Add final section
            if last_pos < len(text) - 100:
                final_section = text[last_pos:].strip()
                if len(final_section) > MAX_CHUNK_CHARS:
                    sub_sections = self.split_by_size(final_section, f"Section {len(sections)+1}", MAX_CHUNK_CHARS)
                    sections.extend(sub_sections)
                else:
                    sections.append((f"Section {len(sections)+1}", final_section))
        
        # If no logical split found, split by size
        if not sections:
            print(f"      📏 No logical structure found, splitting by size")
            sections = self.split_by_size(text, "Part", MAX_CHUNK_CHARS)
        
        return sections
    
    def split_by_size(self, text: str, base_title: str, max_size: int):
        """Split text into size-based chunks"""
        chunks = []
        
        # Split by paragraphs first to maintain readability
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_num = 1
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, save current chunk
            if len(current_chunk) + len(paragraph) + 2 > max_size and current_chunk:
                chunks.append((f"{base_title} {chunk_num}", current_chunk.strip()))
                current_chunk = paragraph
                chunk_num += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # If single paragraph is too large, split it by sentences
            if len(current_chunk) > max_size:
                sentences = current_chunk.split('. ')
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 2 > max_size and temp_chunk:
                        chunks.append((f"{base_title} {chunk_num}", temp_chunk.strip() + "."))
                        temp_chunk = sentence
                        chunk_num += 1
                    else:
                        if temp_chunk:
                            temp_chunk += ". " + sentence
                        else:
                            temp_chunk = sentence
                
                current_chunk = temp_chunk
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append((f"{base_title} {chunk_num}", current_chunk.strip()))
        
        return chunks
    
    def upload_from_url(self, url: str, title: str, metadata: Dict = None):
        """Upload document from URL"""
        print(f"\n📁 Processing URL: {title}")
        print(f"   🔗 URL: {url}")
        
        # Download and extract text
        text = self.download_from_url(url, title)
        
        if text:
            # Upload to SIH
            return self.upload_to_sih(text, title, metadata, url)
        else:
            print(f"   ❌ Failed to extract text from URL")
            self.failed_uploads.append({
                "title": title,
                "error": "Failed to extract text from URL",
                "source_url": url
            })
            return False
    
    def upload_file(self, file_path: str, title: str = None, metadata: Dict = None):
        """Upload local file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"   ❌ File not found: {file_path}")
                return False
            
            file_ext = file_path.suffix.lower()
            title = title or file_path.stem
            
            print(f"\n📁 Processing file: {file_path.name}")
            print(f"   📂 Type: {file_ext}")
            
            # Extract text based on file type
            text = None
            if file_ext == '.pdf':
                text = self.extract_pdf_text(str(file_path))
            elif file_ext == '.docx':
                text = self.extract_docx_text(str(file_path))
            elif file_ext in ['.xlsx', '.xls']:
                text = self.extract_excel_text(str(file_path))
            else:
                print(f"   ❌ Unsupported file type: {file_ext}")
                return False
            
            if text:
                # Add file metadata
                file_metadata = {
                    "filename": file_path.name,
                    "file_type": file_ext[1:],
                    "file_size": file_path.stat().st_size,
                    **(metadata or {})
                }
                return self.upload_to_sih(text, title, file_metadata)
            else:
                print(f"   ❌ Failed to extract text from file")
                return False
                
        except Exception as e:
            print(f"   ❌ File processing error: {e}")
            return False
    
    def upload_from_config(self, config_data: List[Dict]):
        """Upload from configuration data"""
        print(f"🚀 MULTI-FORMAT DATA UPLOAD")
        print("="*60)
        print(f"📚 Found {len(config_data)} documents to upload")
        
        successful_uploads = 0
        
        for i, doc_config in enumerate(config_data, 1):
            print(f"\n📊 Progress: {i}/{len(config_data)} ({i/len(config_data)*100:.1f}%)")
            print("-" * 40)
            
            title = doc_config.get('title', 'Unknown Document')
            url = doc_config.get('url')
            file_path = doc_config.get('file_path')
            metadata = doc_config.get('metadata', {})
            
            success = False
            
            if url:
                success = self.upload_from_url(url, title, metadata)
            elif file_path:
                success = self.upload_file(file_path, title, metadata)
            else:
                print(f"   ❌ No URL or file path provided for: {title}")
            
            if success:
                successful_uploads += 1
            
            # Small delay between uploads
            if i < len(config_data):
                time.sleep(2)
        
        self.print_upload_summary(successful_uploads, len(config_data))
    
    def print_upload_summary(self, successful: int, total: int):
        """Print upload summary"""
        print("\n" + "="*60)
        print("🎯 UPLOAD SUMMARY")
        print("="*60)
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Successfully uploaded: {successful}/{total}")
        print(f"   ❌ Failed uploads: {len(self.failed_uploads)}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if self.upload_results:
            total_chunks = sum(r['chunks'] for r in self.upload_results)
            total_chars = sum(r['characters'] for r in self.upload_results)
            
            print(f"\n📄 PROCESSING STATISTICS:")
            print(f"   🧩 Total chunks created: {total_chunks}")
            print(f"   📝 Total characters: {total_chars:,}")
            print(f"   📊 Average chunks per document: {total_chunks/len(self.upload_results):.1f}")
            
            print(f"\n✅ SUCCESSFULLY UPLOADED:")
            for result in self.upload_results:
                print(f"   📄 {result['title']}: {result['chunks']} chunks")
        
        if self.failed_uploads:
            print(f"\n❌ FAILED UPLOADS:")
            for failure in self.failed_uploads:
                print(f"   📄 {failure['title']}: {failure.get('error', 'Unknown error')[:100]}...")
        
        if success_rate >= 80:
            print(f"\n🎉 SUCCESS! Your data is now in the RAG system!")
        else:
            print(f"\n🔧 Some uploads failed - check errors above")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. 🧪 Test queries against your data")
        print(f"   2. 📊 Monitor accuracy and performance")

def main():
    """Main function with your specific data"""
    uploader = MultiFormatDataUploader()
    
    print("🚀 SIH GROUNDWATER DATA UPLOADER")
    print("="*60)
    
    # Your specific documents
    documents = [
        {
            "title": "DYNAMIC GROUND WATER RESOURCES OF INDIA (As on 31st March 2011)",
            "url": "https://drive.google.com/file/d/1IaK1TdBZXoE-YhWISK-H7I9N6M1xwrQf/view?usp=drivesdk",
            "metadata": {
                "state": "All States",
                "year": 2011,
                "report_type": "groundwater_assessment",
                "agency": "CGWB",
                "language": "English",
                "coverage": "national_level",
                "document_type": "official_report",
                "assessment_date": "2011-03-31"
            }
        }
    ]
    
    print("📋 Documents to upload:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc['title']}")
    
    print(f"\n🔧 Configuration:")
    print(f"   🌐 Server: {BASE_URL}")
    print(f"   🔐 Auth: {'Enabled' if AUTH_TOKEN else 'Disabled'}")
    print(f"   📚 Documents: {len(documents)}")
    
    choice = input(f"\n🚀 Start upload? (y/N): ").lower()
    
    if choice == 'y':
        uploader.upload_from_config(documents)
    else:
        print("❌ Upload cancelled")
        
        # Show installation instructions
        print(f"\n💡 OPTIONAL DEPENDENCIES:")
        if not PDF_SUPPORT:
            print(f"   📄 For PDF support: pip install PyPDF2")
        if not DOCX_SUPPORT:
            print(f"   📝 For DOCX support: pip install python-docx")
        print(f"   📊 For Excel support: pip install pandas openpyxl")

if __name__ == "__main__":
    main()