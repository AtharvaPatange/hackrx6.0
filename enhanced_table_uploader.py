#!/usr/bin/env python3
"""
Enhanced PDF processor with better table extraction for groundwater data
"""

import requests
import PyPDF2
import io
import re
from datetime import datetime
import pdfplumber  # Better for table extraction

def extract_tables_with_pdfplumber(pdf_bytes):
    """Extract tables using pdfplumber which is better for tabular data"""
    try:
        import pdfplumber
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_text = ""
            table_count = 0
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract regular text
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                
                # Extract tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        table_count += 1
                        all_text += f"\n\n--- Table {table_count} (Page {page_num}) ---\n"
                        
                        # Convert table to readable text
                        for row_idx, row in enumerate(table):
                            if row and any(cell for cell in row if cell):  # Skip empty rows
                                # Clean and format row
                                cleaned_row = []
                                for cell in row:
                                    if cell:
                                        cleaned_cell = str(cell).strip().replace('\n', ' ')
                                        cleaned_row.append(cleaned_cell)
                                    else:
                                        cleaned_row.append("")
                                
                                # Join row with separators
                                row_text = " | ".join(cleaned_row)
                                all_text += f"{row_text}\n"
                
                if page_num % 50 == 0:
                    print(f"   Processed {page_num} pages with {table_count} tables...")
        
        print(f"✅ Extracted text with {table_count} tables using pdfplumber")
        return all_text
        
    except ImportError:
        print("⚠️  pdfplumber not available, falling back to PyPDF2")
        return None
    except Exception as e:
        print(f"❌ pdfplumber extraction error: {e}")
        return None

def extract_with_pypdf2_enhanced(pdf_bytes):
    """Enhanced PyPDF2 extraction with better table handling"""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        raw_text = ""
        total_pages = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Enhanced text processing for tables
                    processed_text = enhance_table_text(page_text)
                    raw_text += f"\n\n--- Page {page_num} ---\n\n{processed_text}"
                    
                if page_num % 50 == 0:
                    print(f"   Processed {page_num}/{total_pages} pages...")
                    
            except Exception as e:
                print(f"   ⚠️  Error on page {page_num}: {e}")
                continue
        
        return raw_text
        
    except Exception as e:
        print(f"❌ PyPDF2 extraction error: {e}")
        return None

def enhance_table_text(text):
    """Enhance text extraction to better preserve tabular data"""
    if not text:
        return ""
    
    # Try to identify and format table-like content
    lines = text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            enhanced_lines.append(line)
            continue
        
        # Check if line contains numbers that might be table data
        number_count = len(re.findall(r'\b\d+\.?\d*\b', line))
        word_count = len(line.split())
        
        # If line has many numbers, it's likely table data
        if word_count > 3 and number_count >= 3:
            # Format as table row with clear separators
            # Replace multiple spaces with | separators
            formatted_line = re.sub(r'\s{2,}', ' | ', line)
            enhanced_lines.append(f"TABLE ROW: {formatted_line}")
        else:
            enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def download_and_extract_with_tables():
    """Download and extract groundwater document with enhanced table support"""
    
    url = "https://drive.google.com/uc?export=download&id=1IaK1TdBZXoE-YhWISK-H7I9N6M1xwrQf"
    
    print("📥 Downloading groundwater document...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Download failed: {response.status_code}")
        return None
    
    print(f"✅ Downloaded {len(response.content):,} bytes")
    
    # Try pdfplumber first (better for tables)
    print("📊 Attempting table extraction with pdfplumber...")
    extracted_text = extract_tables_with_pdfplumber(response.content)
    
    # Fallback to enhanced PyPDF2
    if not extracted_text:
        print("📄 Falling back to enhanced PyPDF2 extraction...")
        extracted_text = extract_with_pypdf2_enhanced(response.content)
    
    if not extracted_text:
        print("❌ All extraction methods failed")
        return None
    
    print(f"✅ Extraction complete: {len(extracted_text):,} characters")
    
    # Clean the text
    print("🧹 Cleaning extracted text...")
    clean_text = clean_extracted_text(extracted_text)
    
    print(f"✅ Clean text ready: {len(clean_text):,} characters")
    
    # Quality check
    words = clean_text.split()
    meaningful_words = [w for w in words if len(w) > 3 and w.replace('-', '').isalpha()]
    table_indicators = len(re.findall(r'TABLE ROW:|District|State|Andhra Pradesh|Guntur', clean_text, re.IGNORECASE))
    
    print(f"📊 Quality metrics:")
    print(f"   Total words: {len(words):,}")
    print(f"   Meaningful words: {len(meaningful_words):,}")
    print(f"   Table indicators found: {table_indicators}")
    print(f"   Quality score: {(len(meaningful_words)/len(words)*100) if words else 0:.1f}%")
    
    # Show table preview
    table_lines = [line for line in clean_text.split('\n') if 'TABLE ROW:' in line or 'Guntur' in line]
    if table_lines:
        print(f"\n📋 Table data preview:")
        print("-" * 80)
        for line in table_lines[:5]:
            print(line[:100] + "..." if len(line) > 100 else line)
        print("-" * 80)
    
    return clean_text

def clean_extracted_text(text):
    """Clean extracted text while preserving table structure"""
    if not text:
        return ""
    
    # Remove PDF artifacts but preserve table markers
    text = re.sub(r'%PDF-[\d.]+.*?%%EOF', '', text, flags=re.DOTALL)
    text = re.sub(r'<</.*?>>', '', text, flags=re.DOTALL)
    text = re.sub(r'stream.*?endstream', '', text, flags=re.DOTALL)
    text = re.sub(r'\d+\s+\d+\s+obj.*?endobj', '', text, flags=re.DOTALL)
    
    # Remove binary content but keep printable characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Clean up spacing while preserving table structure
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Always keep table rows
        if 'TABLE ROW:' in line:
            clean_lines.append(line)
            continue
        
        # Skip very short lines unless they're headers
        if len(line) < 3:
            if line:
                clean_lines.append(line)
            continue
        
        # Skip PDF artifacts
        if any(pattern in line for pattern in [
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer',
            '/Filter', '/Length', '/Type', '/Subtype', 'startxref'
        ]):
            continue
        
        # Keep lines with reasonable text content
        alpha_count = sum(1 for c in line if c.isalpha())
        total_count = len(line)
        
        if total_count > 0 and alpha_count / total_count >= 0.2:  # At least 20% letters
            clean_lines.append(line)
    
    # Join and clean up spacing
    result = '\n'.join(clean_lines)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result.strip()

def upload_enhanced_text(text):
    """Upload enhanced text with table data"""
    
    if not text or len(text) < 1000:
        print("❌ No meaningful text to upload")
        return False
    
    print(f"\n📤 Uploading enhanced text with table data...")
    
    # Split into chunks
    chunk_size = 150_000
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    
    print(f"📦 Split into {len(chunks)} chunks")
    
    successful_chunks = 0
    total_processed = 0
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   📋 Uploading chunk {i}/{len(chunks)}...")
        
        payload = {
            "text": chunk,
            "title": f"GROUNDWATER INDIA 2011 with Tables - Part {i}",
            "region_meta": {
                "document": "DYNAMIC GROUND WATER RESOURCES OF INDIA",
                "year": 2011,
                "chunk_number": i,
                "total_chunks": len(chunks),
                "agency": "CGWB",
                "report_type": "groundwater_assessment",
                "extraction_method": "enhanced_tables"
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8001/ingest",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                successful_chunks += 1
                total_processed += result.get('chunks_processed', 0)
                print(f"      ✅ Success: {result.get('chunks_processed', 0)} chunks created")
            else:
                print(f"      ❌ Failed: HTTP {response.status_code}")
                print(f"         Error: {response.text[:100]}...")
        
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print(f"\n🎯 Enhanced upload complete!")
    print(f"   ✅ Successful chunks: {successful_chunks}/{len(chunks)}")
    print(f"   🧩 Total vector chunks: {total_processed}")
    
    return successful_chunks > 0

def main():
    """Main execution with enhanced table extraction"""
    print("🚀 ENHANCED GROUNDWATER DATA UPLOADER WITH TABLES")
    print("=" * 70)
    
    # Extract text with table support
    enhanced_text = download_and_extract_with_tables()
    
    if enhanced_text:
        # Upload to SIH
        success = upload_enhanced_text(enhanced_text)
        
        if success:
            print("\n🎉 SUCCESS! Enhanced groundwater data with tables uploaded!")
            print("\n🧪 Test with these queries:")
            print("   - 'What is the stage of ground water development for Guntur district?'")
            print("   - 'Annual replenishable ground water resource for Andhra Pradesh districts'")
            print("   - 'Ground water draft for irrigation in Visakhapatnam district'")
        else:
            print("\n❌ Upload failed!")
    else:
        print("\n❌ Text extraction failed!")

if __name__ == "__main__":
    main()