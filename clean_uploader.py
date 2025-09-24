#!/usr/bin/env python3
"""
Clean uploader for groundwater data with improved PDF processing
"""

import requests
import PyPDF2
import io
import re
from datetime import datetime

def clean_extracted_text(text):
    """Aggressively clean PDF extracted text"""
    if not text:
        return ""
    
    # Remove PDF headers and metadata
    text = re.sub(r'%PDF-[\d.]+.*?%%EOF', '', text, flags=re.DOTALL)
    text = re.sub(r'<</.*?>>', '', text, flags=re.DOTALL)
    text = re.sub(r'stream.*?endstream', '', text, flags=re.DOTALL)
    text = re.sub(r'\d+\s+\d+\s+obj.*?endobj', '', text, flags=re.DOTALL)
    
    # Remove xref tables and trailers
    text = re.sub(r'xref.*?trailer', '', text, flags=re.DOTALL)
    text = re.sub(r'startxref.*', '', text, flags=re.DOTALL)
    
    # Remove binary content and control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Split into lines and filter
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and very short lines
        if len(line) < 5:
            if line:  # Keep non-empty short lines
                clean_lines.append(line)
            continue
        
        # Skip lines that are mostly numbers/symbols (PDF artifacts)
        alpha_count = sum(1 for c in line if c.isalpha())
        total_count = len(line)
        
        if alpha_count / total_count < 0.4:  # Less than 40% letters
            continue
            
        # Skip lines with PDF-specific patterns
        if any(pattern in line for pattern in [
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer',
            '/Filter', '/Length', '/Type', '/Subtype', 'startxref'
        ]):
            continue
            
        clean_lines.append(line)
    
    # Join and clean up spacing
    result = '\n'.join(clean_lines)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Max 2 consecutive newlines
    
    return result.strip()

def download_and_extract_groundwater_doc():
    """Download and extract text from the groundwater document"""
    
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
    
    # Extract text from PDF
    print("📄 Extracting text from PDF...")
    
    try:
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        raw_text = ""
        total_pages = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    raw_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                    
                if page_num % 50 == 0:
                    print(f"   Processed {page_num}/{total_pages} pages...")
                    
            except Exception as e:
                print(f"   ⚠️  Error on page {page_num}: {e}")
                continue
        
        print(f"✅ Raw extraction complete: {len(raw_text):,} characters")
        
        # Clean the text
        print("🧹 Cleaning extracted text...")
        clean_text = clean_extracted_text(raw_text)
        
        print(f"✅ Clean text ready: {len(clean_text):,} characters")
        
        # Quality check
        words = clean_text.split()
        meaningful_words = [w for w in words if len(w) > 3 and w.replace('-', '').isalpha()]
        
        print(f"📊 Quality metrics:")
        print(f"   Total words: {len(words):,}")
        print(f"   Meaningful words: {len(meaningful_words):,}")
        print(f"   Quality score: {(len(meaningful_words)/len(words)*100) if words else 0:.1f}%")
        
        # Show preview
        print(f"\n📋 Text preview (first 500 chars):")
        print("-" * 60)
        print(clean_text[:500])
        print("-" * 60)
        
        return clean_text
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return None

def upload_clean_text(text):
    """Upload the clean text to SIH system"""
    
    if not text or len(text) < 1000:
        print("❌ No meaningful text to upload")
        return False
    
    print(f"\n📤 Uploading clean text to SIH...")
    
    # Split into chunks (smaller for better metadata handling)
    chunk_size = 150_000  # 150KB chunks
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
            "title": f"GROUNDWATER RESOURCES INDIA 2011 - Part {i}",
            "region_meta": {
                "document": "DYNAMIC GROUND WATER RESOURCES OF INDIA",
                "year": 2011,
                "chunk_number": i,
                "total_chunks": len(chunks),
                "agency": "CGWB",
                "report_type": "groundwater_assessment"
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
    
    print(f"\n🎯 Upload complete!")
    print(f"   ✅ Successful chunks: {successful_chunks}/{len(chunks)}")
    print(f"   🧩 Total vector chunks: {total_processed}")
    
    return successful_chunks > 0

def main():
    """Main execution"""
    print("🚀 CLEAN GROUNDWATER DATA UPLOADER")
    print("=" * 60)
    
    # Extract clean text
    clean_text = download_and_extract_groundwater_doc()
    
    if clean_text:
        # Upload to SIH
        success = upload_clean_text(clean_text)
        
        if success:
            print("\n🎉 SUCCESS! Clean groundwater data uploaded to RAG system!")
            print("\n🧪 Try testing with queries like:")
            print("   - 'What is India's total groundwater resource?'")
            print("   - 'Which states have the highest groundwater availability?'")
            print("   - 'What is the net groundwater availability for irrigation?'")
        else:
            print("\n❌ Upload failed!")
    else:
        print("\n❌ Text extraction failed!")

if __name__ == "__main__":
    main()