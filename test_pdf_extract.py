#!/usr/bin/env python3
"""
Test PDF extraction for the groundwater document
"""

import requests
from multi_format_uploader import MultiFormatDataUploader

def test_pdf_extraction():
    """Test PDF extraction from the Google Drive document"""
    
    uploader = MultiFormatDataUploader()
    
    url = "https://drive.google.com/file/d/1IaK1TdBZXoE-YhWISK-H7I9N6M1xwrQf/view?usp=drivesdk"
    title = "DYNAMIC GROUND WATER RESOURCES TEST"
    
    print("🧪 Testing PDF extraction...")
    print(f"📄 Document: {title}")
    print(f"🔗 URL: {url}")
    
    # Download and extract
    text = uploader.download_from_url(url, title)
    
    if text:
        print(f"\n✅ Extraction successful!")
        print(f"📏 Total length: {len(text):,} characters")
        
        # Show first 1000 characters
        print(f"\n📋 Preview (first 1000 chars):")
        print("-" * 60)
        print(text[:1000])
        print("-" * 60)
        
        # Check for binary content
        non_printable = sum(1 for c in text[:5000] if ord(c) > 127 or (ord(c) < 32 and c not in '\n\r\t'))
        print(f"\n🔍 Quality check:")
        print(f"   Non-printable chars in first 5000: {non_printable}")
        print(f"   Quality score: {((5000-non_printable)/5000)*100:.1f}%")
        
        # Look for meaningful content
        words = text.split()
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
        print(f"   Meaningful words found: {len(meaningful_words)}")
        
        if len(meaningful_words) > 100:
            print("✅ Text extraction looks good!")
            return True
        else:
            print("❌ Text quality poor - mostly binary/metadata")
            return False
    else:
        print("❌ Extraction failed")
        return False

if __name__ == "__main__":
    test_pdf_extraction()