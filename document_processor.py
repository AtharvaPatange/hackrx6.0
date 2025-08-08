# File: document_processor.py

import requests
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def process_documents(urls: list):
    """
    Downloads PDFs from URLs, extracts text, and returns Document objects.
    Optimized for speed.
    """
    all_documents = []
    
    for url in urls:
        print(f"Processing document from URL: {url}")
        try:
            # 1. Download the PDF content with aggressive timeout
            response = requests.get(url, timeout=8)  # Further reduced timeout
            response.raise_for_status()
            print("Document downloaded successfully.")

            # 2. Read the PDF from the in-memory content
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            full_text = ""
            # Limit pages for speed (first 25 pages for balance)
            max_pages = min(len(reader.pages), 25)  # Reduced for speed while maintaining quality
            for page_num in range(max_pages):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            if max_pages < len(reader.pages):
                print(f"Warning: Processing only first {max_pages} pages for speed optimization")
            
            print("Text extracted from PDF pages.")

            # 3. Chunk the extracted text for better accuracy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,   # Increased for better context retention
                chunk_overlap=100,  # Increased overlap to preserve context
                length_function=len,
                separators=["\n\n", "\n", ". ", " "]  # Simplified separators
            )
            chunks = text_splitter.split_text(full_text)
            print(f"Document split into {len(chunks)} chunks.")

            # Convert to Document objects
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": url, "chunk": i}
                )
                all_documents.append(doc)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading document from {url}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error processing document from {url}: {e}")
            raise
    
    return all_documents

# Keep the original function for backward compatibility but optimize it
def process_document_from_url(url: str):
    """
    Legacy function - use process_documents instead for better performance.
    """
    documents = process_documents([url])
    # Convert back to text chunks for legacy compatibility
    chunks = [doc.page_content for doc in documents]
    
    # Use the optimized vector store function
    from vector_store import process_and_store_documents
    process_and_store_documents(url)