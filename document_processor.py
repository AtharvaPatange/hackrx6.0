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
            response = requests.get(url, timeout=10)  # Reduced timeout
            response.raise_for_status()
            print("Document downloaded successfully.")

            # 2. Read the PDF from the in-memory content
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            full_text = ""
            # Limit pages for speed (first 20 pages only in emergency)
            max_pages = min(len(reader.pages), 50)  # Limit for speed
            for page_num in range(max_pages):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    full_text += page_text + "\n"
            
            if max_pages < len(reader.pages):
                print(f"Warning: Processing only first {max_pages} pages for speed optimization")
            
            print("Text extracted from PDF pages.")

            # 3. Chunk the extracted text aggressively for speed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,   # Smaller chunks for faster processing
                chunk_overlap=50,  # Reduced overlap
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