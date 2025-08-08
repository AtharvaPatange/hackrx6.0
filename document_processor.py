# File: document_processor.py

import requests
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import upsert_chunks

def process_document_from_url(url: str):
    """
    Downloads a PDF from a URL, extracts text, chunks it, and triggers the upsert to Pinecone.
    """
    print(f"Downloading document from URL: {url}")
    try:
        # 1. Download the PDF content
        response = requests.get(url, timeout=15) # Reduced timeout for faster failure
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        print("Document downloaded successfully.")

        # 2. Read the PDF from the in-memory content
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        full_text = ""
        for page in reader.pages:
            # Extract text and handle cases where a page might be empty
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        print("Text extracted from all pages.")

        # 3. Chunk the extracted text
        # Using RecursiveCharacterTextSplitter to split by paragraphs, then sentences, etc.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Further reduced for faster processing
            chunk_overlap=100, # Reduced overlap for speed
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Better separators for insurance documents
        )
        chunks = text_splitter.split_text(full_text)
        print(f"Document was split into {len(chunks)} chunks.")

        # 4. Upsert the chunks into the vector store (Pinecone)
        if chunks:
            upsert_chunks(chunks)
        else:
            print("Warning: No text chunks were generated from the document.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the document: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during document processing: {e}")
        raise