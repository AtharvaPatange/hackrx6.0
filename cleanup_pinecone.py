#!/usr/bin/env python3
"""
Script to delete old Pinecone indexes and create new one with correct dimensions
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def manage_pinecone_indexes():
    """Delete old indexes and optionally create new one"""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # List all indexes
        indexes = pc.list_indexes().names()
        print(f"Current indexes: {indexes}")
        
        # Delete old indexes that might have wrong dimensions
        old_indexes = ["hackrx-gemini-index", "hackrx-groq-index"]
        
        for old_index in old_indexes:
            if old_index in indexes:
                print(f"🗑️  Deleting old index: {old_index}")
                pc.delete_index(old_index)
                print(f"✅ Deleted: {old_index}")
            else:
                print(f"ℹ️  Index {old_index} not found, skipping")
        
        # Create new index with correct dimensions
        new_index_name = "hackrx-jina-index"
        if new_index_name not in pc.list_indexes().names():
            print(f"🆕 Creating new index: {new_index_name}")
            pc.create_index(
                name=new_index_name,
                dimension=768,  # Jina embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print(f"✅ Created new index: {new_index_name}")
        else:
            print(f"ℹ️  Index {new_index_name} already exists")
            
        print("\n🎉 Pinecone indexes are now ready!")
        
    except Exception as e:
        print(f"❌ Error managing indexes: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Managing Pinecone Indexes...\n")
    manage_pinecone_indexes()
