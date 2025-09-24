#!/usr/bin/env python3
"""
Create optimized Pinecone index for maximum RAG accuracy
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment
load_dotenv()

def create_optimal_pinecone_index():
    """Create the most accurate Pinecone index configuration"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Optimal configuration for maximum accuracy
    INDEX_CONFIG = {
        "name": "sih-groundwater-accurate",
        "dimension": 768,  # Jina embeddings v2 dimension
        "metric": "cosine",  # Best for semantic similarity
        "spec": ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Fast and reliable
        )
    }
    
    print("🚀 Creating Optimal Pinecone Index for Maximum RAG Accuracy")
    print("="*60)
    print(f"📊 Index Name: {INDEX_CONFIG['name']}")
    print(f"📏 Dimension: {INDEX_CONFIG['dimension']}")
    print(f"📐 Metric: {INDEX_CONFIG['metric']}")
    print(f"☁️  Cloud: AWS")
    print(f"🌍 Region: us-east-1")
    print("="*60)
    
    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_CONFIG['name'] in existing_indexes:
            print(f"⚠️  Index '{INDEX_CONFIG['name']}' already exists!")
            
            # Get index stats
            index = pc.Index(INDEX_CONFIG['name'])
            stats = index.describe_index_stats()
            
            print(f"📈 Current Stats:")
            print(f"   - Total Vectors: {stats.total_vector_count}")
            print(f"   - Dimension: {stats.dimension}")
            
            choice = input("\n🤔 Do you want to delete and recreate? (y/N): ").lower()
            
            if choice == 'y':
                print("🗑️  Deleting existing index...")
                pc.delete_index(INDEX_CONFIG['name'])
                print("✅ Index deleted!")
            else:
                print("✋ Keeping existing index")
                return INDEX_CONFIG['name']
        
        # Create new index
        print("🔨 Creating new optimized index...")
        pc.create_index(**INDEX_CONFIG)
        
        print("✅ Index created successfully!")
        print("\n🎯 OPTIMIZATION FEATURES:")
        print("   ✅ Cosine similarity - Best for semantic search")
        print("   ✅ 768 dimensions - Perfect for Jina embeddings")
        print("   ✅ AWS us-east-1 - Fast and reliable")
        print("   ✅ Serverless - Auto-scaling for efficiency")
        
        return INDEX_CONFIG['name']
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return None

def update_env_file(new_index_name):
    """Update .env file with new index name"""
    
    env_file = ".env"
    
    try:
        # Read current .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update PINECONE_INDEX_NAME
        updated_lines = []
        updated = False
        
        for line in lines:
            if line.startswith('PINECONE_INDEX_NAME='):
                updated_lines.append(f'PINECONE_INDEX_NAME="{new_index_name}"\n')
                updated = True
            else:
                updated_lines.append(line)
        
        # Add if not found
        if not updated:
            updated_lines.append(f'PINECONE_INDEX_NAME="{new_index_name}"\n')
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Updated .env file with new index: {new_index_name}")
        
    except Exception as e:
        print(f"⚠️  Could not update .env file: {e}")
        print(f"📝 Please manually update PINECONE_INDEX_NAME='{new_index_name}'")

if __name__ == "__main__":
    print("🎯 Pinecone Index Optimizer for Maximum RAG Accuracy\n")
    
    # Create optimal index
    new_index = create_optimal_pinecone_index()
    
    if new_index:
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your optimized index is ready")
        print("="*60)
        
        # Update .env file
        update_env_file(new_index)
        
        print("\n📋 NEXT STEPS:")
        print("1. ✅ Restart your SIH server")
        print("2. 📚 Ingest your PDF/Excel/Docs data")
        print("3. 🧪 Test queries for maximum accuracy")
        print("\n🚀 Your RAG system is now optimized for peak performance!")
    else:
        print("❌ Failed to create index. Please check your Pinecone API key.")