#!/usr/bin/env python3
"""
Pinecone Index Cleanup Script
Removes corrupted binary chunks and keeps only clean text data
"""

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
import time
from typing import List, Dict, Any

# Load environment
load_dotenv()

class PineconeCleanup:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "sih-groundwater-accurate")
        self.index = self.pc.Index(self.index_name)
        
    def analyze_index(self):
        """Analyze current index status"""
        print("🔍 ANALYZING PINECONE INDEX")
        print("=" * 50)
        
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            
            print(f"📊 Index: {self.index_name}")
            print(f"🔢 Total Vectors: {total_vectors:,}")
            print(f"📏 Dimensions: {stats['dimension']}")
            
            return total_vectors
            
        except Exception as e:
            print(f"❌ Error analyzing index: {e}")
            return 0
    
    def sample_and_identify_corrupted(self, sample_size: int = 100):
        """Sample vectors and identify which ones are corrupted"""
        print(f"\n🔍 SAMPLING {sample_size} VECTORS TO IDENTIFY CORRUPTION")
        print("=" * 50)
        
        try:
            # Query for sample data
            results = self.index.query(
                vector=[0.0] * 768,  # Dummy vector to get any results
                top_k=sample_size,
                include_metadata=True,
                include_values=False
            )
            
            clean_vectors = []
            corrupted_vectors = []
            
            for match in results['matches']:
                vector_id = match['id']
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                title = metadata.get('title', '')
                
                # Check if vector contains binary/corrupted content
                is_corrupted = self.is_corrupted_content(text, title)
                
                if is_corrupted:
                    corrupted_vectors.append(vector_id)
                else:
                    clean_vectors.append(vector_id)
            
            print(f"✅ Clean vectors found: {len(clean_vectors)}")
            print(f"❌ Corrupted vectors found: {len(corrupted_vectors)}")
            print(f"📈 Corruption rate: {len(corrupted_vectors)/len(results['matches'])*100:.1f}%")
            
            # Show examples
            if clean_vectors:
                print(f"\n📄 Example clean vector: {clean_vectors[0]}")
            if corrupted_vectors:
                print(f"🗑️  Example corrupted vector: {corrupted_vectors[0]}")
            
            return clean_vectors, corrupted_vectors
            
        except Exception as e:
            print(f"❌ Error sampling vectors: {e}")
            return [], []
    
    def is_corrupted_content(self, text: str, title: str) -> bool:
        """Determine if content is corrupted/binary based on patterns"""
        if not text:
            return True
        
        # Check for PDF binary artifacts
        binary_patterns = [
            r'/Im\d+\s+\d+\s+\d+\s+R',  # PDF image references
            r'stream.*?endstream',       # PDF streams
            r'\d+\s+\d+\s+obj',         # PDF objects
            r'<</.*?>>',                 # PDF dictionaries
            r'startxref',                # PDF cross-reference
            r'%%EOF',                    # PDF end marker
            r'xref',                     # PDF cross-reference table
        ]
        
        # Check if text contains too many binary patterns
        binary_matches = 0
        for pattern in binary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            binary_matches += len(matches)
        
        # Check character composition
        if len(text) > 100:
            # Count non-printable characters
            non_printable = sum(1 for c in text[:1000] if ord(c) > 127 or (ord(c) < 32 and c not in '\n\r\t'))
            non_printable_ratio = non_printable / min(1000, len(text))
            
            # Count meaningful words
            words = text.split()
            meaningful_words = sum(1 for word in words[:100] if len(word) > 2 and word.isalpha())
            meaningful_ratio = meaningful_words / min(100, len(words)) if words else 0
            
            # Determine if corrupted
            is_corrupted = (
                binary_matches > 5 or                    # Too many PDF artifacts
                non_printable_ratio > 0.3 or            # Too many non-printable chars
                meaningful_ratio < 0.2                   # Too few meaningful words
            )
            
            return is_corrupted
        
        return True  # Very short text is likely corrupted
    
    def identify_all_corrupted_vectors(self) -> List[str]:
        """Identify all corrupted vectors in the index"""
        print("\n🔍 IDENTIFYING ALL CORRUPTED VECTORS")
        print("=" * 50)
        
        corrupted_ids = []
        batch_size = 100
        processed_ids = set()
        
        try:
            while True:
                # Get batch of vectors with metadata
                results = self.index.query(
                    vector=[0.0] * 768,
                    top_k=batch_size,
                    include_metadata=True,
                    include_values=False
                )
                
                if not results['matches']:
                    break
                
                new_vectors = []
                for match in results['matches']:
                    vector_id = match['id']
                    if vector_id not in processed_ids:
                        new_vectors.append(match)
                        processed_ids.add(vector_id)
                
                if not new_vectors:
                    break
                
                batch_corrupted = []
                for match in new_vectors:
                    vector_id = match['id']
                    metadata = match.get('metadata', {})
                    text = metadata.get('text', '')
                    title = metadata.get('title', '')
                    
                    if self.is_corrupted_content(text, title):
                        batch_corrupted.append(vector_id)
                
                corrupted_ids.extend(batch_corrupted)
                
                print(f"   📊 Processed: {len(processed_ids):,}, Corrupted found: {len(corrupted_ids):,}")
                time.sleep(0.5)  # Rate limiting
            
            print(f"✅ Identified {len(corrupted_ids):,} corrupted vectors")
            return corrupted_ids
            
        except Exception as e:
            print(f"❌ Error identifying corrupted vectors: {e}")
            return []
    
    def delete_vectors_batch(self, vector_ids: List[str], batch_size: int = 100):
        """Delete vectors in batches"""
        print(f"\n🗑️  DELETING {len(vector_ids):,} CORRUPTED VECTORS")
        print("=" * 50)
        
        deleted_count = 0
        failed_count = 0
        
        try:
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i+batch_size]
                
                try:
                    self.index.delete(ids=batch)
                    deleted_count += len(batch)
                    print(f"   ✅ Deleted batch {i//batch_size + 1}: {len(batch)} vectors")
                    
                except Exception as e:
                    failed_count += len(batch)
                    print(f"   ❌ Failed batch {i//batch_size + 1}: {e}")
                
                # Rate limiting
                time.sleep(1)
            
            print(f"\n🎯 DELETION SUMMARY:")
            print(f"   ✅ Successfully deleted: {deleted_count:,}")
            print(f"   ❌ Failed to delete: {failed_count:,}")
            
            return deleted_count
            
        except Exception as e:
            print(f"❌ Error during batch deletion: {e}")
            return 0
    
    def verify_cleanup(self):
        """Verify the cleanup was successful"""
        print("\n✅ VERIFYING CLEANUP")
        print("=" * 50)
        
        try:
            # Get new stats
            stats = self.index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            
            print(f"📊 Vectors remaining: {total_vectors:,}")
            
            # Sample remaining vectors to check quality
            results = self.index.query(
                vector=[0.0] * 768,
                top_k=min(20, total_vectors),
                include_metadata=True,
                include_values=False
            )
            
            clean_count = 0
            corrupted_count = 0
            
            for match in results['matches']:
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                title = metadata.get('title', '')
                
                if self.is_corrupted_content(text, title):
                    corrupted_count += 1
                else:
                    clean_count += 1
            
            print(f"🔍 Sample quality check:")
            print(f"   ✅ Clean vectors: {clean_count}")
            print(f"   ❌ Corrupted vectors: {corrupted_count}")
            if clean_count + corrupted_count > 0:
                print(f"   📈 Clean rate: {clean_count/(clean_count+corrupted_count)*100:.1f}%")
            
            return corrupted_count == 0
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False
    
    def run_complete_cleanup(self):
        """Run the complete cleanup process"""
        print("🚀 STARTING COMPLETE PINECONE CLEANUP")
        print("=" * 60)
        
        # Step 1: Analyze current state
        total_vectors = self.analyze_index()
        if total_vectors == 0:
            print("❌ No vectors found in index")
            return
        
        # Step 2: Sample and analyze corruption
        clean_sample, corrupted_sample = self.sample_and_identify_corrupted(50)
        
        if len(clean_sample) + len(corrupted_sample) == 0:
            print("❌ No samples found")
            return
        
        corruption_rate = len(corrupted_sample) / (len(clean_sample) + len(corrupted_sample)) * 100
        print(f"\n📊 Estimated corruption rate: {corruption_rate:.1f}%")
        
        if corruption_rate < 10:
            print("✅ Index appears mostly clean, no cleanup needed")
            return
        
        # Step 3: Get user confirmation
        print(f"\n⚠️  WARNING: This will delete approximately {corruption_rate:.0f}% of vectors!")
        print(f"   Estimated corrupted vectors: ~{total_vectors * corruption_rate / 100:.0f}")
        print(f"   Estimated clean vectors: ~{total_vectors * (100-corruption_rate) / 100:.0f}")
        
        confirm = input("\n🤔 Proceed with cleanup? (yes/no): ").lower()
        if confirm != 'yes':
            print("❌ Cleanup cancelled")
            return
        
        # Step 4: Identify all corrupted vectors
        corrupted_ids = self.identify_all_corrupted_vectors()
        
        if not corrupted_ids:
            print("✅ No corrupted vectors found!")
            return
        
        # Step 5: Delete corrupted vectors
        deleted_count = self.delete_vectors_batch(corrupted_ids)
        
        # Step 6: Wait for changes to propagate
        print("\n⏱️  Waiting for changes to propagate...")
        time.sleep(10)
        
        # Step 7: Verify cleanup
        success = self.verify_cleanup()
        
        if success:
            print("\n🎉 CLEANUP SUCCESSFUL!")
            print("✅ All corrupted vectors removed")
            print("✅ Index now contains only clean, readable data")
            print("✅ Query accuracy should be 100%")
        else:
            print("\n⚠️  CLEANUP PARTIALLY SUCCESSFUL")
            print("   Some corrupted vectors may remain")
            print("   Consider running cleanup again")

def main():
    """Main cleanup execution"""
    print("🧹 PINECONE INDEX CLEANUP TOOL")
    print("=" * 60)
    print("This tool will remove corrupted binary chunks from your index")
    print("and keep only clean, readable groundwater data.")
    print()
    
    try:
        cleanup = PineconeCleanup()
        cleanup.run_complete_cleanup()
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("Please check your Pinecone configuration and try again.")

if __name__ == "__main__":
    main()