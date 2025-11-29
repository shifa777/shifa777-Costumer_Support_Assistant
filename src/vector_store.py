"""
Vector store module using ChromaDB with GPU-accelerated embeddings.
Provides semantic search over FAQ documents.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from src.config import config
from src.document_processor import DocumentProcessor


class VectorStore:
    """ChromaDB vector store with GPU-accelerated sentence-transformers embeddings"""
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model_name: str = None,
        device: str = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for ChromaDB persistence
            embedding_model_name: Sentence transformer model name
            device: Device for embeddings ('cuda' or 'cpu')
        """
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.persist_directory = persist_directory or config.CHROMA_DB_PATH
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL
        self.device = device or config.EMBEDDING_DEVICE
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model with GPU support
        print(f"\n🚀 Loading embedding model: {self.embedding_model_name}")
        print(f"📍 Device: {self.device}")
        
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        
        if self.device == 'cuda':
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"✅ ChromaDB initialized: {self.persist_directory}")
        print(f"📦 Collection: {self.collection_name}")
        print(f"📊 Existing documents: {self.collection.count()}\n")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using GPU acceleration.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for embedding generation
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
        )
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        clear_existing: bool = False,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            batch_size: Batch size for embedding generation
            clear_existing: Whether to clear existing documents first
            
        Returns:
            Number of documents added
        """
        if clear_existing:
            print("🗑️  Clearing existing collection...")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        if not documents:
            print("⚠️  No documents to add")
            return 0
        
        print(f"\n📝 Adding {len(documents)} documents to vector store...")
        
        # Prepare data
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        print(f"🔮 Generating embeddings (batch_size={batch_size})...")
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Add to ChromaDB
        print(f"💾 Storing in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        print(f"✅ Added {len(documents)} documents to collection '{self.collection_name}'\n")
        return len(documents)
    
    def similarity_search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        top_k = top_k or config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query], batch_size=1)[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to Document objects with scores
        # ChromaDB returns distances (lower is better), convert to similarity (higher is better)
        documents_with_scores = []
        
        if results['ids'] and results['ids'][0]:
            for doc_text, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score (0-1 range)
                # Cosine distance ranges from 0 to 2, so similarity = 1 - (distance / 2)
                similarity = 1 - (distance / 2)
                
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                documents_with_scores.append((doc, similarity))
        
        return documents_with_scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        count = self.collection.count()
        
        stats = {
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': self.persist_directory,
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
            'device': self.device,
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            stats['gpu_name'] = torch.cuda.get_device_name(0)
            stats['gpu_memory_gb'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
        
        return stats
    
    def reset(self):
        """Reset (delete) the collection"""
        print(f"🗑️  Resetting collection '{self.collection_name}'...")
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Collection reset complete\n")


def build_vector_database(clear_existing: bool = False) -> VectorStore:
    """
    Build the vector database from FAQ documents.
    
    Args:
        clear_existing: Whether to clear existing database
        
    Returns:
        Initialized VectorStore with documents
    """
    print("=" * 60)
    print("🏗️  Building Vector Database")
    print("=" * 60)
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Process documents
    documents = processor.process_faq_database()
    
    # Add to vector store
    vector_store.add_documents(documents, clear_existing=clear_existing)
    
    # Show statistics
    stats = vector_store.get_statistics()
    print("\n📊 Vector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Vector Database Build Complete!")
    print("=" * 60 + "\n")
    
    return vector_store


if __name__ == "__main__":
    # Build or rebuild the vector database
    import sys
    
    clear = "--clear" in sys.argv
    
    vector_store = build_vector_database(clear_existing=clear)
    
    # Test query
    print("\n🔍 Testing Search...")
    test_query = "How do I reset my password?"
    results = vector_store.similarity_search(test_query, top_k=3)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} results:\n")
    
    for idx, (doc, score) in enumerate(results, 1):
        print(f"{idx}. Similarity: {score:.4f}")
        print(f"   Question: {doc.metadata.get('question', 'N/A')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print(f"   Content: {doc.page_content[:150]}...\n")
