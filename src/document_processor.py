"""
Document processing module for loading and chunking FAQ data.
Implements recursive character text splitting with configurable chunk size and overlap.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config


class DocumentProcessor:
    """Process FAQ documents with chunking and metadata preservation"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Paragraph -> sentence -> word
            is_separator_regex=False,
        )
        
    def load_faq_database(self, file_path: str = None) -> List[Dict[str, Any]]:
        """
        Load FAQ database from JSON file.
        
        Args:
            file_path: Path to FAQ JSON file (default from config)
            
        Returns:
            List of FAQ dictionaries
        """
        file_path = file_path or config.FAQ_DATABASE_PATH
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"FAQ database not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'faqs' in data:
            return data['faqs']
        else:
            raise ValueError("Unsupported FAQ database format")
    
    def prepare_documents(self, faqs: List[Dict[str, Any]] = None) -> List[Document]:
        """
        Convert FAQ dictionaries to LangChain Document objects with metadata.
        
        Args:
            faqs: List of FAQ dictionaries (loads from config if None)
            
        Returns:
            List of Document objects ready for embedding
        """
        if faqs is None:
            faqs = self.load_faq_database()
        
        documents = []
        
        for idx, faq in enumerate(faqs):
            # Extract fields (handle different structures)
            question = faq.get('question', faq.get('q', ''))
            answer = faq.get('answer', faq.get('a', ''))
            
            # Skip empty entries
            if not question or not answer:
                continue
            
            # Combine question and answer for better context
            content = f"Question: {question}\n\nAnswer: {answer}"
            
            # Prepare metadata
            metadata = {
                'faq_id': faq.get('id', f'FAQ{idx:04d}'),
                'question': question,
                'category': faq.get('category', 'General'),
                'source': 'rooman_faq_database',
                'index': idx,
            }
            
            # Add optional fields
            if 'keywords' in faq:
                metadata['keywords'] = ','.join(faq['keywords']) if isinstance(faq['keywords'], list) else faq['keywords']
            
            if 'escalate_keywords' in faq:
                metadata['escalate_keywords'] = ','.join(faq['escalate_keywords']) if isinstance(faq['escalate_keywords'], list) else faq['escalate_keywords']
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} FAQ documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using recursive character splitter.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents with preserved metadata
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk information to metadata
        for idx, doc in enumerate(chunked_docs):
            doc.metadata['chunk_id'] = idx
            doc.metadata['chunk_size'] = len(doc.page_content)
        
        print(f"✅ Split into {len(chunked_docs)} chunks (avg size: {sum(len(d.page_content) for d in chunked_docs) // len(chunked_docs)} chars)")
        return chunked_docs
    
    def process_faq_database(self, file_path: str = None) -> List[Document]:
        """
        Complete pipeline: load FAQs, prepare documents, and chunk.
        
        Args:
            file_path: Path to FAQ JSON file (default from config)
            
        Returns:
            List of chunked documents ready for vector storage
        """
        print(f"\n📚 Processing FAQ Database...")
        print(f"  File: {file_path or config.FAQ_DATABASE_PATH}")
        print(f"  Chunk Size: {self.chunk_size}")
        print(f"  Chunk Overlap: {self.chunk_overlap}\n")
        
        # Load and prepare documents
        faqs = self.load_faq_database(file_path)
        documents = self.prepare_documents(faqs)
        
        # Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        print(f"\n✨ Processing complete: {len(chunked_documents)} chunks ready for embedding\n")
        return chunked_documents
    
    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the processed documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {}
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        categories = [doc.metadata.get('category', 'Unknown') for doc in documents if 'chunk_id' not in doc.metadata]
        
        return {
            'total_documents': len(documents),
            'total_chunks': sum(1 for doc in documents if 'chunk_id' in doc.metadata),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'categories': len(set(categories)),
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)},
        }


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    
    # Process documents
    documents = processor.process_faq_database()
    
    # Show statistics
    stats = processor.get_statistics(documents)
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample document
    if documents:
        print("\n📄 Sample Document:")
        print(f"  Content: {documents[0].page_content[:200]}...")
        print(f"  Metadata: {documents[0].metadata}")
