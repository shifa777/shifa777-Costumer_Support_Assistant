"""
Configuration management for the SupportAssistant RAG system.
Loads environment variables and provides centralized access to settings.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration class"""
    
    # Project Paths
    BASE_DIR = Path(__file__).parent.parent
    SRC_DIR = BASE_DIR / "src"
    DATA_DIR = BASE_DIR / "data"
    
    # Ollama Configuration
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma2:2b")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Vector Database
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db_rooman"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rooman_faqs")
    
    # Dataset Configuration
    FAQ_DATABASE_PATH: str = os.getenv(
        "FAQ_DATABASE_PATH",
        r"C:\Users\imran\OneDrive\Desktop\rag\json\faq_database.json"
    )
    
    # Text Splitting Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "20000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "4000"))
    
    # RAG Configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # Application Settings
    APP_TITLE: str = os.getenv("APP_TITLE", "Rooman SupportAssistant")
    APP_ICON: str = os.getenv("APP_ICON", "🎓")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Escalation Storage
    ESCALATION_LOG_PATH: str = str(DATA_DIR / "escalated_queries.json")
    
    @classmethod
    def validate_paths(cls) -> dict[str, bool]:
        """Validate that required paths and services exist"""
        validations = {
            "FAQ Database exists": Path(cls.FAQ_DATABASE_PATH).exists(),
            "GPU Available": torch.cuda.is_available(),
            "CUDA Device": cls.EMBEDDING_DEVICE,
        }
        
        if torch.cuda.is_available():
            validations["GPU Name"] = torch.cuda.get_device_name(0)
            validations["GPU Memory (GB)"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
        
        return validations
    
    @classmethod
    def test_ollama_connection(cls) -> bool:
        """Test connection to Ollama"""
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_API_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print(f"{cls.APP_ICON} {cls.APP_TITLE} - Configuration")
        print("=" * 60)
        print(f"\nOllama:")
        print(f"  URL: {cls.OLLAMA_API_URL}")
        print(f"  Model: {cls.OLLAMA_MODEL}")
        print(f"  Connected: {cls.test_ollama_connection()}")
        
        print(f"\nEmbeddings:")
        print(f"  Model: {cls.EMBEDDING_MODEL}")
        print(f"  Device: {cls.EMBEDDING_DEVICE}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"\nVector Database:")
        print(f"  Path: {cls.CHROMA_DB_PATH}")
        print(f"  Collection: {cls.COLLECTION_NAME}")
        
        print(f"\nDataset:")
        print(f"  Path: {cls.FAQ_DATABASE_PATH}")
        print(f"  Exists: {Path(cls.FAQ_DATABASE_PATH).exists()}")
        
        print(f"\nChunking:")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Overlap: {cls.CHUNK_OVERLAP}")
        
        print(f"\nRAG:")
        print(f"  Top-K Results: {cls.TOP_K_RESULTS}")
        print(f"  Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"  Similarity Threshold: {cls.SIMILARITY_THRESHOLD}")
        print("=" * 60)


# Create singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_config()
