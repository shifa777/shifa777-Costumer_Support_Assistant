"""
Script to build the vector database from FAQ data.
Run this before starting the Streamlit app for the first time.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import build_vector_database


def main():
    """Build vector database"""
    print("=" * 70)
    print("🏗️  ROOMAN SUPPORTASSISTANT - VECTOR DATABASE BUILDER")
    print("=" * 70)
    print()
    
    # Check for --clear flag
    clear_existing = "--clear" in sys.argv or "-c" in sys.argv
    
    if clear_existing:
        print("⚠️  WARNING: This will clear the existing vector database!")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ Aborted")
            return
    
    # Build database
    try:
        vector_store = build_vector_database(clear_existing=clear_existing)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the application")
        print("2. Ask questions about Rooman courses")
        print("3. Monitor escalations in the sidebar")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
