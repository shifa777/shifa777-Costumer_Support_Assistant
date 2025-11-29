"""
Escalation management system for handling complex queries.
Logs and tracks queries that need human intervention.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from src.config import config


@dataclass
class EscalatedQuery:
    """Escalated query record"""
    timestamp: str
    query: str
    reasons: List[str]
    confidence_score: float
    sources: List[Dict[str, Any]]
    session_id: Optional[str] = None
    status: str = "pending"  # pending, resolved, closed
    notes: str = ""


class EscalationManager:
    """Manages escalated queries and logging"""
    
    def __init__(self, log_path: str = None):
        """
        Initialize escalation manager.
        
        Args:
            log_path: Path to escalation log JSON file
        """
        self.log_path = log_path or config.ESCALATION_LOG_PATH
        
        # Ensure data directory exists
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not Path(self.log_path).exists():
            self._save_escalations([])
    
    def log_escalation(
        self,
        query: str,
        reasons: List[str],
        confidence_score: float,
        sources: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> EscalatedQuery:
        """
        Log an escalated query.
        
        Args:
            query: The user's query
            reasons: List of escalation reasons
            confidence_score: Confidence score from classification
            sources: Retrieved sources
            session_id: Optional session identifier
            
        Returns:
            EscalatedQuery object
        """
        escalated_query = EscalatedQuery(
            timestamp=datetime.now().isoformat(),
            query=query,
            reasons=reasons,
            confidence_score=confidence_score,
            sources=sources,
            session_id=session_id,
        )
        
        # Load existing escalations
        escalations = self.load_escalations()
        
        # Add new escalation
        escalations.append(asdict(escalated_query))
        
        # Save
        self._save_escalations(escalations)
        
        return escalated_query
    
    def load_escalations(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load escalated queries from log.
        
        Args:
            status: Filter by status (pending, resolved, closed)
            limit: Maximum number of records to return
            
        Returns:
            List of escalation dictionaries
        """
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                escalations = json.load(f)
            
            # Filter by status if specified
            if status:
                escalations = [e for e in escalations if e.get('status') == status]
            
            # Apply limit
            if limit:
                escalations = escalations[-limit:]  # Most recent
            
            return escalations
        
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_escalations(self, escalations: List[Dict[str, Any]]):
        """Save escalations to JSON file"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(escalations, f, indent=2, ensure_ascii=False)
    
    def update_escalation_status(
        self,
        index: int,
        status: str,
        notes: str = "",
    ) -> bool:
        """
        Update the status of an escalation.
        
        Args:
            index: Index of escalation in the log
            status: New status (pending, resolved, closed)
            notes: Optional notes
            
        Returns:
            True if successful, False otherwise
        """
        escalations = self.load_escalations()
        
        if 0 <= index < len(escalations):
            escalations[index]['status'] = status
            if notes:
                escalations[index]['notes'] = notes
            escalations[index]['updated_at'] = datetime.now().isoformat()
            
            self._save_escalations(escalations)
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        escalations = self.load_escalations()
        
        if not escalations:
            return {
                'total': 0,
                'pending': 0,
                'resolved': 0,
                'closed': 0,
            }
        
        stats = {
            'total': len(escalations),
            'pending': sum(1 for e in escalations if e.get('status') == 'pending'),
            'resolved': sum(1 for e in escalations if e.get('status') == 'resolved'),
            'closed': sum(1 for e in escalations if e.get('status') == 'closed'),
        }
        
        # Most common escalation reasons
        all_reasons = []
        for e in escalations:
            all_reasons.extend(e.get('reasons', []))
        
        if all_reasons:
            from collections import Counter
            reason_counts = Counter(all_reasons)
            stats['top_reasons'] = dict(reason_counts.most_common(5))
        
        return stats
    
    def export_to_csv(self, output_path: str = None) -> str:
        """
        Export escalations to CSV format.
        
        Args:
            output_path: Path for CSV file (default: data/escalations.csv)
            
        Returns:
            Path to exported file
        """
        import csv
        
        output_path = output_path or str(Path(config.DATA_DIR) / "escalations.csv")
        escalations = self.load_escalations()
        
        if not escalations:
            return None
        
        # Define CSV fields
        fieldnames = [
            'timestamp', 'query', 'status', 'confidence_score',
            'reasons', 'session_id', 'notes'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for e in escalations:
                row = {
                    'timestamp': e.get('timestamp', ''),
                    'query': e.get('query', ''),
                    'status': e.get('status', 'pending'),
                    'confidence_score': e.get('confidence_score', 0.0),
                    'reasons': '; '.join(e.get('reasons', [])),
                    'session_id': e.get('session_id', ''),
                    'notes': e.get('notes', ''),
                }
                writer.writerow(row)
        
        return output_path


if __name__ == "__main__":
    # Test escalation manager
    manager = EscalationManager()
    
    # Log test escalation
    query = "I need a refund immediately!"
    manager.log_escalation(
        query=query,
        reasons=["Escalation keywords: refund, immediately", "Personal request detected"],
        confidence_score=0.45,
        sources=[
            {'question': 'What is the refund policy?', 'similarity': 0.65}
        ],
        session_id="test_session_123"
    )
    
    print("✅ Logged test escalation\n")
    
    # Get statistics
    stats = manager.get_statistics()
    print("📊 Escalation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load escalations
    print(f"\n📋 Recent Escalations:")
    escalations = manager.load_escalations(limit=5)
    for idx, esc in enumerate(escalations, 1):
        print(f"\n{idx}. Query: {esc['query']}")
        print(f"   Status: {esc['status']}")
        print(f"   Confidence: {esc['confidence_score']:.2%}")
        print(f"   Reasons: {', '.join(esc['reasons'])}")
