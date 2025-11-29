"""
Query classification and escalation logic.
Determines whether a query can be handled by FAQ or needs escalation.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document

from src.config import config


@dataclass
class QueryClassification:
    """Result of query classification"""
    should_escalate: bool
    confidence_score: float
    escalation_reasons: List[str]
    retrieved_docs: List[Tuple[Document, float]]
    classification: str  # 'FAQ' or 'ESCALATE'


class QueryClassifier:
    """
    Analyzes queries to determine if they can be answered by FAQs or need escalation.
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        similarity_threshold: float = None,
    ):
        """
        Initialize query classifier.
        
        Args:
            confidence_threshold: Minimum confidence to avoid escalation
            similarity_threshold: Minimum similarity score for relevant docs
        """
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
    
    def is_greeting(self, query: str) -> bool:
        """Check if query is a simple greeting or small talk"""
        greetings = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'namaste', 'greetings', 'howdy', 'what\'s up', 'whats up', 'sup',
            'how are you', 'how r u', 'how do you do', 'nice to meet you',
            'thanks', 'thank you', 'thx', 'bye', 'goodbye', 'see you', 'cya'
        ]
        
        query_lower = query.lower().strip()
        
        # Exact match or starts with greeting
        return any(query_lower == g or query_lower.startswith(g) for g in greetings)
    
    def classify_query(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
    ) -> QueryClassification:
        """
        Classify a query based on retrieved documents and confidence.
        
        Args:
            query: User query
            retrieved_docs: List of (Document, similarity_score) tuples
            
        Returns:
            QueryClassification result
        """
        escalation_reasons = []
        should_escalate = False
        
        # Check 1: No relevant documents found
        if not retrieved_docs:
            should_escalate = True
            escalation_reasons.append("No relevant FAQs found")
            confidence_score = 0.0
        else:
            # Get the best similarity score
            best_score = retrieved_docs[0][1] if retrieved_docs else 0.0
            confidence_score = best_score
            
            # Check 2: Low similarity score
            if best_score < self.similarity_threshold:
                should_escalate = True
                escalation_reasons.append(f"Low similarity ({best_score:.2f} < {self.similarity_threshold})")
            
            # Check 3: Confidence below threshold
            if best_score < self.confidence_threshold:
                should_escalate = True
                escalation_reasons.append(f"Low confidence ({best_score:.2f} < {self.confidence_threshold})")
            
            # Check 4: Multiple intents (query contains multiple questions)
            if self._has_multiple_intents(query):
                should_escalate = True
                escalation_reasons.append("Multiple questions detected")
            
            # Check 5: Contains escalation keywords
            escalation_keywords = self._check_escalation_keywords(query, retrieved_docs)
            if escalation_keywords:
                should_escalate = True
                escalation_reasons.append(f"Escalation keywords: {', '.join(escalation_keywords)}")
            
            # Check 6: Personal/sensitive request indicators
            if self._is_personal_request(query):
                should_escalate = True
                escalation_reasons.append("Personal/sensitive request detected")
            
            # Check 7: Complaint or negative sentiment
            if self._is_complaint(query):
                should_escalate = True
                escalation_reasons.append("Complaint detected")
        
        classification = 'ESCALATE' if should_escalate else 'FAQ'
        
        return QueryClassification(
            should_escalate=should_escalate,
            confidence_score=confidence_score,
            escalation_reasons=escalation_reasons,
            retrieved_docs=retrieved_docs,
            classification=classification,
        )
    
    def _has_multiple_intents(self, query: str) -> bool:
        """Check if query contains multiple questions/intents"""
        question_markers = ['?', ' and ', ' also ', ' plus ']
        question_count = sum(1 for marker in question_markers if marker in query.lower())
        return question_count >= 2
    
    def _check_escalation_keywords(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
    ) -> List[str]:
        """Check for escalation keywords in query or retrieved docs"""
        # Common escalation keywords
        escalation_keywords = [
            'refund', 'cancel', 'delete account', 'complaint',
            'speak to human', 'talk to agent', 'customer support',
            'not working', 'broken', 'error', 'problem',
            'urgent', 'emergency', 'immediately',
        ]
        
        query_lower = query.lower()
        found_keywords = [kw for kw in escalation_keywords if kw in query_lower]
        
        # Check metadata for escalate_keywords from FAQ
        for doc, _ in retrieved_docs[:1]:  # Check top result
            doc_escalate_kw = doc.metadata.get('escalate_keywords', '')
            if doc_escalate_kw:
                keywords = doc_escalate_kw.split(',')
                for kw in keywords:
                    if kw.strip().lower() in query_lower:
                        found_keywords.append(kw.strip())
        
        return list(set(found_keywords))  # Remove duplicates
    
    def _is_personal_request(self, query: str) -> bool:
        """Check if query is a personal/account-specific request"""
        personal_indicators = [
            'my account', 'my enrollment', 'my payment', 'my certificate',
            'i can\'t', 'i cannot', 'i\'m unable', 'i need help',
            'my login', 'my password', 'change my', 'update my',
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in personal_indicators)
    
    def _is_complaint(self, query: str) -> bool:
        """Check if query is a complaint"""
        complaint_indicators = [
            'disappointed', 'unhappy', 'frustrated', 'angry',
            'terrible', 'horrible', 'worst', 'scam',
            'waste of', 'rip off', 'never', 'always',
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complaint_indicators)
    
    def get_classification_explanation(self, classification: QueryClassification) -> str:
        """Get human-readable explanation of classification"""
        if classification.should_escalate:
            reasons = "\n  - ".join(classification.escalation_reasons)
            return f"⚠️ **Escalation Recommended**\n\n**Reasons:**\n  - {reasons}"
        else:
            return f"✅ **FAQ Response** (Confidence: {classification.confidence_score:.2%})"


if __name__ == "__main__":
    # Test query classifier
    from src.vector_store import VectorStore
    
    vector_store = VectorStore()
    classifier = QueryClassifier()
    
    test_queries = [
        "What courses does Rooman offer?",
        "How do I get a refund and delete my account?",
        "I'm very disappointed with the course quality",
        "My login is not working, please help immediately",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print('=' * 60)
        
        # Search
        docs = vector_store.similarity_search(query, top_k=3)
        
        # Classify
        classification = classifier.classify_query(query, docs)
        
        # Print results
        print(f"\nClassification: {classification.classification}")
        print(f"Confidence: {classification.confidence_score:.2%}")
        print(f"Should Escalate: {classification.should_escalate}")
        
        if classification.escalation_reasons:
            print(f"\nEscalation Reasons:")
            for reason in classification.escalation_reasons:
                print(f"  - {reason}")
        
        if docs:
            print(f"\nTop Result:")
            print(f"  Similarity: {docs[0][1]:.4f}")
            print(f"  Question: {docs[0][0].metadata.get('question', 'N/A')}")
