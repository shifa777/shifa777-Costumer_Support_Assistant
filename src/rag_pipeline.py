"""
RAG Pipeline using LangChain and Ollama.
Orchestrates document retrieval, query classification, and response generation.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.config import config
from src.vector_store import VectorStore
from src.query_classifier import QueryClassifier, QueryClassification


@dataclass
class RAGResponse:
    """Complete RAG pipeline response"""
    query: str
    answer: str
    classification: QueryClassification
    sources: List[Dict[str, Any]]
    should_escalate: bool
    llm_used: bool  # Whether LLM was used to generate response


class RAGPipeline:
    """
    Complete RAG pipeline integrating retrieval, classification, and generation.
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        classifier: QueryClassifier = None,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized VectorStore (creates new if None)
            classifier: QueryClassifier instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.classifier = classifier or QueryClassifier()
        
        # Initialize Ollama LLM
        print(f"\n🤖 Initializing Ollama LLM: {config.OLLAMA_MODEL}")
        print(f"   URL: {config.OLLAMA_API_URL}")
        
        self.llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_API_URL,
            temperature=0.3,  # Lower temperature for more factual responses
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are a helpful support assistant for Rooman Technologies, an IT training institute.
Answer the user's question based ONLY on the provided context from the FAQ database.

IMPORTANT INSTRUCTIONS:
- Provide a clear, helpful, and concise answer
- Use ONLY information from the context below
- If the context doesn't contain enough information to answer fully, say so clearly
- Do NOT make up information or provide answers not supported by the context
- Be friendly and professional
- If you're unsure, admit it

Context from FAQ Database:
{context}

User Question: {question}

Answer:"""
        )
        
        print("✅ RAG Pipeline initialized\n")
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        return self.vector_store.similarity_search(query, top_k=top_k)
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
    ) -> str:
        """
        Generate an answer using the LLM and retrieved context.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents with scores
            
        Returns:
            Generated answer
        """
        # Prepare context from retrieved documents
        context_parts = []
        for idx, (doc, score) in enumerate(retrieved_docs, 1):
            question = doc.metadata.get('question', '')
            category = doc.metadata.get('category', 'General')
            content = doc.page_content
            
            context_parts.append(
                f"[FAQ {idx}] (Category: {category}, Relevance: {score:.2%})\n{content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )
        
        # Generate response
        answer = self.llm.invoke(prompt)
        
        return answer.strip()
    
    def process_query(
        self,
        query: str,
        top_k: int = None,
    ) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve, classify, and generate response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        # Check if query is a greeting first
        if self.classifier.is_greeting(query):
            # Handle greetings without FAQ lookup
            answer = self._generate_greeting_response(query)
            
            return RAGResponse(
                query=query,
                answer=answer,
                classification=QueryClassification(
                    should_escalate=False,
                    confidence_score=1.0,
                    escalation_reasons=[],
                    retrieved_docs=[],
                    classification='GREETING',
                ),
                sources=[],
                should_escalate=False,
                llm_used=False,
            )
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        
        # Step 2: Classify query
        classification = self.classifier.classify_query(query, retrieved_docs)
        
        # Step 3: Generate response
        llm_used = False
        
        if classification.should_escalate:
            # For escalated queries, provide escalation message
            answer = self._generate_escalation_message(classification)
        else:
            # Generate answer using LLM
            answer = self.generate_answer(query, retrieved_docs)
            llm_used = True
        
        # Step 4: Prepare sources
        sources = []
        for doc, score in retrieved_docs[:3]:  # Top 3 sources
            sources.append({
                'question': doc.metadata.get('question', 'N/A'),
                'category': doc.metadata.get('category', 'General'),
                'faq_id': doc.metadata.get('faq_id', 'N/A'),
                'similarity': score,
                'content_preview': doc.page_content[:200] + '...',
            })
        
        return RAGResponse(
            query=query,
            answer=answer,
            classification=classification,
            sources=sources,
            should_escalate=classification.should_escalate,
            llm_used=llm_used,
        )
    
    def _generate_greeting_response(self, query: str) -> str:
        """Generate a friendly greeting response"""
        query_lower = query.lower()
        
        if any(g in query_lower for g in ['hi', 'hello', 'hey', 'greetings', 'namaste']):
            return "Hello! 👋 Welcome to Rooman Technologies Support. I'm here to help you with any questions about our courses, programs, and services. How can I assist you today?"
        
        elif any(g in query_lower for g in ['good morning', 'morning']):
            return "Good morning! ☀️ Welcome to Rooman Technologies Support. How can I help you today?"
        
        elif any(g in query_lower for g in ['good afternoon', 'afternoon']):
            return "Good afternoon! 🌤️ Welcome to Rooman Technologies Support. What can I help you with?"
        
        elif any(g in query_lower for g in ['good evening', 'evening']):
            return "Good evening! 🌆 Welcome to Rooman Technologies Support. How may I assist you?"
        
        elif any(g in query_lower for g in ['how are you', 'how r u']):
            return "I'm doing great, thank you for asking! 😊 I'm here and ready to help you with information about Rooman's courses and programs. What would you like to know?"
        
        elif any(g in query_lower for g in ['thanks', 'thank you', 'thx']):
            return "You're very welcome! 😊 If you have any other questions about Rooman's programs, feel free to ask. Happy to help!"
        
        elif any(g in query_lower for g in ['bye', 'goodbye', 'see you', 'cya']):
            return "Goodbye! 👋 Thank you for using Rooman Technologies Support. Have a great day, and don't hesitate to return if you have more questions!"
        
        else:
            return "Hello! 👋 I'm the Rooman Technologies Support Assistant. I can help you with questions about our courses, admissions, schedules, and more. What would you like to know?"
    
    def _generate_escalation_message(self, classification: QueryClassification) -> str:
        """Generate a helpful escalation message"""
        message = (
            "I appreciate your question, but I'd like to connect you with our support team "
            "to provide you with the most accurate and personalized assistance.\n\n"
            "**How to reach us:**\n"
            "- 📧 Email: online@rooman.net\n"
            "- 📞 Phone: 8069451000\n"
            "- 🏢 Visit: #30, 12th Main, 1st Stage Rajajinagar, Bengaluru - 560 010\n\n"
            "Our team will be happy to help you with your specific needs!"
        )
        
        return message
    
    def stream_answer(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
    ):
        """
        Stream the answer generation (for Streamlit).
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents with scores
            
        Yields:
            Answer chunks
        """
        # Prepare context
        context_parts = []
        for idx, (doc, score) in enumerate(retrieved_docs, 1):
            content = doc.page_content
            context_parts.append(f"[FAQ {idx}]\n{content}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )
        
        # Stream response
        for chunk in self.llm.stream(prompt):
            yield chunk


if __name__ == "__main__":
    # Test the RAG pipeline
    print("=" * 60)
    print("🧪 Testing RAG Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Test queries
    test_queries = [
        "What networking courses does Rooman offer?",
        "Tell me about the Data Science program",
        "I need a refund for my course immediately",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"❓ Query: {query}")
        print('=' * 60)
        
        # Process query
        response = pipeline.process_query(query)
        
        # Display results
        print(f"\n📊 Classification: {response.classification.classification}")
        print(f"🎯 Confidence: {response.classification.confidence_score:.2%}")
        print(f"⚠️  Escalate: {response.should_escalate}")
        
        if response.sources:
            print(f"\n📚 Sources ({len(response.sources)}):")
            for idx, source in enumerate(response.sources, 1):
                print(f"  {idx}. {source['question']} (Similarity: {source['similarity']:.2%})")
        
        print(f"\n💬 Answer:\n{response.answer}\n")
        
        if response.classification.escalation_reasons:
            print(f"⚠️  Escalation Reasons:")
            for reason in response.classification.escalation_reasons:
                print(f"  - {reason}")
