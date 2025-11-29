"""
Rooman SupportAssistant - Premium Dark Theme Streamlit Application
AI-powered FAQ resolution system with escalation management
"""

import streamlit as st
from pathlib import Path
import uuid
from datetime import datetime

from src.config import config
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.escalation_manager import EscalationManager


# Page configuration
st.set_page_config(
    page_title="Rooman SupportAssistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Professional Light Theme CSS
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    /* ==================== GLOBAL STYLES ==================== */
    * {
        font-family: 'Inter', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: 0.01em;
    }
    
    /* ChatGPT-style Dark Background */
    .main {
        background: #212121 !important;
    }
    
    .stApp {
        background: #212121 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #212121 !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #212121 !important;
    }
    
    [data-testid="stBottomBlockContainer"] {
        background: #212121 !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background-color: transparent;}
    
    /* ==================== HERO HEADER ==================== */
    .hero-header {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
    }
    
    .hero-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    
    .hero-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* ==================== CHAT MESSAGES ==================== */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 1rem 0 !important;
        margin: 0.5rem 0 !important;
        box-shadow: none !important;
    }
    
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage:hover {
        box-shadow: none !important;
        border: none !important;
    }
    
    /* User Message - No Container */
    .stChatMessage[data-testid="user-message"] {
        background: transparent !important;
        border: none !important;
        color: #ececf1 !important;
    }
    
    /* Bot Message - No Container */
    .stChatMessage[data-testid="assistant-message"] {
        background: transparent !important;
        border: none !important;
        color: #ececf1 !important;
    }
    
    /* General text color for dark background */
    .stChatMessage p {
        color: #ececf1 !important;
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #2d2d2d;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Section Headers */
    [data-testid="stSidebar"] h2 {
        font-size: 0.875rem;
        text-transform: uppercase;
        color: #cccccc !important;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d2d2d;
        letter-spacing: 0.05em;
    }
    
    /* ==================== STATUS INDICATORS ==================== */
    .status-indicator {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: #1a1a1a;
        border-left: 3px solid #4A90E2;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #ffffff;
    }
    
    .status-indicator .icon {
        margin-right: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* ==================== CARDS ==================== */
    .glass-card {
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border-color: #3d3d3d;
    }
    
    /* ==================== METRIC CARDS ==================== */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px);
    }
    
    .metric-card .label {
        font-size: 0.75rem;
        color: #cccccc;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 600;
        color: #4A90E2;
    }
    
    /* ==================== SLIDERS ==================== */
    .stSlider {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #2d2d2d;
        margin: 0.75rem 0;
    }
    
    /* ==================== SOURCE CARDS ==================== */
    .source-card {
        background: #f6f8fa;
        border-left: 3px solid #4A90E2;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        background: #eef1f5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .source-card .question {
        color: #24292e;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    .source-card .meta {
        color: #586069;
        font-size: 0.85rem;
    }
    
    /* ==================== ESCALATION NOTICE ==================== */
    .escalation-notice {
        background: #fff8e1;
        border: 1px solid #ffb74d;
        border-left: 4px solid #ff9800;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .escalation-notice strong {
        color: #e65100;
        font-size: 1rem;
    }
    
    /* ==================== EXPANDER ==================== */
    .streamlit-expanderHeader {
        background: #f6f8fa !important;
        border: 1px solid #e1e4e8 !important;
        border-radius: 6px !important;
        color: #24292e !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #eef1f5 !important;
        border-color: #d1d5da !important;
    }
    
    /* ==================== CHAT INPUT ==================== */
    .stChatInputContainer {
        border-top: 1px solid #2d2d2d;
        background: #212121;
        padding: 2rem 0;
    }
    
    .stChatInput {
        border-radius: 24px !important;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Chat input wrapper container */
    .stChatInput > div {
        background: #2d2d2d !important;
        border-radius: 24px !important;
        border: 1px solid #404040 !important;
        padding: 0.75rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #565869 !important;
        box-shadow: 0 0 0 1px #565869 !important;
    }
    
    /* Textarea styling */
    .stChatInput textarea {
        background: transparent !important;
        color: #ececf1 !important;
        border: none !important;
        font-size: 1rem !important;
        line-height: 1.75 !important;
        padding: 0.75rem 0 !important;
        min-height: 28px !important;
        max-height: 200px !important;
    }
    
    /* Animated placeholder with fading effect */
    .stChatInput textarea::placeholder {
        color: #8e8ea0 !important;
        animation: placeholderFade 3s ease-in-out infinite;
    }
    
    @keyframes placeholderFade {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .stChatInput textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stChatInput textarea:focus::placeholder {
        animation: none;
        opacity: 0.6;
    }
    
    /* Submit button with icon and label */
    [data-testid="stChatInputSubmitButton"] {
        background: #19c37d !important;
        border: none !important;
        border-radius: 12px !important;
        min-width: 80px !important;
        height: 40px !important;
        padding: 0 1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        transition: all 0.2s ease !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
    }
    
    /* Add "Send" label using pseudo-element */
    [data-testid="stChatInputSubmitButton"]::after {
        content: "Send";
        color: white !important;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background: #1a9f6a !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(25, 195, 125, 0.3) !important;
    }
    
    [data-testid="stChatInputSubmitButton"]:active:not(:disabled) {
        transform: translateY(0);
    }
    
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: #40414f !important;
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    
    [data-testid="stChatInputSubmitButton"]:disabled::after {
        content: "Send";
    }
    
    [data-testid="stChatInputSubmitButton"] svg {
        width: 18px !important;
        height: 18px !important;
        color: white !important;
        flex-shrink: 0;
    }
    
    /* ==================== DIVIDERS ==================== */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #2d2d2d;
    }
    
    /* Sidebar specific dividers */
    [data-testid="stSidebar"] hr {
        border-top: 1px solid #2d2d2d;
    }
    
    /* ==================== CONFIDENCE INDICATORS ==================== */
    .confidence-high {
        color: #28a745;
        font-weight: 600;
    }
    
    .confidence-medium {
        color: #ff9800;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: 600;
    }
    
    /* ==================== METRICS ==================== */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e1e4e8;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f6f8fa;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5da;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #c1c5ca;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system components (cached)"""
    with st.spinner("🚀 Initializing SupportAssistant..."):
        try:
            vector_store = VectorStore()
            pipeline = RAGPipeline(vector_store=vector_store)
            escalation_manager = EscalationManager()
            return pipeline, escalation_manager, True, None
        except Exception as e:
            return None, None, False, str(e)


def get_confidence_class(score: float) -> str:
    """Get CSS class based on confidence score"""
    if score >= 0.7:
        return "confidence-high"
    elif score >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_sources(sources, classification):
    """Display source FAQs with premium card design"""
    if not sources:
        return
    
    with st.expander(f"💡 **People also ask**", expanded=False):
        for idx, source in enumerate(sources, 1):
            confidence_class = get_confidence_class(source['similarity'])
            question = source['question']
            
            st.markdown(f"""
            <div class="source-card">
                <div class="question">Q{idx}: {question}</div>
                <div class="meta">
                    <span class="{confidence_class}">{source['similarity']:.0%} Match</span> • 
                    📁 {source['category']} • ID: {source['faq_id']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_escalation_stats(escalation_manager):
    """Display escalation statistics with Notion/Stripe style cards"""
    stats = escalation_manager.get_statistics()
    
    st.markdown("### 📊 Escalation Analytics")
    
    # Metric cards in 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Total</div>
            <div class="value">{stats.get('total', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Resolved</div>
            <div class="value">{stats.get('resolved', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Pending</div>
            <div class="value">{stats.get('pending', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Closed</div>
            <div class="value">{stats.get('closed', 0)}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    
    # Premium Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1><span class="icon">🎓</span> Rooman SupportAssistant</h1>
        <p>Intelligent FAQ Resolution • Powered by AI • GPU-Accelerated</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    pipeline, escalation_manager, success, error = initialize_system()
    
    if not success:
        st.error(f"❌ Failed to initialize system: {error}")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Premium Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Configuration")
        
        # Neon Status Indicators
        st.markdown("### 🔧 System Status")
        
        vectorstore_count = pipeline.vector_store.collection.count()
        
        st.markdown(f"""
        <div class="status-indicator">
            <span class="icon">✓</span> Ollama Connected
        </div>
        <div class="status-indicator">
            <span class="icon">📊</span> {vectorstore_count:,} Documents
        </div>
        <div class="status-indicator">
            <span class="icon">⚡</span> GPU: {config.EMBEDDING_DEVICE.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Glass Configuration Cards
        st.markdown("### 🎯 AI Configuration")
        
        # Confidence slider with icon label
        st.markdown("**🎯 Confidence Meter**")
        confidence_threshold = st.slider(
            "Confidence Meter",
            min_value=0.0,
            max_value=1.0,
            value=config.CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Lower values escalate more queries",
            key="confidence_slider",
            label_visibility="collapsed"
        )
        
        pipeline.classifier.confidence_threshold = confidence_threshold
        
        # Top-K slider with icon label
        st.markdown("**📄 Results to Retrieve**")
        top_k = st.slider(
            "Results to Retrieve",
            min_value=1,
            max_value=10,
            value=config.TOP_K_RESULTS,
            help="Number of FAQs to search",
            key="topk_slider",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Escalation Stats
        display_escalation_stats(escalation_manager)
        
        st.markdown("---")
        
        # Glass Action Buttons
        st.markdown("### 🛠️ Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 View Logs", use_container_width=True):
                st.session_state.show_escalations = True
        
        with col2:
            if st.button("📥 Export Report", use_container_width=True):
                csv_path = escalation_manager.export_to_csv()
                if csv_path:
                    st.success("✅ Exported!")
                else:
                    st.warning("No data")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Show escalations if requested
    if st.session_state.get('show_escalations', False):
        st.markdown("## 📋 Escalated Queries")
        
        escalations = escalation_manager.load_escalations(limit=20)
        
        if not escalations:
            st.info("No escalated queries yet")
        else:
            for idx, esc in enumerate(reversed(escalations), 1):
                with st.expander(f"{idx}. {esc['query'][:80]}... ({esc['status']})"):
                    st.markdown(f"**Timestamp:** {esc['timestamp']}")
                    st.markdown(f"**Status:** {esc['status']}")
                    st.markdown(f"**Confidence:** {esc['confidence_score']:.2%}")
                    st.markdown(f"**Reasons:**")
                    for reason in esc['reasons']:
                        st.markdown(f"- {reason}")
        
        if st.button("← Back to Chat"):
            st.session_state.show_escalations = False
            st.rerun()
        
        st.stop()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Show sources
                if "sources" in metadata:
                    display_sources(metadata["sources"], metadata.get("classification"))
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Rooman courses..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Process query
                response = pipeline.process_query(prompt, top_k=top_k)
                
                # Display answer
                st.markdown(response.answer)
        
        # Save assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "metadata": {
                "sources": response.sources,
                "classification": response.classification,
                "escalated": response.should_escalate,
            }
        })
        
        # Handle escalation logging
        if response.should_escalate:
            escalation_manager.log_escalation(
                query=prompt,
                reasons=response.classification.escalation_reasons,
                confidence_score=response.classification.confidence_score,
                sources=response.sources,
                session_id=st.session_state.session_id,
            )
        
        # Rerun to display the new message with metadata from history
        st.rerun()


if __name__ == "__main__":
    main()
