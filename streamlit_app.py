# streamlit_app.py
# Ask About RAG! - Streamlit Frontend
# Simple interface for querying the RAG system with optional document display

import streamlit as st
import os
from typing import Dict, List, Tuple
from query_llm_with_retrieved_docs import query_llm_with_retrieved_docs

# Page configuration
st.set_page_config(
    page_title="Ask About RAG!",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .query-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .answer-container {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 1rem;
    }
    .document-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .document-header {
        font-weight: bold;
        color: #856404;
        margin-bottom: 0.5rem;
    }
    .document-text {
        color: #333;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .error-container {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Ask About RAG!</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Query a RAG system about RAG and optionally view the source documents</p>', unsafe_allow_html=True)
    
    # Query input section
    with st.container():
        #st.markdown('<div class="query-container">', unsafe_allow_html=True)
        
        # User query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., Tell me about ColBERT",
            help="Ask any question about the RAG system or related topics"
        )
        
        # Show documents toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            show_documents = st.checkbox(
                "Show retrieved documents text",
                value=False,
                help="Check this to see the full text of documents that were used to generate the answer"
            )
        
        with col2:
            search_button = st.button(
                "Search",
                type="primary",
                use_container_width=True
            )
        
        #st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query when button is clicked or Enter is pressed
    if search_button and query.strip():
        process_query(query.strip(), show_documents)


def process_query(query: str, show_documents: bool):
    """Process the user query and display results."""
    
    try:
        # Show loading spinner
        with st.spinner('Thinking... Searching through documents and generating answer...'):
            # Call the RAG function
            answer, referenced_docs = query_llm_with_retrieved_docs(query)
        
        # Display the answer
        #st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown("### Answer")
        st.markdown(f'<div class="document-text">{answer}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show document count and references info
        if referenced_docs:
            st.info(f"Answer generated using {len(referenced_docs)} referenced document(s)")        
        
        # Display referenced documents if requested
        if show_documents and referenced_docs:
            st.markdown("---")
            st.markdown("### Referenced Documents")
            
            for i, doc in enumerate(referenced_docs, 1):
                display_document(doc, i)       
        
            
    except Exception as e:
        # Handle errors gracefully
        st.markdown('<div class="error-container">', unsafe_allow_html=True)
        st.markdown("### Error")
        st.markdown(f"An error occurred while processing your query: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show debug info in expander
        with st.expander("Debug Information"):
            st.text(f"Error type: {type(e).__name__}")
            st.text(f"Error message: {str(e)}")
            
            # Check if required environment variables are set
            st.text("Environment check:")
            st.text(f"- OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")

def display_document(doc: Dict, index: int):
    """Display a single document in a formatted container."""
    
    chunk_id = doc.get('chunk_id', 'Unknown')
    text = doc.get('text', doc.get('preview', 'No text available'))
    
    # Create expandable document section
    with st.expander(f"Document {index} (ID: {chunk_id})"):
        # Document text
        st.markdown("**Document Text:**")
        st.markdown(f'<div class="document-text">{text}</div>', unsafe_allow_html=True)

# Sidebar with information
def show_sidebar():
    """Display sidebar with app information and tips."""
    with st.sidebar:
        st.markdown("### About this app:")
        st.markdown("""
        This app uses a Retrieval-Augmented Generation (RAG) system to answer your questions about the RAG technology based on a curated knowledge base.
        
        **How it works:**
        1. Searches through documents using hybrid search (keyword + semantic)
        2. Uses GPT to generate answers based on relevant documents
        3. Optionally shows you the source documents
        
        **Tips:**
        - Be specific in your questions
        - Try questions about RAG, ColBERT, semantic search, etc.
        - Enable "Show documents" to see sources
        """)
        
        st.markdown("---")
        st.markdown("### Example Question")
        st.markdown("""
        - "Tell me about ColBERT"        
        """)

if __name__ == "__main__":
    show_sidebar()
    main()
