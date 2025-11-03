# Ask About RAG! 🔍

A simple **Retrieval-Augmented Generation (RAG)** system that answers questions about RAG technology using hybrid search (semantic + keyword matching).

## What it does

- Ask questions about RAG technology
- Get AI-generated answers based on a curated knowledge base
- See the source documents used to generate the answer (optional)

## Example questions

- "What is RAG?"
- "Tell me about ColBERT"
- "How does semantic search work?"

## How it works

1. **Hybrid Search**: Combines BM25 keyword search with FAISS semantic search
2. **Document Retrieval**: Finds relevant chunks from the knowledge base
3. **AI Generation**: Uses OpenAI GPT to generate answers from retrieved documents
4. **Source Display**: Shows you which documents were used (optional)

## Project Structure

```
├── streamlit_app.py              # Main web app
├── query_llm_with_retrieved_docs.py  # Core RAG function
├── hybrid_search.py              # Combines keyword + semantic search
├── semantic_search.py            # FAISS vector search
├── keyword_search.py             # BM25 keyword search
├── faiss_store.py               # FAISS index management
├── embedding_generator.py       # Generate embeddings for documents
├── bm25_preprocessor.py         # BM25 text preprocessing
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── data/                        # Knowledge base and indices
    ├── embeddings/              # FAISS vector indices
    │   ├── rag_transcript_index.faiss
    │   ├── rag_transcript_index_config.json
    │   └── rag_transcript_index_mapping.json
    ├── bm25_indices/           # BM25 keyword indices  
    │   └── rag_transcript_bm25.json
    └── metadata/               # Processed document chunks
        └── RAG_Transcript_chunks.json
```

## Requirements

- Python 3.11+
- OpenAI API key
- Create `.env` file from `.env.example` template
- Dependencies in `requirements.txt`
