# query_llm_with_retrieved_docs.py
# Simple RAG implementation with HybridSearcher and OpenAI

import os
import re
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import HybridSearcher

load_dotenv()

def query_llm_with_retrieved_docs(query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Query LLM with retrieved documents using hybrid search.
    
    Args:
        query: User's question
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, referenced_documents)
    """
    # Initialize components
    searcher = HybridSearcher()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Step 1: Retrieve documents
    documents = searcher.search(query, top_k)
    print(f"""Count of documents retrieved: {len(documents)}""")
    
    if not documents:
        return "Sorry, the search did not return any documents relevant to your query.", []
    
    # Step 2: Format context for LLM
    context = ""
    for i, doc in enumerate(documents, 1):
        text = doc.get('text', doc.get('preview', ''))
        context += f"Document {i}: {text}\n\n"
    
    # Step 3: Create prompt
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the information provided in the context documents below. 

IMPORTANT INSTRUCTIONS:
- Use only the information from the provided documents
- If you cannot find relevant information in the documents, respond with: "Sorry, none of the documents retrieved were helpful in answering your query."
- Do not use phrases like "According to Document 1" or "Document 2 states"
- Simply provide a direct answer based on the information
- At the end of your response, add a <References> section listing the document numbers that were helpful. Only include documents that directly contributed to your answer.
- Format: <References>Document 1, Document 3</References>

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""
    
    # Step 4: Get LLM response
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Step 5: Parse response and references
        answer, referenced_docs = parse_response(full_response, documents)
        print(f"""Count of documents helpful: {len(referenced_docs)}""")
        
        
        return answer, referenced_docs
        
    except Exception as e:
        return f"Error: {str(e)}", []


def parse_response(response: str, documents: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Parse LLM response to separate answer from references.
    
    Args:
        response: Full LLM response
        documents: Original retrieved documents
        
    Returns:
        Tuple of (answer, referenced_documents)
    """
    # Extract references section
    references_pattern = r'<References>(.*?)</References>'
    references_match = re.search(references_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if references_match:
        # Get clean answer (everything before <References>)
        answer = response[:references_match.start()].strip()
        
        # Parse referenced document numbers
        references_text = references_match.group(1)
        doc_numbers = re.findall(r'Document\s+(\d+)', references_text, re.IGNORECASE)
        
        # Get referenced documents
        referenced_docs = []
        for doc_num in doc_numbers:
            try:
                doc_index = int(doc_num) - 1  # Convert to 0-based index
                if 0 <= doc_index < len(documents):
                    referenced_docs.append(documents[doc_index])
            except (ValueError, IndexError):
                continue
                
        return answer, referenced_docs
    else:
        # No references found, return full response
        return response, []         # ('ColBERT stands for...', [{chunk_id:41, hybrid_score:0.01622, keyword_score:4.1093, semantic_score:0.3465, text: 'bi-encoder, it's still reasonably fast and...'},
                                    #                            {chunk_id:40, hybrid_score:0.01621, keyword_score:3.1093, semantic_score:0.3765, text: 'document pair and you won't have...'}
                                    # ])


# Example usage
if __name__ == "__main__":
    # Test the function
    test_query = "Tell me about ColBERT."
    
    print("Testing RAG Query...")
    print(f"Query: {test_query}")
    print("-" * 50)
    
    answer, references = query_llm_with_retrieved_docs(test_query)
    
    print("ANSWER:")
    print(answer)
    print()
    
    if references:
        print(f"REFERENCED DOCUMENTS ({len(references)}):")
        for i, doc in enumerate(references, 1):
            chunk_id = doc.get('chunk_id', 'Unknown')
            score = doc.get('hybrid_score', 0.0)
            print(f"{i}. Document {chunk_id} (Score: {score:.4f})")
    else:
        print("No helpful documents were found!")
