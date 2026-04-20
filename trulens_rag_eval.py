import time
import numpy as np
from trulens.core import TruSession, Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI
from trulens.apps.app import TruApp
from trulens.dashboard.run import run_dashboard

from rag_app import RAGApp


# Set up TruLens for evaluation of the RAG system

session = TruSession()
provider = OpenAI()

f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons, 
    name="Answer Relevance",
    selectors={
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        )    
    }
)

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons, 
    name="Groundedness",
    selectors={
        "source": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True,      # combine all context chunks into one blob
        ),
        "statement": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        )
    }
)

f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons, 
    name="Context Relevance",
    selectors={
        "question": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False,     # score each chunk individually, then average
        )
    },
    agg=np.mean
)


rag = RAGApp()
tru_rag = TruApp(
    rag,
    app_name="hybrid-rag",
    app_version="v1",
    feedbacks=[f_answer_relevance, f_groundedness, f_context_relevance],
)


# Test Queries
test_queries = [
    "What is semantic search",
    "How does hybrid search work?",
    "What is Reciprocal Rank Fusion?",
    "What is COLBERT?",
    "What is TF-IDF?",
    "What is keyword search?"
]


# Run Evaluation on a query
def run_eval(query: str):
    print(f"\nQuery: {query}")

    with tru_rag as recording:
        answer = rag.query(query)

    print(f"Answer: {answer[:200]}...")
    #print(f"References retrieved: {len(result['context'])}")

    # retrieve_feedback_results() waits for scores — no need for manual polling
    feedback_results = recording.retrieve_feedback_results(timeout=60)

    if feedback_results is not None and not feedback_results.empty:
        print("\nScores:")
        for metric_name in feedback_results.columns:
            score = feedback_results[metric_name].iloc[0]            
            print(f"  {metric_name}: {score:.2f}")


if __name__ == "__main__":
    for query in test_queries:
        run_eval(query)

    print("\n── Leaderboard ──────────────────────────────")
    leaderboard = session.get_leaderboard()
    print(leaderboard.to_string())

    # Optionally launch the TruLens dashboard
    run_dashboard(session)




