from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from query_llm_with_retrieved_docs import query_llm_with_retrieved_docs

class RAGApp:

    def __init__(self):
        self._last_references = []


    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={           
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return"
        }
    )
    def retrieve_and_answer(self, user_query: str) -> list:
        """Retrieves and caches context chunks."""
        answer, references = query_llm_with_retrieved_docs(user_query)
        self._last_answer = answer
        self._last_references = references
        return references


    @instrument(
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "user_query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def query(self, user_query: str) -> str:
        """Uses cached result from retrieve()."""
        self.retrieve_and_answer(user_query)           # populates self._last_answer
        return self._last_answer