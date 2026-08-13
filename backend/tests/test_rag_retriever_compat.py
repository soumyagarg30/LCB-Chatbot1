from __future__ import annotations

from types import SimpleNamespace

from rag_system import RAGSystem
from langchain_core.documents import Document


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def _get_relevant_documents(self, query):
        return [doc for doc in self.docs if query.lower() in doc.page_content.lower()]


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, **kwargs):
        return FakeRetriever(self.docs)


def test_search_relevant_context_uses_compat_retriever_api():
    rag = object.__new__(RAGSystem)
    docs = [SimpleNamespace(page_content="Alpha beta gamma", metadata={})]
    rag.vectorstore = FakeVectorStore(docs)
    rag.in_memory_docs = docs

    context = rag.search_relevant_context("beta", k=2)

    assert "Alpha beta gamma" in context


def test_keyword_retrieval_prioritizes_microorganism_mechanisms():
    rag = object.__new__(RAGSystem)
    rag.vectorstore = None
    rag.in_memory_docs = [
        Document(
            page_content="General operational information about materials and functions.",
            metadata={"type": "uploaded_document"},
        ),
        Document(
            page_content=(
                "Microbial Mechanism & Functions:\n"
                "- Azospirillum - Nitrogen fixation\n"
                "- PSB - Unlocks phosphorus"
            ),
            metadata={"type": "mechanism", "source_type": "brand_kb"},
        ),
    ]

    docs = rag._keyword_ranked_docs(
        "Can you enumerate every named microorganism and its functions?", k=1
    )

    assert docs[0].metadata["type"] == "mechanism"
    assert "Nitrogen fixation" in docs[0].page_content
