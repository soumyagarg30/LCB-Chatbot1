from __future__ import annotations

from types import SimpleNamespace

from rag_system import RAGSystem


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
    rag.vectorstore = FakeVectorStore([
        SimpleNamespace(page_content="Alpha beta gamma", metadata={})
    ])
    rag.in_memory_docs = []

    context = rag.search_relevant_context("beta", k=2)

    assert "Alpha beta gamma" in context
