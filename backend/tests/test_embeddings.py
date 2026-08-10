from rag_system import RAGSystem
from langchain_community.vectorstores import Chroma


def test_local_embedding_adapter_returns_chroma_compatible_lists():
    rag = RAGSystem(collection_name="embedding_type_test")
    embeddings = rag.embeddings

    assert isinstance(embeddings.embed_documents(["soil health"]), list)
    assert isinstance(embeddings.embed_documents(["soil health"])[0], list)
    assert isinstance(embeddings.embed_query("soil health"), list)

    vectorstore = Chroma.from_texts(
        ["Healthy soil supports crop growth."],
        embedding=embeddings,
        collection_name="embedding_type_test",
        client=rag.chroma_client,
    )
    assert vectorstore.similarity_search("soil", k=1)

    try:
        rag.chroma_client.delete_collection("embedding_type_test")
    except Exception:
        pass


def test_fresh_chroma_client_can_replace_a_deleted_collection():
    rag = RAGSystem(collection_name="collection_replacement_test")
    collection_name = rag.collection_name
    Chroma.from_texts(
        ["old collection"], embedding=rag.embeddings,
        collection_name=collection_name, client=rag.chroma_client,
    )

    rag.vectorstore = None
    rag.chroma_client.delete_collection(collection_name)
    rag.chroma_client = rag._create_chroma_client()
    replacement = Chroma.from_texts(
        ["new collection"], embedding=rag.embeddings,
        collection_name=collection_name, client=rag.chroma_client,
    )

    assert replacement.similarity_search("new", k=1)
    rag.chroma_client.delete_collection(collection_name)
