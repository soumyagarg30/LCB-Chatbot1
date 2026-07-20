# rag_system.py
import json
import os
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Dict, Any, Optional


class RAGSystem:
    def __init__(self, gemini_api_key: str, collection_name: str = "brand_kb"):
        """
        RAG system for brand & product knowledge (Gemini + Chroma).

        Args:
            gemini_api_key: Google AI Studio (Gemini) API key
            collection_name: ChromaDB collection name
        """
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")

        self.gemini_api_key = gemini_api_key
        self.collection_name = collection_name

        # Gemini embeddings (optional; fall back to keyword-based retrieval if unavailable)
        self.embeddings = None
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_api_key
            )
        except Exception as e:
            print(f"⚠️ Gemini embeddings unavailable; using keyword-based fallback: {e}")

        # ChromaDB persistent client
        backend_dir = Path(__file__).resolve().parent
        chroma_path = backend_dir / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Text splitter (only used when a field is long)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", ";", "•", "—", "- "]
        )

        # Vector DB + cached summary text
        self.vectorstore = None
        self.in_memory_docs: List[Document] = []
        self.profile_summary = ""  # kept for backward-compat with your app.py
        self.data_cache: Dict[str, Any] = {}

    # --------------- Data loading ---------------

    def _brand_json_path(self, json_path: Optional[str] = None) -> Path:
        """Resolve brand_data.json path (defaults to same folder as this file)."""
        if json_path:
            return Path(json_path)
        return Path(__file__).parent / "brand_data.json"

    def load_brand_data(self, json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load brand/product/FAQ data from JSON.
        Expected top-level keys: brand, products[], mechanism?, faqs[]
        """
        path = self._brand_json_path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Brand data not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Minimal validation
        if "brand" not in data or "name" not in data["brand"]:
            raise ValueError("brand_data.json must include: { 'brand': { 'name': ... } }")
        data.setdefault("products", [])
        data.setdefault("faqs", [])
        data.setdefault("mechanism", {})
        self.data_cache = data
        return data

    # --------------- Summary ---------------

    def _generate_summary_text(self, data: Dict[str, Any]) -> str:
        """Generate a concise high-level summary of the knowledge base."""
        brand = data.get("brand", {})
        name = brand.get("name", "Unknown Brand")
        tagline = brand.get("tagline", "")
        desc = brand.get("description", "")

        parts = [
            f"Brand Knowledge Base for: {name}",
        ]
        if tagline:
            parts.append(f"Tagline: {tagline}")
        if desc:
            parts.append(f"Description: {desc}")

        products = data.get("products", [])
        parts.append(f"Total product/crop entries: {len(products)}")
        if products:
            crop_list = ", ".join([p.get("crop", "N/A") for p in products[:12]])
            if len(products) > 12:
                crop_list += ", ..."
            parts.append(f"Crops/segments covered: {crop_list}")

        benefits = brand.get("benefits", [])
        if benefits:
            parts.append("Key Benefits:")
            for b in benefits[:8]:
                parts.append(f"- {b}")

        return "\n".join(parts)

    # --------------- Documents ---------------

    def _maybe_split(self, text: str) -> List[str]:
        """Split overly long strings into sub-chunks (keeps atomicity reasonable)."""
        if not text or len(text) <= 1000:
            return [text] if text else []
        return self.text_splitter.split_text(text)

    def _create_documents_from_brand(self, data: Dict[str, Any]) -> List[Document]:
        """
        Create atomic documents:
          - One brand overview doc
          - One per product/crop (plus sub-chunks for long fields)
          - One per FAQ
          - Optional mechanism overview
        """
        docs: List[Document] = []

        # Brand overview
        brand = data.get("brand", {})
        overview_lines = [
            f"Brand: {brand.get('name', 'N/A')}",
            f"Tagline: {brand.get('tagline', 'N/A')}",
            f"Description: {brand.get('description', 'N/A')}",
        ]
        if brand.get("benefits"):
            overview_lines.append("Benefits:")
            overview_lines.extend([f"- {b}" for b in brand["benefits"]])

        if brand.get("purchase_links"):
            overview_lines.append("Purchase Links:")
            overview_lines.extend([f"- {u}" for u in brand["purchase_links"]])

        docs.append(Document(
            page_content="\n".join([l for l in overview_lines if l]),
            metadata={"type": "brand_overview"}
        ))

        # Mechanism (if present)
        mech = data.get("mechanism", {})
        mech_lines = []
        if mech.get("microbes"):
            mech_lines.append("Microbial Mechanism & Functions:")
            mech_lines.extend([f"- {m}" for m in mech["microbes"]])
        if mech_lines:
            for chunk in self._maybe_split("\n".join(mech_lines)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"type": "mechanism"}
                ))

        # Products / Crops
        for prod in data.get("products", []):
            crop = prod.get("crop", "N/A")
            applications = prod.get("applications", [])
            mech_text = prod.get("mechanism", "")

            base = [f"Crop/Product: {crop}"]
            if applications:
                base.append("Applications:")
                base.extend([f"- {a}" for a in applications])
            if mech_text:
                base.append(f"Mechanism: {mech_text}")

            full_text = "\n".join(base)
            for chunk in self._maybe_split(full_text):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"type": "product", "crop": crop}
                ))

        # FAQs
        for idx, faq in enumerate(data.get("faqs", []), start=1):
            q = faq.get("q", "").strip()
            a = faq.get("a", "").strip()
            if not q and not a:
                continue
            qa_text = f"Q: {q}\nA: {a}"
            for chunk in self._maybe_split(qa_text):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"type": "faq", "id": idx}
                ))

        return docs

    # --------------- Build Vectorstore ---------------

    def build_vectorstore(self, json_path: Optional[str] = None):
        """
        Builds the vector database from brand_data.json (or custom path) and
        caches a high-level summary.
        """
        print("🔧 Building vector database from JSON...")

        data = self.load_brand_data(json_path=json_path)
        self.profile_summary = self._generate_summary_text(data)
        print("✅ Brand summary generated and cached.")

        documents = self._create_documents_from_brand(data)
        print(f"📄 Created {len(documents)} atomic brand/product/FAQ documents.")

        self.in_memory_docs = documents

        if self.embeddings is not None:
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    client=self.chroma_client
                )
                print("✅ Vector database built successfully!")
            except Exception as e:
                print(f"⚠️ Chroma build failed; using keyword-based fallback: {e}")
                self.vectorstore = None
        else:
            self.vectorstore = None
            print("✅ Keyword-based retrieval ready.")

    # --------------- Retrieval ---------------

    def get_summary_document(self) -> str:
        """Return the cached high-level summary text."""
        return self.profile_summary

    def _keyword_ranked_docs(self, query: str, k: int = 5) -> List[Document]:
        q_tokens = {t for t in query.lower().replace("/", " ").split() if t}
        scored = []
        for doc in self.in_memory_docs:
            text = doc.page_content.lower()
            token_count = sum(1 for t in q_tokens if t in text)
            if token_count == 0:
                continue
            scored.append((token_count, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant documents using Chroma when available, otherwise fall back to
        keyword-based matching over cached documents.
        """
        if self.vectorstore is not None:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': k, 'fetch_k': 25, 'lambda_mult': 0.25}
            )
            docs = retriever.get_relevant_documents(query)
        else:
            docs = self._keyword_ranked_docs(query, k=k)

        # De-duplicate by page_content
        unique_docs: List[Document] = []
        seen = set()
        for d in docs:
            if d.page_content not in seen:
                unique_docs.append(d)
                seen.add(d.page_content)

        print(f"🔍 Retrieved {len(unique_docs)} unique contexts for LLM.")
        ctx_parts = []
        for i, d in enumerate(unique_docs, 1):
            ctx_parts.append(f"Relevant Information {i}:\n{d.page_content}")
        return "\n\n".join(ctx_parts)

    # --------------- Backward-compat for app.py ---------------

    def get_personal_info(self) -> Dict[str, Any]:
        """
        Kept to avoid changing your existing app.py.
        Returns brand name/tagline in the same shape app.py expects.
        """
        if not self.data_cache:
            # Ensure data is loaded even if build_vectorstore() hasn't been called yet
            self.load_brand_data()
        brand = self.data_cache.get("brand", {})
        return {
            "name": brand.get("name", "Brand"),
            "title": brand.get("tagline", "Brand Assistant")
        }

    # Optional: explicit brand getter if you update app.py later
    def get_brand_info(self) -> Dict[str, Any]:
        if not self.data_cache:
            self.load_brand_data()
        brand = self.data_cache.get("brand", {})
        return {
            "name": brand.get("name", ""),
            "tagline": brand.get("tagline", ""),
            "description": brand.get("description", ""),
            "benefits": brand.get("benefits", []),
            "purchase_links": brand.get("purchase_links", [])
        }
