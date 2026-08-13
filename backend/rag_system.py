# rag_system.py
import json
import os
import hashlib
import math
import re
import threading
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from knowledge_ingester import ingest_sources


class LocalEmbeddingAdapter:
    def __init__(self, model_name: str, device: str, local_files_only: bool = False):
        if local_files_only:
            # Force offline behavior even if an inherited shell setting had
            # disabled it; the model cache has already been checked above.
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            for env_var in ('TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE'):
                os.environ.pop(env_var, None)

        model_name_or_path = model_name
        # Avoid a lengthy network retry at every startup when the model has not
        # yet been downloaded. Resolve a cached snapshot to its physical path so
        # SentenceTransformer never tries to contact Hugging Face for metadata.
        if local_files_only and not os.path.exists(model_name):
            hf_home = Path(os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface')))
            cache_name = 'models--' + model_name.replace('/', '--')
            snapshots = hf_home / 'hub' / cache_name / 'snapshots'
            cached_snapshots = list(snapshots.iterdir()) if snapshots.exists() else []
            if not cached_snapshots:
                raise ValueError(
                    f"Embedding model '{model_name}' is not cached. Set "
                    "EMBEDDING_ALLOW_DOWNLOAD=true once while online to enable semantic retrieval."
                )
            model_name_or_path = str(cached_snapshots[-1])

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            # Support running backend with system Python while using the repo's local virtualenv.
            venv_site_packages = Path(__file__).resolve().parents[1] / '.venv' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
            if venv_site_packages.exists():
                sys.path.insert(0, str(venv_site_packages))
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    pass
                else:
                    SentenceTransformer = SentenceTransformer
            if 'SentenceTransformer' not in locals():
                raise ValueError(
                    'The sentence_transformers package is not installed. '
                    'Install it via `pip install sentence-transformers`.'
                ) from e

        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY']
        saved_proxies = {k: os.environ.pop(k, None) for k in proxy_vars}
        try:
            try:
                self.impl = SentenceTransformer(
                    model_name_or_path=model_name_or_path,
                    device=device,
                    local_files_only=local_files_only,
                )
            except TypeError:
                # Some sentence-transformers versions do not accept local_files_only in the constructor.
                self.impl = SentenceTransformer(
                    model_name_or_path=model_name_or_path,
                    device=device,
                )
        except OSError as e:
            raise ValueError(
                'Failed to load the local sentence-transformers model. Ensure the model is cached locally and offline mode is enabled.'
            ) from e
        finally:
            for k, v in saved_proxies.items():
                if v is not None:
                    os.environ[k] = v

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Chroma expects a list of numeric lists. Passing a NumPy matrix makes
        # its internal truthiness check fail with "array is ambiguous".
        return self.impl.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query: str):
        return self.impl.encode([query], convert_to_numpy=True)[0].tolist()


class HashEmbeddingAdapter:
    """Small offline embedding fallback that keeps Chroma persistence available.

    It is intentionally deterministic and dependency-free. A sentence-transformer
    model, when available, remains the preferred semantic retriever; this fallback
    provides lexical similarity and ensures uploaded chunks are still saved in
    Chroma's SQLite database on a fresh/offline installation.
    """
    dimensions = 384

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[\w-]+", (text or "").lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str):
        return self._embed(query)


class RAGSystem:
    def __init__(self, collection_name: str = "brand_kb"):
        """
        RAG system for brand & product knowledge.

        Args:
            collection_name: ChromaDB collection name
        """
        self.collection_name = collection_name

        # Embeddings are initialized lazily to keep startup fast.
        self.embeddings = None
        self._embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self._embedding_device = os.getenv('EMBEDDING_DEVICE', 'cpu')
        download_allowed = os.getenv('EMBEDDING_ALLOW_DOWNLOAD', 'false').lower() in ('1', 'true', 'yes')
        local_only_env = os.getenv('EMBEDDING_LOCAL_FILES_ONLY')
        self._embedding_local_only = not download_allowed if local_only_env is None else local_only_env.lower() in ('1', 'true', 'yes')
        self._embeddings_initialized = False

        # ChromaDB persistent client
        backend_dir = Path(__file__).resolve().parent
        chroma_path = backend_dir / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path = chroma_path
        self.chroma_client = self._create_chroma_client()
        # Flask can receive an upload and a manual rebuild concurrently. Chroma
        # collection deletion/creation must be one atomic operation per process.
        self._vectorstore_lock = threading.RLock()

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

    def _create_chroma_client(self):
        """Create a client without cached handles to deleted collections."""
        return chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

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

        Every document gets a `source` + `source_type` in its metadata so the
        retrieval layer can always report where an answer came from.
        """
        docs: List[Document] = []
        brand_name = data.get("brand", {}).get("name", "Brand")

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
            metadata={
                "type": "brand_overview",
                "source_type": "brand_kb",
                "source": f"{brand_name} Brand Overview",
            }
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
                    metadata={
                        "type": "mechanism",
                        "source_type": "brand_kb",
                        "source": f"{brand_name} Knowledge Base - Mechanism",
                    }
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
                    metadata={
                        "type": "product",
                        "crop": crop,
                        "source_type": "brand_kb",
                        "source": f"{brand_name} Knowledge Base - {crop}",
                    }
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
                    metadata={
                        "type": "faq",
                        "id": idx,
                        "source_type": "brand_kb",
                        "source": f"{brand_name} FAQ #{idx}",
                    }
                ))

        return docs

    def _ensure_embeddings(self):
        """Initialize the embedding adapter only when it is needed."""
        if self._embeddings_initialized:
            return
        self._embeddings_initialized = True
        try:
            self.embeddings = LocalEmbeddingAdapter(
                model_name=self._embedding_model_name,
                device=self._embedding_device,
                local_files_only=self._embedding_local_only,
            )
            print(f"✅ Using local embedding model: {self._embedding_model_name} ({self._embedding_device}); local_only={self._embedding_local_only}")
        except Exception as e:
            self.embeddings = HashEmbeddingAdapter()
            print(f"ℹ️ Semantic embedding model unavailable; using offline Chroma hash embeddings: {e}")

    def prepare_in_memory_docs(self, json_path: Optional[str] = None) -> List[Document]:
        """Load brand JSON and uploaded chunks for keyword retrieval without building the vectorstore."""
        if not self.data_cache:
            self.load_brand_data(json_path=json_path)
        if not self.profile_summary:
            self.profile_summary = self._generate_summary_text(self.data_cache)

        if not self.in_memory_docs:
            documents = self._create_documents_from_brand(self.data_cache)
            try:
                from db_utils import get_all_document_chunks
                uploaded_chunks = get_all_document_chunks()
                if uploaded_chunks:
                    for chunk in uploaded_chunks:
                        chunk_text = chunk.get('chunk_text') or ''
                        if not chunk_text:
                            continue
                        filename = chunk.get('filename') or f"document {chunk.get('document_id')}"
                        meta = {
                            'type': 'uploaded_document',
                            'source': f"Uploaded Document: {filename}",
                            'source_type': chunk.get('source_type') or 'uploaded_document',
                            'chunk_index': chunk.get('chunk_index'),
                            'document_id': chunk.get('document_id')
                        }
                        documents.append(Document(
                            page_content=chunk_text,
                            metadata=meta
                        ))
                    print(f"📚 Prepared {len(uploaded_chunks)} uploaded chunk(s) for keyword retrieval.")
            except Exception as e:
                print(f"⚠️ Could not prepare uploaded document chunks: {e}")
            self.in_memory_docs = documents

        return self.in_memory_docs

    # --------------- Source ingestion ---------------

    def ingest_sources_into_vectorstore(self, files: Optional[List[str]] = None,
                                        urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Ingest local files or website URLs and append them to the current knowledge base.
        """
        ingested = ingest_sources(files=files, urls=urls)
        if not ingested:
            return []

        documents: List[Document] = []
        for item in ingested:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            raw_source_type = item.get("source_type", "unknown")
            raw_source = item.get("source", "")

            # Give a friendly, human-facing label depending on whether this came
            # from a website URL or a locally uploaded file.
            if raw_source_type in ("url", "website", "web"):
                friendly_source = f"Website: {raw_source}"
            elif raw_source_type in ("file", "local_file"):
                friendly_source = f"File: {raw_source}"
            else:
                friendly_source = raw_source or "Ingested Source"

            documents.append(Document(
                page_content=content,
                metadata={
                    "type": "ingested_source",
                    "source_type": raw_source_type,
                    "source": friendly_source,
                    "origin": raw_source,
                }
            ))

        if not documents:
            return []

        self.in_memory_docs.extend(documents)

        if self.embeddings is not None:
            try:
                if self.vectorstore is None:
                    self.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=self.collection_name,
                        client=self.chroma_client
                    )
                else:
                    self.vectorstore.add_documents(documents)
                print(f"✅ Added {len(documents)} ingested document(s) to the vector database.")
            except Exception as e:
                print(f"⚠️ Failed to add ingested documents to Chroma: {e}")
        else:
            print("✅ Added ingested document(s) to the in-memory knowledge cache.")

        return ingested

    # --------------- Build Vectorstore ---------------

    def build_vectorstore(self, json_path: Optional[str] = None):
        """Build the index serially so concurrent rebuilds cannot invalidate it."""
        with self._vectorstore_lock:
            return self._build_vectorstore(json_path=json_path)

    def _build_vectorstore(self, json_path: Optional[str] = None):
        """
        Builds the vector database from brand_data.json (or custom path),
        uploaded documents from SQLite, and caches a high-level summary.
        """
        print("🔧 Building vector database from JSON and uploaded documents...")

        documents = self.prepare_in_memory_docs(json_path=json_path)
        self.profile_summary = self._generate_summary_text(self.data_cache)
        print("✅ Brand summary generated and cached.")
        print(f"📄 Prepared {len(documents)} atomic brand/product/FAQ documents.")

        # The in-memory document cache already includes uploaded document chunks.
        self.in_memory_docs = documents

        self._ensure_embeddings()
        if self.embeddings is not None:
            try:
                # Chroma PersistentClient stores documents, metadata and embeddings in
                # chroma_db/chroma.sqlite3. Recreate this named collection so deleted
                # or replaced uploads cannot leave stale/duplicate chunks behind.
                self.vectorstore = None
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                except Exception:
                    pass  # The collection does not exist on the first run.
                # PersistentClient caches collection objects. Recreate it after a
                # delete so no LangChain wrapper can use the old collection UUID.
                self.chroma_client = self._create_chroma_client()
                # Index in bounded batches. Large upload libraries otherwise send
                # every chunk through the embedding model at once and can exhaust
                # local CPU/RAM before Chroma receives any vectors.
                batch_size = max(1, int(os.getenv('CHROMA_INGEST_BATCH_SIZE', '64')))
                first_batch = documents[:batch_size]
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    client=self.chroma_client
                )
                for start in range(batch_size, len(documents), batch_size):
                    self.vectorstore.add_documents(documents[start:start + batch_size])
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
        stop_words = {
            "a", "about", "an", "and", "are", "can", "could", "every", "explain",
            "for", "how", "i", "in", "is", "it", "its", "me", "named", "of",
            "please", "tell", "the", "to", "what", "which", "with", "you",
        }

        def normalized_tokens(value: str) -> set[str]:
            tokens = set()
            for token in re.findall(r"[a-z0-9]+", (value or "").lower()):
                if token in stop_words:
                    continue
                # Lightweight normalization keeps the offline retriever useful
                # for singular/plural wording without adding another dependency.
                if len(token) > 4 and token.endswith("ies"):
                    token = token[:-3] + "y"
                elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                    token = token[:-1]
                if token in {"microbe", "microbial", "microorganism"}:
                    token = "microorganism"
                tokens.add(token)
            return tokens

        q_tokens = normalized_tokens(query)
        scored = []
        for doc in self.in_memory_docs:
            doc_tokens = normalized_tokens(doc.page_content)
            token_count = len(q_tokens & doc_tokens)
            if token_count == 0:
                continue
            metadata = doc.metadata or {}
            boost = 0
            if metadata.get("type") == "mechanism" and q_tokens & {"microorganism", "mechanism", "function"}:
                boost = 10
            scored.append((token_count + boost, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def _document_source_label(self, doc: Document) -> str:
        """Resolve a human-readable source label for a single retrieved document."""
        md = doc.metadata or {}
        source_type = (md.get("source_type") or md.get("type") or "").lower()
        label = md.get("source") or md.get("source_name") or md.get("filename")

        if source_type in ("brand_kb", "brand knowledge base"):
            return "Brand Knowledge Base"
        if source_type in ("uploaded_document", "file_upload", "uploaded", "uploaded_document_chunk"):
            return label or "Uploaded Document"
        if source_type in ("url", "website", "web", "ingested_source"):
            return label or "Website Source"

        if label:
            return label
        if source_type:
            return str(source_type).replace("_", " ").title()
        return "Knowledge Base"

    def _document_source_info(self, doc: Document) -> Dict[str, Any]:
        """Structured source info (label + type) for a single retrieved document."""
        md = doc.metadata or {}
        return {
            "label": self._document_source_label(doc),
            "source_type": md.get("source_type") or md.get("type") or "unknown",
            "record_type": md.get("type"),
            "chunk_index": md.get("chunk_index"),
            "document_id": md.get("document_id"),
            "crop": md.get("crop"),
            "faq_id": md.get("id") if md.get("type") == "faq" else None,
        }

    def search_relevant_context(
        self,
        query: str,
        k: int = 5,
        return_sources: bool = False,
    ) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
        """
        Retrieve relevant documents using Chroma when available, otherwise fall back to
        keyword-based matching over cached documents.

        Args:
            query: user query text
            k: number of documents to retrieve
            return_sources: if True, returns a (context_str, sources_list) tuple instead
                of just the context string. `sources_list` is a de-duplicated list of
                {"label": ..., "source_type": ...} dicts describing where each piece of
                context came from (e.g. brand KB, an uploaded document, or a website).

        Backward compatible: existing callers that only expect a string keep working
        as long as return_sources is left False (the default).
        """
        if not self.in_memory_docs:
            self.prepare_in_memory_docs()

        if self.vectorstore is not None:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': k, 'fetch_k': 25, 'lambda_mult': 0.25}
            )
            docs = []
            for method_name in ("get_relevant_documents", "_get_relevant_documents"):
                method = getattr(retriever, method_name, None)
                if callable(method):
                    try:
                        docs = method(query)
                        break
                    except TypeError:
                        continue
            if not docs:
                docs = self._keyword_ranked_docs(query, k=k)
            # Blend in lexical matches so exact terms from short FAQs and uploaded
            # material remain easy to find with the lightweight offline fallback.
            docs.extend(self._keyword_ranked_docs(query, k=k))
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
        sources: List[Dict[str, Any]] = []
        seen_source_records = set()

        for i, d in enumerate(unique_docs, 1):
            src_label = self._document_source_label(d)

            # Truncate long page content to keep prompt manageable
            snippet = (d.page_content[:1200] + '...') if len(d.page_content) > 1200 else d.page_content
            ctx_parts.append(f"[Source: {src_label}] Relevant Information {i}:\n{snippet}")

            source_info = self._document_source_info(d)
            source_key = (
                source_info.get("label"), source_info.get("record_type"),
                source_info.get("document_id"), source_info.get("chunk_index"),
                source_info.get("crop"), source_info.get("faq_id"),
            )
            if source_key not in seen_source_records:
                seen_source_records.add(source_key)
                sources.append(source_info)

        context_text = "\n\n".join(ctx_parts)

        if return_sources:
            return context_text, sources
        return context_text

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
