# LCB Chatbot Backend with Ollama and RAG

This is the backend API service for the AI Profile Assistant, integrated with a true RAG (Retrieval-Augmented Generation) system.

## Features

- Flask-based RESTful API
- Local Ollama integration (default: `llama3.2:3b`, no API key)
- Automatic LLM-as-a-judge evaluation for every chatbot response
- Supervisor routing from general chat to the appropriate specialist agent
- Provider-independent LLM client for any OpenAI-compatible API
- **True RAG System**: Uses ChromaDB vector database
- Document chunking and vectorization
- Semantic similarity search
- CORS support for cross-origin requests
- Loads personal data from JSON file
- Health check endpoint
- Local deployment, no external hosting required

## RAG System Architecture

```
User Query → Vectorization → Similarity Search → Context Enhancement → AI Response Generation
    ↓
Personal Data → Document Chunking → Vector Storage → ChromaDB
```

### RAG Workflow:

1. **Document Processing**: Convert `personal_data.json` to structured documents
2. **Text Chunking**: Use recursive character splitter to divide documents into chunks
3. **Vectorization**: Use local `all-MiniLM-L6-v2` sentence-transformer embeddings when its model is cached; use the offline deterministic fallback only when semantic embeddings are unavailable
4. **Storage**: Store document chunks, metadata, and vectors in ChromaDB's persistent SQLite file: `chroma_db/chroma.sqlite3`
5. **Retrieval**: Search for most relevant document chunks based on user query
6. **Generation**: Pass retrieved context as prompt to AI for response generation

## Local Development

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy the environment variable template:

```bash
cp env.example .env
```

The default local configuration needs no API key:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

### 3. Run Server

```bash
python app.py
```

Install the model once, then start the server:

```bash
ollama pull llama3.2:3b
python app.py
```

The server starts at `http://localhost:5001` and automatically builds the vector database.

For higher-quality semantic retrieval, run the server once with internet access to cache the embedding model:

```bash
EMBEDDING_ALLOW_DOWNLOAD=true python app.py
```

After that download completes, set `EMBEDDING_ALLOW_DOWNLOAD=false` again for fully offline starts.

Large document libraries are indexed in batches (default: 64 chunks); adjust
`CHROMA_INGEST_BATCH_SIZE` downward if the machine has limited RAM.

To switch to OpenAI, DeepSeek, Groq, vLLM, or another OpenAI-compatible API, set:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key
```

## API Endpoints

### POST /api/chat

Send a message to the AI assistant (uses RAG retrieval).

**Request Body:**
```json
{
  "message": "What is your background?"
}
```

**Response:**
```json
{
  "response": "I am Your Name, a Software Engineer...",
  "success": true,
  "context_used": "Relevant Information 1:\nName: Your Name\nTitle: Software Engineer..."
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "AI Assistant API is running",
  "api_key": "configured",
  "rag_system": "initialized"
}
```

### POST /api/rebuild-vectorstore

Rebuild the vector database (use after updating personal data).

**Response:**
```json
{
  "message": "Vector database rebuilt successfully",
  "success": true
}
```

## Personal Data Configuration

Edit the `personal_data.json` file to customize your personal information. The file structure includes:

- `basic`: Basic information (name, title, email, etc.)
- `skills`: Skills list
- `experience`: Work experience
- `projects`: Project experience
- `education`: Education background
- `certifications`: Certifications
- `interests`: Interests and hobbies
- `careerGoals`: Career objectives

**After updating personal data, call the `/api/rebuild-vectorstore` endpoint to rebuild the vector database.**

## RAG System Advantages

### Compared to Traditional Methods:

1. **Precise Retrieval**: Only retrieves information relevant to the question
2. **Reduced Hallucination**: Based on retrieved real information
3. **Scalability**: Supports large documents and complex queries
4. **Efficiency**: Avoids sending all information to AI
5. **Cost Optimization**: Reduces token usage

### Technical Features:

- **Document Chunking**: Intelligent splitting while maintaining semantic integrity
- **Vector Search**: Based on semantic similarity, not keyword matching
- **Context Enhancement**: Dynamically builds most relevant context
- **Metadata Management**: Adds type labels to each document chunk

## Environment Variables

- `LLM_PROVIDER`: `ollama` (default) or `openai_compatible`
- `OLLAMA_BASE_URL`: Ollama server URL (default `http://localhost:11434`)
- `OLLAMA_MODEL`: Ollama model (default `llama3.2:3b`)
- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`: OpenAI-compatible provider settings
- `LLM_JUDGE_ENABLED`: evaluate every main, marketing, and tracker chatbot answer (default: `true`)

Each chatbot response includes a `judgment` object with an overall score,
`pass`/`warning`/`fail` verdict, dimension scores, issues, and improvement feedback.
Judging uses a second call to the configured LLM. If that call is unavailable,
the original chatbot response is still returned with `status: "unavailable"`.

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── rag_system.py          # RAG system core module
├── personal_data.json     # Personal data
├── requirements.txt       # Python dependencies
├── env.example           # Environment variable template
├── .env                  # Environment variables (local)
├── chroma_db/            # Vector database storage (auto-generated)
└── README.md            # Documentation
```

## Notes

1. Run `ollama pull llama3.2:3b` before starting the default configuration.
2. Uploaded documents are chunked and stored in the Chroma SQLite vector index, and in the application document table for listing and deletion.
3. The chat endpoint always retrieves seeded FAQ/brand data and uploaded document chunks, then adds recent conversation history for continuity.
4. The vector database is stored in `chroma_db/chroma.sqlite3`.
