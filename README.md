# HackRx Intelligent Query-Retrieval System

A FastAPI-based intelligent document processing and question-answering system that uses modern AI services.

## Features

- 📄 **PDF Document Processing**: Download and extract text from PDF URLs
- 🤖 **AI-Powered Q&A**: Uses Groq's LLaMA model for intelligent answers
- 🔍 **Semantic Search**: Jina AI embeddings for accurate document retrieval
- 📊 **Vector Storage**: Pinecone for efficient similarity search
- 🚀 **Fast API**: High-performance REST API with FastAPI

## Tech Stack

- **LLM**: Groq (LLaMA-3.1-8b-instant)
- **Embeddings**: Jina AI (jina-embeddings-v2-base-en)
- **Vector DB**: Pinecone
- **Web Framework**: FastAPI
- **Deployment**: Render

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### POST `/api/v1/hackrx/run`
Process a document and answer questions about it.

**Headers:**
```
Authorization: Bearer YOUR_API_AUTH_TOKEN
Content-Type: application/json
```

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The main topic is...",
    "The key findings are..."
  ]
}
```

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key
- `JINA_API_KEY`: Your Jina AI API key  
- `PINECONE_API_KEY`: Your Pinecone API key
- `API_AUTH_TOKEN`: Bearer token for API authentication

## Deployment

This app is configured for easy deployment on Render. The `render.yaml` file contains all necessary configuration.

## License

MIT License
