# File: main.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas import RunRequest, RunResponse
from config import API_AUTH_TOKEN
from document_processor import process_document_from_url
from vector_store import query_pinecone
from llm_services import get_answer_from_llm

# --- App Initialization ---
app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="API for processing documents and answering questions using LLMs.",
    version="1.0.0"
)

# --- Authentication ---
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to verify the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(request: RunRequest, authorized: bool = Depends(verify_token)):
    """
    Main endpoint to process a document and answer questions.
    """
    try:
        # 1. Process the document from the URL
        # This will download, parse, chunk, embed, and store the document in Pinecone
        process_document_from_url(request.documents)

        # 2. Loop through questions and generate answers
        all_answers = []
        for question in request.questions:
            print(f"\nProcessing question: {question}")
            
            # a. Retrieve relevant context from Pinecone
            context = query_pinecone(question)
            
            # b. Generate answer using LLM with the context
            answer = get_answer_from_llm(question, context)
            all_answers.append(answer)
            print(f"Generated Answer: {answer}")
        
        return RunResponse(answers=all_answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )

# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}