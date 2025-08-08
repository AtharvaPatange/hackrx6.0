# File: main.py

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas import RunRequest, RunResponse
from config import API_AUTH_TOKEN
from vector_store import query_pinecone, process_and_store_documents
from llm_services import get_answer_from_llm
import asyncio
import time

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
    Optimized for sub-30-second response times.
    """
    start_time = time.time()
    try:
        print(f"Starting processing at {time.strftime('%H:%M:%S')}")
        
        # 1. Process the document from the URL (with aggressive timeout)
        print("Step 1: Processing document...")
        process_start = time.time()
        
        # Use our optimized function with caching
        process_and_store_documents(request.documents)
        
        process_time = time.time() - process_start
        print(f"Document processing completed in {process_time:.2f} seconds")
        
        # Aggressive timeout check - if document processing took too long, abort
        elapsed = time.time() - start_time
        if elapsed > 22:  # Leave 8 seconds for questions
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Processing timeout - document processing took too long"
            )

        # 2. Loop through questions and generate answers (with aggressive timeout)
        print("Step 2: Generating answers...")
        all_answers = []
        
        # Limit to maximum 3 questions for speed (hackathon optimization)
        questions_to_process = request.questions[:3]
        if len(request.questions) > 3:
            print(f"Warning: Processing only first 3 of {len(request.questions)} questions for speed")
        
        for i, question in enumerate(questions_to_process):
            # Check timeout for each question - be very aggressive
            elapsed = time.time() - start_time
            if elapsed > 27:  # Only 3 seconds left total
                print(f"Timeout reached after processing {i} questions")
                # Return partial results for remaining questions
                remaining = len(questions_to_process) - i
                all_answers.extend(["Processing timeout - unable to complete all questions"] * remaining)
                break
                
            print(f"Processing question {i+1}/{len(questions_to_process)}: {question[:50]}...")
            question_start = time.time()
            
            # a. Retrieve relevant context from Pinecone (reduced chunks)
            context = query_pinecone(question, top_k=3)  # Reduced from 5 to 3 for speed
            
            # b. Generate answer using LLM with the context
            answer = get_answer_from_llm(question, context)
            all_answers.append(answer)
            
            question_time = time.time() - question_start
            print(f"Question {i+1} completed in {question_time:.2f} seconds")
        
        # Add empty answers for any skipped questions beyond 3
        if len(request.questions) > 3:
            remaining_count = len(request.questions) - len(all_answers)
            all_answers.extend(["Question limit exceeded - processing only first 3 questions for speed"] * remaining_count)
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return RunResponse(answers=all_answers)

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error occurred after {total_time:.2f} seconds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )

# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}