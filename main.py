# File: main.py

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas import RunRequest, RunResponse
from config import API_AUTH_TOKEN
from vector_store import query_pinecone, process_and_store_documents
from llm_services import get_answer_from_llm
import asyncio
import time
import concurrent.futures

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

def process_single_question(question: str, question_num: int, total_questions: int):
    """Process a single question with context retrieval and LLM answer generation."""
    print(f"Processing question {question_num}/{total_questions}: {question[:50]}...")
    question_start = time.time()
    
    # a. Retrieve relevant context from Pinecone (reduced chunks)
    context = query_pinecone(question, top_k=3)  # Reduced from 5 to 3 for speed
    
    # b. Generate answer using LLM with the context
    answer = get_answer_from_llm(question, context)
    
    question_time = time.time() - question_start
    print(f"Question {question_num} completed in {question_time:.2f} seconds")
    
    return answer

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
        
        # More generous timeout for processing all questions
        elapsed = time.time() - start_time
        if elapsed > 120:  # Increased timeout to focus on accuracy
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Processing timeout - document processing took too long"
            )

        # 2. Process questions in parallel for speed
        print("Step 2: Generating answers...")
        questions_to_process = request.questions
        print(f"Processing all {len(questions_to_process)} questions")
        
        # Use ThreadPoolExecutor for parallel processing with reduced workers for speed
        all_answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
            # Submit all questions for parallel processing
            future_to_question = {
                executor.submit(process_single_question, question, i+1, len(questions_to_process)): i 
                for i, question in enumerate(questions_to_process)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_question):
                question_index = future_to_question[future]
                try:
                    answer = future.result()
                    all_answers.append((question_index, answer))
                except Exception as e:
                    print(f"Question {question_index + 1} failed: {e}")
                    all_answers.append((question_index, f"Error processing question: {e}"))
                
                # Check timeout - focus on accuracy over speed
                elapsed = time.time() - start_time
                if elapsed > 180:  # Increased timeout for accuracy
                    print(f"Timeout reached after processing {len(all_answers)} questions")
                    # Add placeholder answers for remaining questions
                    remaining = len(questions_to_process) - len(all_answers)
                    for i in range(remaining):
                        all_answers.append((len(all_answers), "Processing timeout - unable to complete all questions"))
                    break
        
        # Sort answers by original question order
        all_answers.sort(key=lambda x: x[0])
        final_answers = [answer for _, answer in all_answers]
        
        # All questions processed
        print(f"Successfully processed {len(final_answers)} questions")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return RunResponse(answers=final_answers)

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