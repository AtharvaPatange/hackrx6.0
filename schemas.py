# File: schemas.py

from pydantic import BaseModel, Field
from typing import List

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions.")