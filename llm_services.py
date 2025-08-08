from groq import Groq
import requests
from config import GROQ_API_KEY, JINA_API_KEY, LLM_MODEL, EMBEDDING_MODEL

# --- Initialize Groq client ---

# Initialize Groq client for LLM
groq_client = Groq(api_key=GROQ_API_KEY)

def get_embedding(text: str):
    """Generates an embedding for a given text chunk using Jina AI."""
    try:
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        data = {
            'input': [text],
            'model': EMBEDDING_MODEL
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['data'][0]['embedding']
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def get_answer_from_llm(question: str, context: str):
    """
    Uses Groq's LLaMA model to generate an answer based on a question and retrieved context.
    """
    system_prompt = """
    You are an expert insurance policy analyst and question-answering system. Your task is to provide detailed, accurate answers based ONLY on the provided document context.

    Instructions:
    1. Read the context carefully and extract the most relevant information for each question
    2. Provide comprehensive answers with specific details, numbers, and conditions mentioned in the document
    3. Include waiting periods, percentages, limits, and specific requirements when mentioned
    4. If multiple conditions apply, list them clearly
    5. Use precise language from the document
    6. If the answer cannot be found in the provided context, state: "The answer could not be found in the provided document."
    7. Do not use external knowledge - only use information from the provided context
    8. Be thorough but concise, focusing on the key details that directly answer the question
    """
    
    user_prompt = f"""Based on the following document context, please answer the question with comprehensive details:

Context:
---
{context}
---

Question: {question}

Please provide a detailed answer based solely on the information in the context above."""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL,
            temperature=0.1,  # Slightly increased for more detailed responses
            max_tokens=800,   # Reduced for faster responses
            timeout=10,       # 10 second timeout per question
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        raise