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
    You are an expert Question-Answering system. Your task is to answer the user's question based *only* on the provided text context.
    Do not use any external knowledge.
    If the context does not contain the answer, state that "The answer could not be found in the provided document."
    Your answer must be concise and directly address the question.
    """
    
    user_prompt = f"Context:\n---\n{context}\n---\nQuestion: {question}"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=1000,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        raise