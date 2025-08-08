from groq import Groq
import requests
from config import GROQ_API_KEY, JINA_API_KEY, LLM_MODEL, EMBEDDING_MODEL

# --- Initialize Groq client ---

# Initialize Groq client for LLM
groq_client = Groq(api_key=GROQ_API_KEY)

def get_embeddings_from_jina(texts: list):
    """Generates embeddings for multiple texts at once using Jina AI."""
    try:
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        data = {
            'input': texts,
            'model': EMBEDDING_MODEL
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return [item['embedding'] for item in result['data']]
        
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise

def get_embedding(text: str):
    """Generates an embedding for a single text using Jina AI."""
    return get_embeddings_from_jina([text])[0]

def get_answer_from_llm(question: str, context: str):
    """
    Uses Groq's LLaMA model to generate an answer based on a question and retrieved context.
    Optimized for speed while maintaining quality.
    """
    system_prompt = """You are an expert insurance analyst. Provide accurate, detailed answers based ONLY on the document context provided.

Instructions:
- Extract relevant information directly from the context
- Include specific numbers, percentages, conditions, and requirements
- If info isn't in the context, state: "Not found in the provided document"
- Be thorough but concise
- Focus on key details that answer the question"""
    
    user_prompt = f"""Context:
{context}

Question: {question}

Answer based on the context:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=600,   # Reduced for speed
            timeout=8,        # Reduced timeout for faster responses
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        raise