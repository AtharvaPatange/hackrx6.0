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
        
        response = requests.post(url, headers=headers, json=data, timeout=15)  # Reduced timeout
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
    Optimized for accuracy and detailed responses matching expected format.
    """
    system_prompt = """You are an expert insurance policy analyst. Your task is to provide comprehensive, accurate answers based ONLY on the provided document context.

Instructions:
- Read the context thoroughly and extract ALL relevant information for each question
- Provide complete details including specific numbers, durations, percentages, conditions, and requirements
- Include eligibility criteria, waiting periods, limits, exceptions, and sub-conditions
- Use precise language and terminology from the document
- Provide detailed explanations, not just brief answers
- If the answer cannot be found in the context, state: "The specific information is not found in the provided document"
- Be thorough and comprehensive - include all relevant policy terms

Example detailed format:
- For grace period: Include the exact number of days and what it applies to
- For waiting periods: Include exact duration and what conditions apply
- For coverage: Include what's covered, eligibility requirements, and any limitations
- For definitions: Provide complete definition with all criteria mentioned"""
    
    user_prompt = f"""Based on the following insurance policy document context, provide a comprehensive and accurate answer with all specific details:

Context:
{context}

Question: {question}

Provide a detailed answer with all specific numbers, conditions, requirements, and limitations mentioned in the context:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL,
            temperature=0.0,  # Set to 0 for maximum accuracy and consistency
            max_tokens=800,   # Significantly increased for detailed responses
            timeout=15,       # Increased timeout for thorough analysis
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        raise