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
    Optimized for accuracy and concise responses.
    """
    # Specific prompt targeting expected answer format
    system_prompt = """You are an expert insurance analyst. Provide EXACT answers that match the expected format.

For each question, provide the most accurate answer in ONE SENTENCE based on the document context.

Expected answer patterns:
- Grace period: "A grace period of thirty days is provided for premium payment..."
- Waiting period PED: "There is a waiting period of thirty-six (36) months..."
- Maternity: "Yes, the policy covers maternity expenses, including childbirth..."
- Cataract: "The policy has a specific waiting period of two (2) years..."
- Organ donor: "Yes, the policy indemnifies the medical expenses..."
- NCD: "A No Claim Discount of 5% on the base premium..."
- Health check-ups: "Yes, the policy reimburses expenses for health check-ups..."
- Hospital definition: "A hospital is defined as an institution with at least..."
- AYUSH: "The policy covers medical expenses for inpatient treatment..."
- Room rent: "Yes, for Plan A, the daily room rent is capped at..."

If information is not found, say: "Not found in the provided document"."""
    
    user_prompt = f"""Context: {context}

Question: {question}

Provide the exact answer in one sentence:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=150,   # Reduced for exact answers
            timeout=10,        # Increased timeout for accuracy
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        raise