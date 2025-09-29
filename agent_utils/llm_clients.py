import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def get_gemini_client(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """Return a configured Gemini client for LangChain usage."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key
    )