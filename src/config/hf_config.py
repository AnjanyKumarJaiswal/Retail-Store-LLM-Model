from huggingface_hub import login
from dotenv import load_dotenv
import os

def HuggingFaceInitialization(api_key=None):
    load_dotenv()
    api_key = os.getenv('HF_KEY')
    return login(api_key)