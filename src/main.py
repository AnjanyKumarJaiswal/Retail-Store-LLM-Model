from fastapi import FastAPI
import uvicorn
from models.trained_gpt import GenerateResponses
from config.hf_config import HuggingFaceInitialization

app = FastAPI()


@app.get('/')
async def test_run():
    return {"message":"The Server is working"}

@app.get('/chat')
async def llm_chat(query: str):
    HuggingFaceInitialization()
    trained_model = './utils/checkpoint-850'
    model_name = 'gpt2-medium'
    model_generated_response = GenerateResponses.model_responses(
        query=query,
        trained_model=trained_model,
        model_name=model_name
    )
    return {"message":f"Model Response: ${model_generated_response}"}


if __name__ == "__main__":
    uvicorn.run(app=app , host='localhost',port=8000)
    