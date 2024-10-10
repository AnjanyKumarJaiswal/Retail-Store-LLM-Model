# Retail-Store-LLM-Model

This project focuses on fine-tuning an OpenAI GPT model using the OpenAI API for a specific domain, such as customer service in a retail store. The goal is to create a model capable of generating relevant and accurate responses to customers.

## Problem Statement
Fine-tune an OpenAI GPT model for a specific domain (e.g., customer service). The objective is to train a custom response model that can effectively handle queries from that domain.

## Prerequisites
- Python 3.7 or higher
- OpenAI API Key
- Virtual environment (recommended)

## Setup Instructions

### 1. Create a Virtual Environment
First, set up a virtual environment to manage dependencies:
```bash
python3 -m venv app
app\Scripts\activate

```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Fine Tune Model File
```bash
cd src/models
python fine_tuned_gpt.py
```
### 4. Run the Backend
you can chat with the fine-tuned model at the /chat endpoint.
```bash
cd ..
uvicorn main:app --reload
```

## Conclusion
This project helps in building a fine-tuned GPT model for customer service in a retail store environment. The model can generate customized responses, improving customer interactions. The backend is built using FastAPI, providing a user-friendly interface for querying the model.


