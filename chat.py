# RAG Chatbot Demo App

import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = os.getenv("BASE_API_URL")
LANGFLOW_ID = os.getenv("LANGFLOW_ID")
FLOW_ID = os.getenv("FLOW_ID")
APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "ChatInput-NDeRx": {},
  "ChatOutput-c8nFg": {},
  "ParseData-zgD8I": {},
  "OpenAIModel-nq1Mg": {},
  "File-0dOQn": {},
  "Prompt-CQmRB": {}
}


# Initialize FastAPI app
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    endpoint: Optional[str] = ENDPOINT or FLOW_ID
    tweaks: Optional[dict] = TWEAKS
    output_type: str = "chat"
    input_type: str = "chat"
    application_token: Optional[str] = APPLICATION_TOKEN


@app.post("/chat")
async def chat_with_bot(request: ChatRequest) -> dict:
    """
    API Endpoint to send a message to the chatbot and return the response.
    """
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{request.endpoint}"

    payload = {
        "input_value": request.message,
        "output_type": request.output_type,
        "input_type": request.input_type,
    }

    headers = {"Content-Type": "application/json"}
    if request.tweaks:
        payload["tweaks"] = request.tweaks
    if request.application_token:
        headers["Authorization"] = f"Bearer {request.application_token}"

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTP error if request fails
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to chatbot API: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server on http://127.0.0.1:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)