import os
import uuid
import time
import json
import asyncio
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# FastAPI app
app = FastAPI(title="Gemini Backend")

# CORS - En liberal ayarlar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0"
]

current_key_index = 0

# Models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

def get_next_key():
    """Get next API key"""
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key

def convert_content(content):
    """Convert OpenAI content to Gemini"""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                if item.image_url.url.startswith("data:"):
                    header, base64_data = item.image_url.url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            except:
                parts.append({"text": "[Image error]"})
    
    return parts or [{"text": ""}]

def convert_messages(messages):
    """Convert messages to Gemini format"""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": convert_content(msg.content)
        }
        for msg in messages
    ]

async def stream_response(text: str, model: str):
    """Stream in OpenAI format"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    
    # Content chunks
    if text:
        for i in range(0, len(text), 30):
            chunk = text[i:i+30]
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0.01)
    
    # Final chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

@app.options("/{path:path}")
async def options_handler():
    """Handle preflight"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "keys": len(API_KEYS),
        "version": "1.0"
    }

@app.get("/v1/models")
async def models():
    """List models"""
    return {
        "object": "list",
        "data": [
            {"id": "gemini-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": "gemini-pro-vision", "object": "model", "created": int(time.time()), "owned_by": "google"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint"""
    try:
        # Get API key
        api_key = get_next_key()
        
        # Convert messages
        gemini_messages = convert_messages(request.messages)
        
        # Generation config
        config = {"temperature": request.temperature}
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        
        # Call Gemini API
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent",
                json={
                    "contents": gemini_messages,
                    "generationConfig": config
                },
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            )
            
            if response.status_code != 200:
                return {
                    "error": {
                        "message": f"Gemini API error: {response.status_code}",
                        "type": "api_error"
                    }
                }
            
            result = response.json()
            
            # Extract text
            text = ""
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
            
            if not text:
                text = "Üzgünüm, yanıt oluşturamadım."
            
            # Return response
            if request.stream:
                return StreamingResponse(
                    stream_response(text, request.model),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Gemini API Backend", "status": "running"}