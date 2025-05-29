from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import json
import uuid
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test API key
API_KEY = "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4"

@app.get("/")
async def root():
    return {"message": "OK", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": int(time.time())}

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-pro", "object": "model", "created": int(time.time()), "owned_by": "google"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        # Get request data
        data = await request.json()
        
        # Extract message
        messages = data.get("messages", [])
        if not messages:
            return JSONResponse({"error": {"message": "No messages"}}, status_code=400)
        
        last_message = messages[-1]
        user_content = last_message.get("content", "")
        
        # Simple Gemini API call
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                json={
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": user_content}]
                    }]
                },
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": API_KEY
                }
            )
            
            if response.status_code != 200:
                return JSONResponse({
                    "error": {"message": f"API Error: {response.status_code}"}
                }, status_code=500)
            
            result = response.json()
            
            # Extract response
            text = "No response"
            if result.get("candidates"):
                candidate = result["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    text = candidate["content"]["parts"][0].get("text", "No text")
            
            # Return OpenAI format
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gemini-pro",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
            
    except Exception as e:
        return JSONResponse({
            "error": {"message": str(e), "type": "internal_error"}
        }, status_code=500)

# Vercel handler
from mangum import Mangum
handler = Mangum(app)