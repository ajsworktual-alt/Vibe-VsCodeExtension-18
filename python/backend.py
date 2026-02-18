import os
import json
import sys
import io
import subprocess
import time
import fnmatch
import re
import ast
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()

app = FastAPI()

# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not configured")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Initialize Gemini client
try:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Warning: Gemini client initialization failed: {e}")
    client = None

# System prompt for AI assistant
SYSTEM_PROMPT = """
You are an advanced VS Code Extension AI Assistant.
Always write advanced-level code.
If a project requires more than 1,000 lines of code, provide only up to 1,000 lines.
You must provide code for all required languages used in the project.
If the user requests a project that requires multiple files and folders, create all necessary files and folders with essential code included.

You operate in STRICT ACTION MODE.

GENERAL RULES:
- Always return ONLY valid JSON when performing actions.
- Never use markdown.
- Never use backticks.
- Never include explanations.
- Never include comments.
- Never include triple quotes.
- Never include actual newlines inside JSON strings. Use \\n for all line breaks.
- All JSON must be syntactically valid.

GREETING RULE:
If the user says: hi, hello, hey
Return EXACTLY: {"action": "greeting", "message": "Hello, Good to see you.!"}

AVAILABLE ACTIONS:
{
  "action": "create_folder",
  "folder": "<folder_name>"
}

{
  "action": "create_file",
  "path": "<relative_path/file.py>",
  "content": "<full file content with \\n>"
}

{
  "action": "create_project",
  "folder": "<project_name>",
  "files": [{"path": "<relative_path/file1.py>", "content": "<content>"}]
}

{
  "action": "update_file",
  "path": "<relative_path/file.py>",
  "content": "<full corrected file content with \\n>"
}

{
  "action": "debug_file",
  "path": "<relative_path/file.py>"
}

{
  "action": "run_file",
  "path": "<relative_path/file.py>",
  "environment": "none"
}

{
  "action": "search_files",
  "keyword": "",
  "file_type": ".py",
  "max_results": 10
}

{
  "action": "search_in_files",
  "keyword": "",
  "file_pattern": "*.py",
  "max_results": 10
}

{
  "action": "get_file_info",
  "path": "<relative_path/file.py>"
}

IMPORTANT:
- Return exactly one valid JSON object per response.
- Never mix raw code and JSON.
- Never wrap JSON in markdown.
"""

class MessageRequest(BaseModel):
    text: str
    workspace_path: str = ""
    conversation_history: str = ""
    files: list = []

class ActionResponse(BaseModel):
    action: str
    message: str = ""
    path: str = ""
    content: str = ""
    folder: str = ""
    files: list = []

@app.get("/")
async def home():
    return {"status": "VibeCoding Backend is running", "version": "2.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat")
async def chat(request: MessageRequest):
    """Process user message and return AI response with actions"""
    try:
        if not client:
            return {
                "error": "AI service not available",
                "action": "error",
                "message": "Gemini API is not configured"
            }
        
        # Build full prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n"
        if request.conversation_history:
            full_prompt += f"Conversation history:\n{request.conversation_history}\n\n"
        full_prompt += f"User: {request.text}\nAssistant:"
        
        # Call Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        
        assistant_reply = response.text.strip()
        
        return {
            "success": True,
            "response": assistant_reply,
            "action": "response"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": "error"
        }

@app.post("/generate")
async def generate_code(request: MessageRequest):
    """Generate code based on user request"""
    try:
        if not client:
            return {
                "success": False,
                "error": "AI service not available",
                "action": "error"
            }
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser request: {request.text}\n\nProvide complete code with file structure."
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        
        return {
            "success": True,
            "response": response.text,
            "action": "generate"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": "error"
        }

@app.post("/debug")
async def debug_file(request: MessageRequest):
    """Analyze and debug code"""
    try:
        if not client:
            return {
                "success": False,
                "error": "AI service not available",
                "action": "error"
            }
        
        full_prompt = f"""
{SYSTEM_PROMPT}

The following code has errors. Please analyze and provide fixed code:

{request.text}

Return ONLY the corrected code in JSON format:
{{
  "action": "update_file",
  "path": "filename.py",
  "content": "fixed code here"
}}
"""
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )
        
        return {
            "success": True,
            "response": response.text,
            "action": "debug"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": "error"
        }

@app.post("/search")
async def search_files(request: MessageRequest):
    """Search files in workspace (returns search parameters for VS Code to execute)"""
    try:
        keyword = request.text
        file_type = ".py"
        
        return {
            "success": True,
            "action": "search_files",
            "keyword": keyword,
            "file_type": file_type,
            "max_results": 10
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "action": "error"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)