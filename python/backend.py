import os
import requests
import json
import ast
import sys
import io
import subprocess
import threading
import time
import fnmatch
import re
from pathlib import Path
from datetime import datetime
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from contextlib import redirect_stdout
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Server mode: when True, no local file writes are performed.
# All file operations are returned as messages to the extension.
# ------------------------------------------------------------
SERVER_MODE = True

# ------------------------------------------------------------
# Gemini API setup
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not configured")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3-pro-preview"  # or whatever model you use

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="VibeCoding AI Backend")

# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    pending_action: Optional[dict] = None
    files: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    messages: List[dict]

# ------------------------------------------------------------
# System prompt (unchanged)
# ------------------------------------------------------------
SYSTEM_PROMPT = """
You are an advanced VS Code Extension AI Assistant.

- Always write advanced-level code.
- If a project requires more than 1,000 lines of code, provide only up to 1,000 lines.
- If a project requires more than 10,000 lines of code, still provide only up to 1,000 lines.
- You must provide code for all required languages used in the project.
- If the user requests a project that requires multiple files and folders, create all necessary files and folders with essential code included. Do not create empty files or folders.

You operate in STRICT ACTION MODE.

GENERAL RULES:
- Always return ONLY valid JSON when performing actions.
- Never use markdown.
- Never use backticks.
- Never include explanations.
- Never include comments.
- Never include triple quotes.
- Never include actual newlines inside JSON strings.
- Use \\n for all line breaks.
- All JSON must be syntactically valid.
- Be precise and deterministic.

GREETING RULE:
If the user says: hi, hello, hey
Return EXACTLY:
Hello, Good to see you.!

----------------------------------------
AVAILABLE ACTIONS
----------------------------------------

CREATE FOLDER:
{
  "action": "create_folder",
  "folder": "<folder_name>"
}

CREATE FILE (fails if exists):
{
  "action": "create_file",
  "path": "<relative_path/file.py>",
  "content": "<full file content with \\n>"
}

CREATE PROJECT (multiple files):
{
  "action": "create_project",
  "folder": "<project_name>",
  "files": [
    {
      "path": "<relative_path/file1.py>",
      "content": "<full file content with \\n>"
    }
  ]
}

UPDATE FILE (overwrite entire file):
{
  "action": "update_file",
  "path": "<relative_path/file.py>",
  "content": "<full corrected file content with \\n>"
}

DEBUG FILE (auto-fix mode):
{
  "action": "update_file",
  "path": "<relative_path/file.py>",
  "content": "<fully corrected file content with \\n>"
}

RUN FILE:
{
  "action": "run_file",
  "path": "<relative_path/file.py>",
  "environment": "none"
}

SEARCH FILES:
{
  "action": "search_files",
  "keyword": "<term>",
  "file_type": ".py",
  "max_results": 10
}

SEARCH FOLDERS:
{
  "action": "search_folders",
  "keyword": "<term>",
  "max_results": 10
}

SEARCH INSIDE FILES:
{
  "action": "search_in_files",
  "keyword": "<term>",
  "file_pattern": "*.py",
  "max_results": 10
}

GET FILE INFO:
{
  "action": "get_file_info",
  "path": "<relative_path/file.py>"
}
OPERATION MODE RULES:

1. If performing file system actions (create, update, delete, run, search):
   → Return valid JSON only.

2. If user asks for explanation or example code:
   → Return normal formatted code (no JSON).

3. If debugging file:
   → Return update_file JSON with corrected full content.

4. Never mix raw code and JSON.
5. Never wrap JSON in markdown.
----------------------------------------
AUTO-HEALING RULES (When Debugging)
----------------------------------------

When fixing a file:

- Fix the ENTIRE file.
- Ensure 100% valid Python syntax.
- Fix incorrect imports.
- Fix indentation.
- Fix logical errors if obvious.
- Preserve working functionality.
- Add minimal safe error handling if missing.
- Return the FULL corrected file.
- Use \\n for line breaks.
- Never explain changes.
- Never return partial code.
- Never return debug analysis.
- Return ONLY update_file action.

----------------------------------------
IMPORTANT
----------------------------------------

After create_file or create_project,
ALWAYS suggest running the main file using run_file action in a separate JSON response.

Never combine two actions in one JSON object.
Return exactly one valid JSON object per response.
"""

# ------------------------------------------------------------
# Utility functions (unchanged, but used only for validation etc.)
# ------------------------------------------------------------
def format_file_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def validate_python_code(code, filename):
    try:
        ast.parse(code)
        return None, None
    except SyntaxError as e:
        lines = code.split('\n')
        error_line = lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else ""
        pointer = " " * (e.offset - 1) + "^" if e.offset else ""
        suggestion = get_syntax_error_suggestion(e.msg, error_line)
        error_msg = (
            f"SyntaxError: {e.msg}\n"
            f"  File: {filename}\n"
            f"  Line: {e.lineno}\n"
            f"  Column: {e.offset}\n"
            f"  Code: {error_line.strip()}\n"
            f"        {pointer}"
        )
        if suggestion:
            error_msg += f"\n  Suggestion: {suggestion}"
        return error_msg, e.lineno
    except Exception as e:
        return f"Error: {str(e)}", None

def get_syntax_error_suggestion(error_msg, error_line):
    suggestions = {
        'invalid syntax': "Check for missing colons (:), brackets, or quotes",
        'unexpected EOF': "Check for unclosed brackets, quotes, or parentheses",
        'EOL while scanning string literal': "Check for unclosed quotes in strings",
        'unexpected indent': "Check indentation - Python uses consistent indentation",
        'unindent does not match': "Check that indentation levels match",
        'Missing parentheses': "Add missing parentheses ()",
        'invalid character': "Remove or replace invalid characters",
    }
    for key, suggestion in suggestions.items():
        if key.lower() in error_msg.lower():
            return suggestion
    if ':' not in error_line and any(keyword in error_line for keyword in ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with', 'elif', 'else']):
        return "Missing colon (:) at the end of the statement"
    if '(' in error_line and ')' not in error_line:
        return "Missing closing parenthesis )"
    if '[' in error_line and ']' not in error_line:
        return "Missing closing bracket ]"
    if '{' in error_line and '}' not in error_line:
        return "Missing closing brace }"
    return None

def analyze_error(error_msg, code, filename):
    # (same as original, returns dict)
    analysis = {
        'error_type': None,
        'error_message': error_msg,
        'line_number': None,
        'suggestions': [],
        'common_causes': [],
        'fix_examples': []
    }
    # Determine error type
    if 'SyntaxError' in error_msg:
        analysis['error_type'] = 'Syntax Error'
        analysis['common_causes'] = [
            'Missing colons (:) after control statements',
            'Unclosed brackets, parentheses, or quotes',
            'Incorrect indentation',
            'Invalid characters or typos'
        ]
    elif 'IndentationError' in error_msg:
        analysis['error_type'] = 'Indentation Error'
        analysis['common_causes'] = [
            'Mixed tabs and spaces',
            'Incorrect indentation level',
            'Missing indentation in block'
        ]
    elif 'NameError' in error_msg:
        analysis['error_type'] = 'Name Error'
        analysis['common_causes'] = [
            'Variable not defined',
            'Typo in variable name',
            'Variable defined in different scope',
            'Missing import statement'
        ]
    elif 'TypeError' in error_msg:
        analysis['error_type'] = 'Type Error'
        analysis['common_causes'] = [
            'Operating on incompatible types',
            'Wrong number of arguments',
            'NoneType operations',
            'String/number concatenation'
        ]
    elif 'IndexError' in error_msg or 'KeyError' in error_msg:
        analysis['error_type'] = 'Index/Key Error'
        analysis['common_causes'] = [
            'Accessing index out of range',
            'Key not found in dictionary',
            'Empty list/dict access',
            'Off-by-one errors'
        ]
    elif 'AttributeError' in error_msg:
        analysis['error_type'] = 'Attribute Error'
        analysis['common_causes'] = [
            'Method/property doesn\'t exist on object',
            'NoneType attribute access',
            'Wrong object type',
            'Missing import or module'
        ]
    elif 'ImportError' in error_msg or 'ModuleNotFoundError' in error_msg:
        analysis['error_type'] = 'Import Error'
        analysis['common_causes'] = [
            'Module not installed',
            'Incorrect module name',
            'Circular import',
            'Module not in PYTHONPATH'
        ]
    elif 'ZeroDivisionError' in error_msg:
        analysis['error_type'] = 'Zero Division Error'
        analysis['common_causes'] = [
            'Division by zero',
            'Modulo by zero',
            'Uninitialized denominator'
        ]
    elif 'FileNotFoundError' in error_msg:
        analysis['error_type'] = 'File Not Found Error'
        analysis['common_causes'] = [
            'File doesn\'t exist at path',
            'Wrong file path',
            'Permission denied',
            'Relative path issues'
        ]
    else:
        analysis['error_type'] = 'Runtime Error'
        analysis['common_causes'] = [
            'Logic error in code',
            'Unexpected input data',
            'Resource not available',
            'External dependency failure'
        ]
    
    # Extract line number if present
    import re
    line_match = re.search(r'line (\d+)', error_msg, re.IGNORECASE)
    if line_match:
        analysis['line_number'] = int(line_match.group(1))
    
    # Generate suggestions based on error type
    if analysis['line_number'] and code:
        lines = code.split('\n')
        if 0 < analysis['line_number'] <= len(lines):
            error_line = lines[analysis['line_number'] - 1]
            analysis['error_line'] = error_line.strip()
            
            # Add specific suggestions based on line content
            if analysis['error_type'] == 'Syntax Error':
                if ':' not in error_line and any(kw in error_line for kw in ['if', 'for', 'while', 'def', 'class']):
                    analysis['suggestions'].append("Add a colon (:) at the end of the line")
                if '(' in error_line and ')' not in error_line:
                    analysis['suggestions'].append("Add missing closing parenthesis )")
    
    return analysis

def format_error_analysis(analysis):
    lines = [f"[ERROR ANALYSIS] {analysis['error_type']}"]
    lines.append("-" * 50)
    if analysis['line_number']:
        lines.append(f"Location: Line {analysis['line_number']}")
        if 'error_line' in analysis:
            lines.append(f"Code: {analysis['error_line']}")
    lines.append(f"\nMessage: {analysis['error_message']}")
    if analysis['common_causes']:
        lines.append("\nCommon Causes:")
        for cause in analysis['common_causes']:
            lines.append(f"  • {cause}")
    if analysis['suggestions']:
        lines.append("\nSuggested Fixes:")
        for suggestion in analysis['suggestions']:
            lines.append(f"  → {suggestion}")
    return "\n".join(lines)

def execute_and_capture_errors(code):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, {"__name__": "__main__"})
        return None, stdout_capture.getvalue(), stderr_capture.getvalue()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}", stdout_capture.getvalue(), stderr_capture.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def extract_json_objects(s):
    if not isinstance(s, str):
        return []
    objects = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{':
            start = i
            brace_depth = 0
            in_string = False
            escape_next = False
            while i < n:
                char = s[i]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            try:
                                obj_str = s[start:i+1]
                                obj = json.loads(obj_str)
                                objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                i += 1
        i += 1
    return objects

def check_gemini_available():
    try:
        client.models.list()
        return True
    except:
        return False

# ------------------------------------------------------------
# Action handlers (server mode: return messages instead of executing)
# ------------------------------------------------------------
def create_folder_action(folder: str) -> dict:
    """Return a message to create a folder."""
    return {"type": "create_folder", "folder_path": folder}

def create_file_action(path: str, content: str) -> dict:
    """Return a message to create/update a file."""
    return {"type": "create_file", "file_path": path, "content": content}

def create_files_action(files: List[dict]) -> dict:
    """Return a message to create multiple files."""
    return {"type": "create_files", "files": files}

def status_message(text: str) -> dict:
    return {"type": "status", "text": text}

def error_message(text: str) -> dict:
    return {"type": "error", "text": text}

def response_message(text: str) -> dict:
    return {"type": "response", "text": text}

def confirmation_message(text: str, action: dict) -> dict:
    return {"type": "confirmation", "text": text, "action": action}

# For actions that require confirmation (file exists etc.), we generate a confirmation message.
def handle_create_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    """
    In server mode, we only simulate existence check.
    If not confirmed and file would exist, return confirmation message.
    Otherwise return create_file action.
    """
    # In server mode we cannot know if file exists on client side,
    # so we always just send the create_file action. The extension will handle duplicates.
    # For simplicity, we never ask for confirmation; the extension manages it.
    return [create_file_action(path, content)]

def handle_update_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    return [create_file_action(path, content)]  # same as create
    
def handle_create_project(folder: str, files: List[dict]) -> List[dict]:
    # Return a single compound action
    return [{"type": "create_project", "folder": folder, "files": files}]

def handle_run_file(path: str, environment: str = "none") -> List[dict]:
    # Return a message that the extension will interpret to run the file.
    # The extension will open a terminal and execute.
    return [{"type": "run_file", "path": path, "environment": environment}]

def handle_debug_file(path: str, debug_stage: str = "all") -> List[dict]:
    # For debugging, we could either:
    # 1. Ask the extension to send the file content, then we analyze and return a fixed version.
    # 2. Or we just return a debug_file message and the extension will handle it.
    # Here we'll return a message that the extension will interpret to debug (i.e., send file to /debug endpoint later).
    # But we also need a separate /debug endpoint that accepts file content and returns fixed content.
    # For now, we'll just return a status that we are debugging (the extension will need to implement it).
    return [status_message(f"Debugging {path}... (auto-fix not implemented yet)")]

# ------------------------------------------------------------
# AI processing
# ------------------------------------------------------------
def process_message(user_input: str, conversation_history: str = "") -> str:
    """Call Gemini and return the raw text reply."""
    if not check_gemini_available():
        return "Error: Cannot connect to Gemini API. Please check your API key."
    greeting_keywords = ['hi', 'hello', 'hey', 'help', 'start']
    user_lower = user_input.lower().strip()
    if any(user_lower.startswith(kw) for kw in greeting_keywords) and len(user_input) < 20:
        return "Hello! Great to connect. What are we building today?"
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nConversation history:\n{conversation_history}\n\nUser: {user_input}\nAssistant:"
        response = client.models.generate_content(model=GEMINI_MODEL, contents=full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def process_user_message(request: ChatRequest) -> List[dict]:
    """
    Main entry point for /chat.
    Returns a list of messages (dicts) to be sent back to the extension.
    """
    messages = []

    # 1. If there is a pending action, handle it as a confirmation.
    if request.pending_action:
        action_data = request.pending_action
        act = action_data.get("action") or action_data.get("intent")
        if act == "create_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "update_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "create_folder":
            folder = action_data.get("folder") or action_data.get("folder_path")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder in pending action"))
        elif act == "create_project":
            folder = action_data.get("folder") or action_data.get("project")
            files = action_data.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder or files in pending action"))
        elif act == "run_file":
            path = action_data.get("path") or action_data.get("file_path")
            env = action_data.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path in pending action"))
        else:
            messages.append(error_message(f"Unknown pending action: {act}"))
        return messages

    # 2. No pending action: process the user message normally.
    assistant_reply = process_message(request.message, request.conversation_history)
    messages.append(response_message(assistant_reply))

    # 3. Extract JSON actions from the reply and handle them.
    json_objects = extract_json_objects(assistant_reply)
    for obj in json_objects:
        action = obj.get("action") or obj.get("intent")
        if not action:
            continue
        act = action.strip().lower()

        if act in ("create_folder", "create folder", "createfolder"):
            folder = obj.get("folder") or obj.get("name")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder name"))

        elif act in ("create_project", "create project", "createproject"):
            folder = obj.get("folder") or obj.get("name") or obj.get("project")
            files = obj.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder name or files list"))

        elif act in ("create_file", "create file", "createfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("update_file", "update file", "updatefile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("run_file", "run file", "runfile", "test_file", "test file", "testfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            env = obj.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path"))

        elif act in ("debug_file", "debug file", "debugfile"):
            path = obj.get("path") or obj.get("filename") or obj.get("file")
            stage = obj.get("stage", "all")
            if path:
                messages.extend(handle_debug_file(path, stage))
            else:
                messages.append(error_message("Missing path"))

        # For search actions, we could either handle them here (by searching the server's filesystem, which is useless)
        # or we can forward them to the extension. Since the extension already implements search locally,
        # we can ignore these actions or return a message that they are handled locally.
        # To keep compatibility, we'll just ignore them (the extension will not receive them anyway because they are in the AI text).
        # But if they appear as separate JSON, we should perhaps return a status that search is done locally.
        elif act in ("search_files", "search files", "searchfiles"):
            messages.append(status_message("File search is handled locally by the extension."))
        elif act in ("search_folders", "search folders", "searchfolders"):
            messages.append(status_message("Folder search is handled locally by the extension."))
        elif act in ("search_in_files", "search in files", "searchinfiles", "grep"):
            messages.append(status_message("Content search is handled locally by the extension."))
        elif act in ("get_file_info", "get file info", "getfileinfo", "file_info"):
            messages.append(status_message("File info is handled locally by the extension."))
        else:
            messages.append(error_message(f"Unknown action: {act}"))

    return messages

# ------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "VibeCoding AI Backend is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        msgs = process_user_message(request)
        return ChatResponse(messages=msgs)
    except Exception as e:
        return ChatResponse(messages=[{"type": "error", "text": f"Server error: {str(e)}"}])

# Optional /debug endpoint for file fixing (if needed)
class DebugRequest(BaseModel):
    file_path: str
    content: str

@app.post("/debug")
async def debug_endpoint(req: DebugRequest):
    """
    Receives a file's content, analyzes it, and returns a fixed version.
    """
    try:
        # Here you could call Gemini to fix the file.
        # For now, return the content unchanged.
        return {"fixed_content": req.content, "errors": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# For running directly (not used on Render)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)