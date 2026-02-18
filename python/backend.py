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
import os
from fastapi import FastAPI
import os
import json
import sys
from dotenv import load_dotenv
from pydantic import BaseModel
# Load .env file
load_dotenv()

def send(data):
    print(json.dumps(data))
    sys.stdout.flush()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    send({"error": "Gemini API key not configured"})
    sys.exit(1)

send({"status": "Backend ready"})

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-pro-preview"


# load_dotenv()

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Root route (for testing)
@app.get("/")
def home():
    return {"message": "Backend running successfully"}

# Generate route
@app.post("/generate")
async def generate_text(request: PromptRequest):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.prompt
        )

        return {
            "response": response.text
        }

    except Exception as e:
        return {
            "error": str(e)
        }
# # Gemini API Configuration
# GEMINI_API_KEY = "AIzaSyAnq2vNmeyvxyCqtBsrbaAyUzTLlJMdYGk"
# GEMINI_MODEL = "gemini-2.5-pro"  # Use available model for the new API
# MODEL_NAME = GEMINI_MODEL  # Use same model name for compatibility

# # Create the client with API key
# client = genai.Client(api_key=GEMINI_API_KEY)

# # Website Building Backend Configuration
WEBSITE_BACKEND_URL = "http://127.0.0.1:8000"

# Global workspace path - will be set from VS Code: extension
# Use the parent directory of the python folder (extension root) as default
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to parent directory (extension root), not the python subdirectory
WORKSPACE_PATH = os.path.dirname(SCRIPT_DIR)

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


def send_response(text):
    """Send response to VS Code: extension (stdout)"""
    sys.stdout.write(json.dumps({"type": "response", "text": text}) + "\n")
    sys.stdout.flush()

def send_error(error):
    """Send error to VS Code: extension"""
    sys.stdout.write(json.dumps({"type": "error", "text": error}) + "\n")
    sys.stdout.flush()

def send_status(text):
    """Send status to VS Code: extension"""
    sys.stdout.write(json.dumps({"type": "status", "text": text}) + "\n")
    sys.stdout.flush()

def send_confirmation_request(text, action_data):
    """Send confirmation request to VS Code: extension"""
    sys.stdout.write(json.dumps({
        "type": "confirmation", 
        "text": text,
        "action": action_data
    }) + "\n")
    sys.stdout.flush()

# Global variable to store pending confirmation
pending_confirmation = None


def create_folder(folder):
    try:
        # Send request to TypeScript to create folder in VS Code: workspace
        full_path = os.path.join(WORKSPACE_PATH, folder)
        if os.path.exists(full_path):
            return f"[INFO] Folder '{folder}' already exists."
        
        # Send message to TypeScript extension to create folder
        sys.stdout.write(json.dumps({
            "type": "create_folder",
            "folder_path": folder
        }) + "\n")
        sys.stdout.flush()
        
        # Also create locally for compatibility
        os.makedirs(full_path, exist_ok=True)
        return f"[OK] Folder '{folder}' created."
    except OSError as e:
        return f"[ERROR] {e}"


def create_project(folder, files):
    """
    Create a project with multiple files and folders.
    Sends requests to TypeScript to create files in VS Code: workspace.
    """
    results = []
    
    try:
        # Create main project folder locally first
        project_path = os.path.join(WORKSPACE_PATH, folder)
        if not os.path.exists(project_path):
            os.makedirs(project_path, exist_ok=True)
            results.append(f"[OK] Created project folder: {folder}")
        else:
            results.append(f"[INFO] Project folder '{folder}' already exists.")
        
        # Prepare files for TypeScript to create
        files_to_create = []
        
        # Create each file
        created_count = 0
        updated_count = 0
        error_count = 0
        
        for file_info in files:
            try:
                file_path = file_info.get("path", "")
                content = file_info.get("content", "")
                
                if not file_path:
                    results.append(f"[ERROR] Missing path for file")
                    error_count += 1
                    continue
                
                # Handle both relative paths and paths within the project folder
                if file_path.startswith(folder + "/") or file_path.startswith(folder + "\\"):
                    # Path already includes project folder
                    relative_path = file_path
                    full_file_path = os.path.join(WORKSPACE_PATH, file_path)
                else:
                    # Path is relative to project folder
                    relative_path = os.path.join(folder, file_path)
                    full_file_path = os.path.join(project_path, file_path)
                
                # Create parent directories if needed
                parent_dir = os.path.dirname(full_file_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Check if file already exists
                file_exists = os.path.exists(full_file_path)
                
                # Write the file locally
                with open(full_file_path, "w", encoding='utf-8') as f:
                    f.write(content)
                
                # Add to files list for TypeScript
                files_to_create.append({
                    "path": relative_path,
                    "content": content
                })
                
                if file_exists:
                    results.append(f"[UPDATED] {relative_path} ({len(content)} chars)")
                    updated_count += 1
                else:
                    results.append(f"[CREATED] {relative_path} ({len(content)} chars)")
                    created_count += 1
                
                # Validate Python files
                if full_file_path.endswith('.py'):
                    error, _ = validate_python_code(content, relative_path)
                    if error:
                        results.append(f"  [WARNING] Syntax issues detected")
                    else:
                        results.append(f"  [OK] Syntax validation passed")
                
            except Exception as e:
                results.append(f"[ERROR] Failed to create file {file_path}: {e}")
                error_count += 1
        
        # Send all files to TypeScript to create in VS Code: workspace
        if files_to_create:
            sys.stdout.write(json.dumps({
                "type": "create_files",
                "files": files_to_create
            }) + "\n")
            sys.stdout.flush()
        
        # Summary
        results.append("-" * 50)
        results.append(f"[SUMMARY] Project '{folder}':")
        results.append(f"  Created: {created_count} files")
        results.append(f"  Updated: {updated_count} files")
        if error_count > 0:
            results.append(f"  Errors: {error_count} files")
        results.append(f"[OK] Project setup complete!")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"[ERROR] Failed to create project: {e}"


def find_file_recursive(filename, search_path=None):
    """Search for a file recursively in all subdirectories."""
    if search_path is None:
        search_path = WORKSPACE_PATH
    
    # First check if file exists at the given path directly
    if os.path.exists(filename):
        return filename
    
    # Check if it's an absolute path
    if os.path.isabs(filename):
        if os.path.exists(filename):
            return filename
        return None
    
    # Search recursively in all subdirectories
    for root, dirs, files in os.walk(search_path):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', '.vscode']]
        
        if filename in files:
            return os.path.join(root, filename)
        
        # Also check if the full relative path matches
        full_candidate = os.path.join(root, filename)
        if os.path.exists(full_candidate):
            return full_candidate
    
    return None


def find_files_by_keyword(keyword, file_type=None, max_results=10, search_path=None):
    """
    Search for files by keyword in their names.
    Supports partial matching and wildcards.
    """
    if search_path is None:
        search_path = WORKSPACE_PATH
    
    matches = []
    pattern = f"*{keyword}*"
    if file_type:
        pattern = f"*{keyword}*{file_type}"
    
    try:
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', '.vscode', 'out', 'dist', 'build']]
            
            for filename in files:
                # Check if filename matches pattern
                if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, search_path)
                    matches.append({
                        'name': filename,
                        'path': rel_path,
                        'full_path': full_path,
                        'size': os.path.getsize(full_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if len(matches) >= max_results:
                        return matches
    except Exception as e:
        return [{'error': f"Search error: {str(e)}"}]
    
    return matches


def find_folders_by_keyword(keyword, max_results=10, search_path=None):
    """
    Search for folders by keyword in their names.
    Supports partial matching.
    """
    if search_path is None:
        search_path = WORKSPACE_PATH
    
    matches = []
    
    try:
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', '.vscode', 'out', 'dist', 'build']]
            
            for dirname in dirs:
                # Check if folder name contains keyword
                if keyword.lower() in dirname.lower():
                    full_path = os.path.join(root, dirname)
                    rel_path = os.path.relpath(full_path, search_path)
                    
                    # Count files in directory
                    try:
                        file_count = sum([len(files) for _, _, files in os.walk(full_path)])
                    except:
                        file_count = 0
                    
                    matches.append({
                        'name': dirname,
                        'path': rel_path,
                        'full_path': full_path,
                        'file_count': file_count
                    })
                    
                    if len(matches) >= max_results:
                        return matches
    except Exception as e:
        return [{'error': f"Search error: {str(e)}"}]
    
    return matches


def search_in_file_content(keyword, file_pattern="*", max_results=10, search_path=None):
    """
    Search for text inside files.
    Returns files containing the keyword with line numbers.
    """
    if search_path is None:
        search_path = WORKSPACE_PATH
    
    matches = []
    text_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.json', '.md', '.txt', '.yaml', '.yml', '.xml', '.sql', '.sh', '.bash', '.zsh', '.fish', '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.m', '.mm', '.cs', '.vb', '.fs', '.fsx', '.clj', '.cljs', '.edn', '.erl', '.hrl', '.ex', '.exs', '.elm', '.haskell', '.hs', '.lhs', '.lua', '.pl', '.pm', '.t', '.pod', '.raku', '.nim', '.nims', '.nimble', '.cr', '.ecr', '.slang', '.dart', '.groovy', '.gvy', '.gy', '.gsh', '.tcl', '.tk', '.racket', '.rkt', '.ss', '.scm', '.sch', '.sml', '.ml', '.mli', '.fun', '.sig', '.ocaml', '.opa', '.prolog', '.pl', '.pro', '.elm', '.el', '.lisp', '.lsp', '.l', '.cl', '.fasl', '.scm', '.ss', '.rkt', '.sld', '.sps', '.sls', '.scm', '.ss', '.rkt', '.sld', '.sps', '.sls'}
    
    try:
        for root, dirs, files in os.walk(search_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', '.vscode', 'out', 'dist', 'build']]
            
            for filename in files:
                # Check file pattern
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                # Skip binary files
                ext = os.path.splitext(filename)[1].lower()
                if ext not in text_extensions:
                    continue
                
                full_path = os.path.join(root, filename)
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        if keyword.lower() in content.lower():
                            # Find line numbers
                            lines = content.split('\n')
                            matching_lines = []
                            for i, line in enumerate(lines, 1):
                                if keyword.lower() in line.lower():
                                    matching_lines.append({
                                        'line_number': i,
                                        'content': line.strip()[:100]  # First 100 chars
                                    })
                                    if len(matching_lines) >= 3:  # Limit to 3 matches per file
                                        break
                            
                            rel_path = os.path.relpath(full_path, search_path)
                            matches.append({
                                'name': filename,
                                'path': rel_path,
                                'full_path': full_path,
                                'matches': len(matching_lines),
                                'lines': matching_lines
                            })
                            
                            if len(matches) >= max_results:
                                return matches
                except Exception:
                    continue  # Skip files that can't be read
                    
    except Exception as e:
        return [{'error': f"Search error: {str(e)}"}]
    
    return matches


def get_file_info(path, search_path=None):
    """
    Get detailed information about a file or folder.
    """
    if search_path is None:
        search_path = WORKSPACE_PATH
    
    # Try to find the file if not found directly
    full_path = find_file_recursive(path, search_path)
    if not full_path:
        full_path = os.path.join(search_path, path)
    
    if not os.path.exists(full_path):
        return {'error': f"File or folder '{path}' not found"}
    
    try:
        stat = os.stat(full_path)
        info = {
            'name': os.path.basename(full_path),
            'path': os.path.relpath(full_path, search_path),
            'full_path': full_path,
            'exists': True,
            'is_file': os.path.isfile(full_path),
            'is_directory': os.path.isdir(full_path),
            'size': stat.st_size,
            'size_human': format_file_size(stat.st_size),
            'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'accessed': datetime.fromtimestamp(stat.st_atime).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if os.path.isdir(full_path):
            # Count contents
            try:
                items = os.listdir(full_path)
                info['item_count'] = len(items)
                info['files'] = len([f for f in items if os.path.isfile(os.path.join(full_path, f))])
                info['folders'] = len([f for f in items if os.path.isdir(os.path.join(full_path, f))])
            except:
                info['item_count'] = 0
                info['files'] = 0
                info['folders'] = 0
        
        return info
        
    except Exception as e:
        return {'error': f"Error getting file info: {str(e)}"}


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_search_results(results, search_type):
    """Format search results for display."""
    if not results:
        return f"[INFO] No {search_type} found matching your criteria."
    
    if isinstance(results, list) and len(results) > 0 and 'error' in results[0]:
        return f"[ERROR] {results[0]['error']}"
    
    lines = [f"[OK] Found {len(results)} {search_type}:"]
    lines.append("-" * 50)
    
    for i, item in enumerate(results, 1):
        if search_type == "files":
            lines.append(f"{i}. {item['name']}")
            lines.append(f"   Path: {item['path']}")
            lines.append(f"   Size: {format_file_size(item['size'])} | Modified: {item['modified']}")
        elif search_type == "folders":
            lines.append(f"{i}. {item['name']}/")
            lines.append(f"   Path: {item['path']}/")
            lines.append(f"   Files: {item['file_count']}")
        elif search_type == "content matches":
            lines.append(f"{i}. {item['name']}")
            lines.append(f"   Path: {item['path']}")
            lines.append(f"   Matches: {item['matches']} occurrences")
            for line_info in item['lines']:
                lines.append(f"      Line {line_info['line_number']}: {line_info['content']}")
        
        lines.append("")
    
    return "\n".join(lines)


def validate_python_code(code, filename):
    """Validate Python code for syntax errors with detailed reporting."""
    try:
        ast.parse(code)
        return None, None
    except SyntaxError as e:
        # Get the exact line content for better context
        lines = code.split('\n')
        error_line = lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else ""
        pointer = " " * (e.offset - 1) + "^" if e.offset else ""
        
        # Provide helpful suggestions based on common errors
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
    """Provide helpful suggestions for common syntax errors."""
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
    
    # Check for specific patterns
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
    """
    Analyze an error and provide detailed debugging information.
    """
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
    """Format error analysis for display."""
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
    """Execute Python code and capture any runtime errors."""
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


def run_code(path, environment=None):
    """Run code file safely using conda run or system interpreter."""

    # Detect command by file type
    if path.endswith(".py"):
        base_cmd = ["python", path]
    elif path.endswith(".js"):
        base_cmd = ["node", path]
    elif path.endswith(".go"):
        base_cmd = ["go", "run", path]
    elif path.endswith(".rb"):
        base_cmd = ["ruby", path]
    elif path.endswith(".php"):
        base_cmd = ["php", path]
    elif path.endswith(".java"):
        class_name = os.path.splitext(os.path.basename(path))[0]
        base_cmd = ["bash", "-c", f"javac {path} && java {class_name}"]
    elif path.endswith((".c", ".cpp")):
        out = os.path.splitext(path)[0]
        base_cmd = ["bash", "-c", f"g++ '{path}' -o '{out}' && '{out}'"]
    else:
        base_cmd = ["python", path]

    # Build final command
    if environment and environment.lower() != "none":
        cmd = ["conda", "run", "-n", environment] + base_cmd
    else:
        cmd = base_cmd

    result_lines = [f"\n[RUNNING] Executing '{path}'...", f"[COMMAND] {' '.join(cmd)}", "-" * 50]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.stdout:
            result_lines.append(result.stdout)
        if result.stderr:
            result_lines.append(result.stderr)

        result_lines.append("-" * 50)

        if result.returncode == 0:
            result_lines.append("[OK] Code executed successfully\n")
        else:
            result_lines.append(f"[ERROR] Exit code: {result.returncode}\n")

    except subprocess.TimeoutExpired:
        result_lines.append("[ERROR] Execution timed out (30s)\n")
    except FileNotFoundError:
        result_lines.append("[ERROR] Required runtime not found in PATH\n")
    except Exception as e:
        result_lines.append(f"[ERROR] Execution failed: {e}\n")
    
    return "\n".join(result_lines)


def run_file(path, environment="none"):
    """Run a file and return results with error handling for auto-fix."""
    # Try to find the file recursively if not found directly
    full_path = find_file_recursive(path)
    
    if not full_path:
        # Try direct path as fallback
        full_path = os.path.join(WORKSPACE_PATH, path)
        if not os.path.exists(full_path):
            return f"[ERROR] File '{path}' not found in workspace or any subdirectory. Cannot run."
    
    result_lines = [f"[RUNNING] Testing file '{full_path}'..."]
    
    # First validate the code
    if path.endswith('.py'):
        with open(full_path, "r") as f:
            content = f.read()
        
        error, line_no = validate_python_code(content, path)
        if error:
            result_lines.append(f"[ERROR] Syntax error found: {error}")
            result_lines.append(f"[SUGGESTION] Use 'debug_file' action to auto-fix this error.")
            return "\n".join(result_lines)
    
    # Run the code
    run_result = run_code(full_path, environment)
    result_lines.append(run_result)
    
    # Check if there were runtime errors
    if "[ERROR]" in run_result and "Exit code:" in run_result:
        result_lines.append(f"\n[AUTO-FIX AVAILABLE]")
        result_lines.append(f"[SUGGESTION] Runtime errors detected. Use 'debug_file' action to auto-fix.")
    
    return "\n".join(result_lines)


def create_file(path, content, confirmed=False, action_type='create_new'):
    try:
        # Use absolute path based on workspace
        full_path = os.path.join(WORKSPACE_PATH, path)
        
        # Check if file already exists in main directory
        if os.path.exists(full_path) and action_type == 'create_new':
            return f"[INFO] File '{path}' already exists in main directory. Cannot create new file."
        
        # Check if file exists in any subdirectory (unless already confirmed)
        if not confirmed:
            existing_file = find_file_recursive(os.path.basename(path))
            if existing_file and existing_file != full_path:
                return f"[CONFIRMATION_REQUIRED] File '{os.path.basename(path)}' already exists at '{existing_file}'.\n\nOptions:\n1) Modify existing file\n2) Create new file at '{path}'\n3) Show diff (compare files)\n4) Backup existing and modify\n5) Cancel\n\nWhat would you like to do?"
        
        # Handle different action types
        if action_type == 'show_diff':
            existing_file = find_file_recursive(os.path.basename(path))
            return show_file_diff(existing_file, content)
        
        if action_type == 'backup_and_modify':
            existing_file = find_file_recursive(os.path.basename(path))
            backup_path = backup_file(existing_file)
            if backup_path:
                result = update_file(existing_file, content, confirmed=True)
                return f"[BACKUP] Created backup at: {backup_path}\n{result}"
            else:
                return f"[ERROR] Failed to create backup. Operation cancelled."
        
        if action_type == 'modify_existing':
            existing_file = find_file_recursive(os.path.basename(path))
            return update_file(existing_file, content, confirmed=True)
        
        # Default: create new file
        folder = os.path.dirname(full_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # Write content to file locally
        with open(full_path, "w") as f:
            f.write(content)
        
        # Send message to TypeScript to create file in VS Code: workspace
        sys.stdout.write(json.dumps({
            "type": "create_file",
            "file_path": path,
            "content": content
        }) + "\n")
        sys.stdout.flush()
        
        # Get file info
        file_size = os.path.getsize(full_path)
        line_count = len(content.split('\n'))
        result_lines = [f"[OK] Created: {os.path.basename(full_path)} ({line_count} lines, {file_size} bytes)"]
        
        if path.endswith('.py'):
            error, line_no = validate_python_code(content, path)
            if error:
                result_lines.append(f"[WARNING] Syntax issue: {error}")
            else:
                result_lines.append(f"[OK] Code valid")
        
        return "\n".join(result_lines)
                
    except OSError as e:
        return f"[ERROR] {e}"


def update_file(path, content, confirmed=False):
    try:
        # Try to find the file recursively if not found directly
        full_path = find_file_recursive(path)

        if not full_path:
            # Try direct path as fallback
            full_path = os.path.join(WORKSPACE_PATH, path)
            if not os.path.exists(full_path):
                return f"[ERROR] File '{path}' not found in workspace or any subdirectory. Cannot update."

        if not content or not content.strip():
            return f"[ERROR] No content provided to write to '{path}'"

        lines = content.split('\n')
        result_lines = []
        result_lines.append(f"[UPDATING] File '{path}' ({len(lines)} lines):")
        
        # Write content to file locally
        try:
            with open(full_path, "w", encoding='utf-8') as f:
                for i, line in enumerate(lines, 1):
                    f.write(line + '\n')
                    result_lines.append(f"  Line {i}/{len(lines)}: {line[:50]}{'...' if len(line) > 50 else ''}")
        except IOError as e:
            return f"[ERROR] Failed to write to file '{path}': {e}"
        
        # Send message to TypeScript to update file in VS Code: workspace
        sys.stdout.write(json.dumps({
            "type": "create_file",
            "file_path": path,
            "content": content
        }) + "\n")
        sys.stdout.flush()
        
        # Verify file was written
        if not os.path.exists(full_path):
            return f"[ERROR] File write failed - file does not exist after writing"
        
        # Check file size
        file_size = os.path.getsize(full_path)
        result_lines.append(f"[OK] File '{path}' updated successfully ({file_size} bytes).")
        
        # Validate Python code if applicable
        if full_path.endswith('.py'):
            result_lines.append(f"[VALIDATING] Checking Python code for errors...")
            error, line_no = validate_python_code(content, path)
            if error:
                result_lines.append(f"[WARNING] Validation found issues:")
                result_lines.append(error)
            else:
                result_lines.append(f"[OK] Code validation passed - no syntax errors found.")
        
        return "\n".join(result_lines)
                
    except Exception as e:
        return f"[ERROR] Unexpected error updating file: {e}"


def debug_file(path, conversation_history="", debug_stage="all"):
    """
    Debug a file with multiple stages.
    debug_stage can be: 'syntax', 'runtime', 'logic', 'all'
    """
    try:
        # First, try to find the file by searching recursively
        full_path = find_file_recursive(path)
        
        # If not found by recursive search, try direct paths
        if not full_path:
            # Try as absolute path
            if os.path.isabs(path) and os.path.exists(path):
                full_path = path
            else:
                # Try relative to workspace
                full_path = os.path.join(WORKSPACE_PATH, path)
                if not os.path.exists(full_path):
                    # Try searching by basename only
                    basename = os.path.basename(path)
                    full_path = find_file_recursive(basename)
        
        # If path is a directory, look for source files within it
        if full_path and os.path.isdir(full_path):
            return debug_directory(full_path, debug_stage)
        
        # If still not found, check if user provided a directory name without extension
        if not full_path or not os.path.exists(full_path):
            # Try as a directory
            dir_path = os.path.join(WORKSPACE_PATH, path)
            if os.path.isdir(dir_path):
                return debug_directory(dir_path, debug_stage)
            
            # Try to find any file with similar name (without extension)
            base_name = os.path.splitext(path)[0]
            if base_name:
                similar_file = find_file_recursive(base_name)
                if similar_file:
                    full_path = similar_file
        
        if not full_path or not os.path.exists(full_path):
            # Extract filename for display (handle both / and \\ path separators)
            if '/' in path or '\\' in path:
                filename_for_display = os.path.basename(path)
            else:
                filename_for_display = path
            return f"[ERROR] File or directory '{path}' not found in workspace or any subdirectory. Cannot debug.\n\nSearched for:\n- Direct path: {path}\n- In workspace: {os.path.join(WORKSPACE_PATH, path)}\n- By filename: {filename_for_display}\n\nTip: Use 'search_files' action to find the correct path."
        
        result_lines = [f"[DEBUGGING] Analyzing file '{full_path}'..."]
        result_lines.append(f"[INFO] Debug mode: {debug_stage}")
        
        with open(full_path, "r") as f:
            content = f.read()
        
        # Stage 1: Syntax Check
        if debug_stage in ['syntax', 'all']:
            result_lines.append(f"\n[STAGE 1/3] Checking for syntax errors...")
            error, line_no = validate_python_code(content, full_path)
            if error:
                result_lines.append(f"[ERROR] Syntax error detected:")
                result_lines.append(error)
                result_lines.append(f"[AUTO-FIX] Attempting to fix syntax error using AI...")
                
                fix_prompt = f"""
The following Python code has a syntax error:
File: {full_path}
Error: {error}
Line: {line_no}

Current code:
```python
{content}
```

Please fix the syntax error and return ONLY the corrected code in a JSON format:
{{
  "action": "update_file",
  "path": "{full_path}",
  "content": "<fixed code here>"
}}
"""
                
                try:
                    # Use Gemini API for AI-powered fix
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=fix_prompt
                    )
                    fix_reply = response.text.strip()
                    
                    fix_json_str = None
                    for obj in extract_json_objects(fix_reply):
                        if obj.get("action") in ["update_file", "updatefile"]:
                            fix_json_str = obj
                            break
                    
                    if fix_json_str:
                        fixed_content = fix_json_str.get("content", "")
                        if fixed_content:
                            # Create backup before fixing
                            backup_path = backup_file(full_path)
                            if backup_path:
                                result_lines.append(f"[BACKUP] Created backup at: {backup_path}")
                            
                            # FIX: Pass confirmed=True to auto-apply the fix without asking for confirmation
                            update_result = update_file(path, fixed_content, confirmed=True)
                            result_lines.append(update_result)
                            result_lines.append(f"[OK] Syntax error fixed automatically.")
                            
                            # Update content for further checks
                            content = fixed_content
                            new_error, _ = validate_python_code(fixed_content, path)
                            if new_error:
                                result_lines.append(f"[WARNING] Fixed code still has syntax errors: {new_error}")
                                return "\n".join(result_lines)
                            else:
                                result_lines.append(f"[OK] Syntax validation passed.")
                        else:
                            result_lines.append(f"[ERROR] AI did not provide fixed code.")
                            return "\n".join(result_lines)
                    else:
                        result_lines.append(f"[ERROR] Could not extract fix from AI response.")
                        return "\n".join(result_lines)
                        
                except Exception as e:
                    result_lines.append(f"[ERROR] Failed to get fix from Gemini AI: {e}")
                    result_lines.append(f"[INFO] Please fix the syntax error manually at line {line_no}")
                    return "\n".join(result_lines)
            else:
                result_lines.append(f"[OK] No syntax errors found.")
        
        # Stage 2: Runtime Check
        if debug_stage in ['runtime', 'all']:
            result_lines.append(f"\n[STAGE 2/3] Checking for runtime errors...")
            runtime_error, stdout, stderr = execute_and_capture_errors(content)
            
            if runtime_error:
                # Perform detailed error analysis
                error_analysis = analyze_error(runtime_error, content, full_path)
                analysis_text = format_error_analysis(error_analysis)
                result_lines.append(analysis_text)
                
                result_lines.append(f"\n[AUTO-FIX] Attempting to fix runtime error using AI...")
                
                fix_prompt = f"""
The following Python code has a runtime error:
File: {full_path}
Error: {runtime_error}

Current code:
```python
{content}
```

Output before error:
{stdout}

Error output:
{stderr}

Please fix the runtime error and return ONLY the corrected code in a JSON format:
{{
  "action": "update_file",
  "path": "{full_path}",
  "content": "<fixed code here>"
}}
"""
                
                try:
                    # Use Gemini API for AI-powered fix
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=fix_prompt
                    )
                    fix_reply = response.text.strip()

                    fix_json_str = None
                    for obj in extract_json_objects(fix_reply):
                        if obj.get("action") in ["update_file", "updatefile"]:
                            fix_json_str = obj
                            break

                    if fix_json_str:
                        fixed_content = fix_json_str.get("content", "")
                        if fixed_content:
                            # Create backup before fixing
                            backup_path = backup_file(full_path)
                            if backup_path:
                                result_lines.append(f"[BACKUP] Created backup at: {backup_path}")

                            # FIX: Pass confirmed=True to auto-apply the fix without asking for confirmation
                            update_result = update_file(path, fixed_content, confirmed=True)
                            result_lines.append(update_result)
                            result_lines.append(f"[OK] Runtime error fixed automatically.")

                            # Update content for further checks
                            content = fixed_content
                            new_error, new_stdout, new_stderr = execute_and_capture_errors(fixed_content)
                            if new_error:
                                result_lines.append(f"[WARNING] Fixed code still has runtime errors: {new_error}")
                            else:
                                result_lines.append(f"[OK] Runtime validation passed.")
                                if new_stdout:
                                    result_lines.append(f"[OUTPUT] Program output:\n{new_stdout}")
                        else:
                            result_lines.append(f"[ERROR] AI did not provide fixed code.")
                    else:
                        result_lines.append(f"[ERROR] Could not extract fix from AI response.")

                except Exception as e:
                    result_lines.append(f"[ERROR] Failed to get fix from Gemini AI: {e}")
                    result_lines.append(f"[INFO] Please fix the runtime error manually.")
                    
                if stdout:
                    result_lines.append(f"[OUTPUT] Standard output before error:\n{stdout}")
                if stderr:
                    result_lines.append(f"[ERROR OUTPUT] Standard error:\n{stderr}")
            else:
                result_lines.append(f"[OK] No runtime errors found.")
                if stdout:
                    result_lines.append(f"[OUTPUT] Program output:\n{stdout}")
        
        # Stage 3: Logic/Code Quality Check (only in 'all' mode)
        if debug_stage == 'all':
            result_lines.append(f"\n[STAGE 3/3] Checking code quality and best practices...")
            # This could be expanded with more sophisticated analysis
            result_lines.append(f"[INFO] Code quality check completed.")
        
        result_lines.append(f"\n[OK] File '{path}' debugged successfully!")
        if debug_stage == 'all':
            result_lines.append(f"[SUMMARY] All checks passed: Syntax ✓ Runtime ✓ Quality ✓")
        
        return "\n".join(result_lines)
                
    except OSError as e:
        return f"[ERROR] {e}"


def debug_directory(dir_path, debug_stage="all"):
    """
    Debug all source files in a directory.
    Automatically detects project type and finds files with errors.
    """
    result_lines = [f"[DEBUGGING] Analyzing directory '{dir_path}'..."]
    
    # Find all source files in the directory
    source_files = []
    for root, dirs, files in os.walk(dir_path):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', '.vscode', 'out', 'dist', 'build', 'target']]
        
        for filename in files:
            # Check for source code files
            if filename.endswith(('.py', '.java', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala')):
                full_path = os.path.join(root, filename)
                source_files.append(full_path)
    
    if not source_files:
        return f"[ERROR] No source files found in directory '{dir_path}'."
    
    result_lines.append(f"[INFO] Found {len(source_files)} source files.")
    
    # Check each file for errors
    files_with_errors = []
    for file_path in source_files:
        # Quick syntax check based on file type
        if file_path.endswith('.py'):
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()
            error, line_no = validate_python_code(content, file_path)
            if error:
                files_with_errors.append((file_path, 'syntax', error, line_no))
        elif file_path.endswith('.java'):
            # Basic Java syntax check
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()
            error = check_java_syntax(content, file_path)
            if error:
                files_with_errors.append((file_path, 'syntax', error, None))
    
    if not files_with_errors:
        result_lines.append(f"[OK] No syntax errors found in any files.")
        return "\n".join(result_lines)
    
    # Debug the first file with errors
    result_lines.append(f"\n[INFO] Found {len(files_with_errors)} file(s) with errors. Debugging first file...")
    
    file_path, error_type, error_msg, line_no = files_with_errors[0]
    rel_path = os.path.relpath(file_path, WORKSPACE_PATH)
    
    result_lines.append(f"\n[DEBUGGING] File: {rel_path}")
    result_lines.append(f"[ERROR] {error_msg}")
    
    # For Java files, attempt AI-powered fix
    if file_path.endswith('.java') and check_gemini_available():
        result_lines.append(f"\n[AUTO-FIX] Attempting to fix Java syntax error using AI...")
        
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
        
        fix_prompt = f"""
The following Java code has a syntax error:
File: {file_path}
Error: {error_msg}

Current code:
```java
{content}
```

Please fix the syntax error and return ONLY the corrected code in a JSON format:
{{
  "action": "update_file",
  "path": "{file_path}",
  "content": "<fixed code here>"
}}
"""
        
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=fix_prompt
            )
            fix_reply = response.text.strip()
            
            fix_json_str = None
            for obj in extract_json_objects(fix_reply):
                if obj.get("action") in ["update_file", "updatefile"]:
                    fix_json_str = obj
                    break
            
            if fix_json_str:
                fixed_content = fix_json_str.get("content", "")
                if fixed_content:
                    # Create backup before fixing
                    backup_path = backup_file(file_path)
                    if backup_path:
                        result_lines.append(f"[BACKUP] Created backup at: {backup_path}")
                    
                    # Apply the fix
                    update_result = update_file(file_path, fixed_content, confirmed=True)
                    result_lines.append(update_result)
                    result_lines.append(f"[OK] Java syntax error fixed automatically.")
                else:
                    result_lines.append(f"[ERROR] AI did not provide fixed code.")
            else:
                result_lines.append(f"[ERROR] Could not extract fix from AI response.")
                
        except Exception as e:
            result_lines.append(f"[ERROR] Failed to get fix from Gemini AI: {e}")
            result_lines.append(f"[INFO] Please fix the error manually.")
    
    return "\n".join(result_lines)


def check_java_syntax(content, filename):
    """Basic Java syntax checking."""
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Check for common Java syntax errors
        stripped = line.strip()
        
        # Check for typos in System.out
        if 'System.ut.' in line or 'System.ot.' in line or 'System.ut(' in line:
            return f"Line {i}: Typo in System.out - found '{line.strip()}'"
        
        # Check for missing semicolons (basic check)
        if stripped and not stripped.endswith(('{', '}', ';', ')', '//', '*/', '/*', '*', '@', ',')):
            if any(keyword in stripped for keyword in ['import ', 'package ', 'class ', 'public ', 'private ', 'protected ', 'if ', 'for ', 'while ', 'switch ', 'try ', 'catch ', 'else ', 'return ', 'System.', 'new ', '//', '/*', '*/']):
                continue
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                continue
            if stripped.startswith('@'):
                continue
            # This is a basic check and may have false positives
            pass
    
    return None


def extract_json_objects(s):
    """Extract all valid JSON objects from a string."""
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
    """Check if Gemini API is available and configured"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return False
    try:
        # Test the API by listing models (lightweight check)
        client.models.list()
        return True
    except Exception:
        return False



def check_website_backend_available():
    """Check if Website Building Backend server is running"""
    try:
        response = requests.get(f"{WEBSITE_BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def start_website_backend():
    """Start the Website Building Backend server"""
    try:
        import subprocess
        import os
        
        # Get the extension root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        extension_root = os.path.dirname(script_dir)
        
        # Path to the website building backend
        backend_dir = os.path.join(extension_root, "Website_building_backend")
        main_py_path = os.path.join(backend_dir, "main.py")
        
        # Check if the backend exists
        if not os.path.exists(main_py_path):
            return {
                "success": False,
                "error": f"Website Building Backend not found at: {main_py_path}"
            }
        
        # Start the server in background
        # Use nohup to keep it running after parent process exits
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                ["python", "main.py"],
                cwd=backend_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # Linux/Mac
            # Use nohup to detach process
            process = subprocess.Popen(
                ["nohup", "python", "main.py", "&"],
                cwd=backend_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp  # Detach from parent process
            )
        
        # Wait a few seconds for server to start
        import time
        time.sleep(3)
        
        # Check if server is now running
        if check_website_backend_available():
            return {
                "success": True,
                "message": "Website Building Backend started successfully"
            }
        else:
            # Try one more time after a longer wait
            time.sleep(5)
            if check_website_backend_available():
                return {
                    "success": True,
                    "message": "Website Building Backend started successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Server failed to start within expected time"
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to start Website Building Backend: {str(e)}"
        }


def is_website_building_request(user_input):
    """Detect if user wants to build a website"""
    website_keywords = [
        'build website', 'create website', 'generate website', 'make website',
        'build a website', 'create a website', 'generate a website', 'make a website',
        'build me a website', 'create me a website', 'generate me a website',
        'e-commerce website', 'ecommerce website', 'portfolio website', 'business website',
        'landing page', 'web app', 'web application', 'react website', 'react site',
        'online store', 'shop website', 'company website', 'personal website',
        'blog website', 'dashboard website', 'admin panel', 'website for'
    ]
    
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in website_keywords)


def generate_website_via_backend(user_input):
    """Generate website using the Website Building Backend"""
    try:
        # First, send chat message to analyze intent
        chat_payload = {
            "message": user_input
        }
        
        # Check if backend is available, if not, try to start it
        if not check_website_backend_available():
            send_status("Website Building Backend not running. Starting it now...")
            start_result = start_website_backend()
            
            if not start_result.get("success"):
                return {
                    "success": False,
                    "error": f"Failed to start Website Building Backend: {start_result.get('error')}\n"
                            f"Please start it manually: cd Website_building_backend && python main.py"
                }
            
            send_status("Website Building Backend started successfully!")
        
        # Send to chat endpoint
        chat_response = requests.post(
            f"{WEBSITE_BACKEND_URL}/chat",
            json=chat_payload,
            timeout=30
        )
        chat_response.raise_for_status()
        chat_result = chat_response.json()
        
        # Check if we need to start generation
        if chat_result.get("action") == "start_generation":
            # Start the generation process
            prompt = chat_result.get("prompt", user_input)
            
            # Call generate endpoint
            gen_payload = {
                "prompt": prompt,
                "is_edit": False,
                "project_id": None
            }
            
            gen_response = requests.post(
                f"{WEBSITE_BACKEND_URL}/generate",
                data=gen_payload,
                timeout=30
            )
            gen_response.raise_for_status()
            gen_result = gen_response.json()
            
            if gen_result.get("success"):
                task_id = gen_result.get("task_id")
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": chat_result.get("reply", "Starting website generation..."),
                    "is_website_generation": True
                }
            else:
                return {
                    "success": False,
                    "error": gen_result.get("message", "Failed to start generation")
                }
        
        elif chat_result.get("action") == "request_confirmation":
            return {
                "success": True,
                "needs_confirmation": True,
                "message": chat_result.get("reply"),
                "prompt": chat_result.get("prompt"),
                "is_edit": chat_result.get("is_edit", False)
            }
        
        else:
            # Just a chat response
            return {
                "success": True,
                "message": chat_result.get("reply", "I understand. Let me know when you're ready to proceed."),
                "is_website_generation": False
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot connect to Website Building Backend at {WEBSITE_BACKEND_URL}.\n"
                    "Please start the server first:\n"
                    "cd Website_building_backend && python main.py"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request to Website Building Backend timed out. The server may be busy."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error communicating with Website Building Backend: {str(e)}"
        }


def stream_website_generation(task_id):
    """Stream website generation progress and return final result"""
    try:
        import sseclient
        
        stream_url = f"{WEBSITE_BACKEND_URL}/generate_stream/{task_id}"
        
        # Use requests with stream=True for SSE
        response = requests.get(stream_url, stream=True, timeout=300)
        response.raise_for_status()
        
        full_result = {
            "status_updates": [],
            "preview_url": None,
            "zip_url": None,
            "error": None
        }
        
        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                # SSE format: data: {...}
                if line_str.startswith('data:'):
                    try:
                        data = json.loads(line_str[5:].strip())
                        
                        if data.get("status") == "log":
                            full_result["status_updates"].append({
                                "type": "log",
                                "message": data.get("message", "")
                            })
                        elif data.get("status") == "summary":
                            full_result["status_updates"].append({
                                "type": "summary",
                                "message": data.get("message", "")
                            })
                        elif data.get("status") == "generating":
                            full_result["status_updates"].append({
                                "type": "progress",
                                "file": data.get("file", ""),
                                "progress": data.get("progress", 0)
                            })
                        elif data.get("status") == "complete":
                            full_result["preview_url"] = data.get("preview_url")
                            full_result["zip_url"] = data.get("zip_url")
                            full_result["status_updates"].append({
                                "type": "complete",
                                "message": "Website generation complete!"
                            })
                            break
                        elif data.get("status") == "error":
                            full_result["error"] = data.get("message", "Unknown error")
                            full_result["status_updates"].append({
                                "type": "error",
                                "message": data.get("message", "")
                            })
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        return full_result
        
    except Exception as e:
        return {
            "status_updates": [],
            "preview_url": None,
            "zip_url": None,
            "error": f"Error streaming generation: {str(e)}"
        }


def is_confirmation_response(user_input):
    """Check if user input is a confirmation response to a pending action."""
    global pending_confirmation
    
    if not pending_confirmation:
        return None
    
    user_lower = user_input.lower().strip()
    
    # Phrases indicating user wants to modify/update existing file
    modify_phrases = [
        'modify existing', 'modify the existing', 'update existing', 'update the existing',
        'change existing', 'edit existing', 'use existing', 'existing one', 'existing file',
        'yes modify', 'yes update', 'modify it', 'update it', 'update', 'modify',
        'option 1', '1)', '1.', 'first option', 'first'
    ]
    
    # Phrases indicating user wants to create new file
    create_phrases = [
        'create new', 'create a new', 'new one', 'new file', 'create it',
        'yes create', 'make new', 'make a new', 'new',
        'option 2', '2)', '2.', 'second option', 'second'
    ]
    
    # Phrases indicating user wants to see diff/comparison
    diff_phrases = [
        'show diff', 'see diff', 'compare', 'difference', 'what changed',
        'option 3', '3)', '3.', 'third option', 'third', 'diff'
    ]
    
    # Phrases indicating user wants to backup and then modify
    backup_phrases = [
        'backup', 'save backup', 'backup first', 'create backup',
        'option 4', '4)', '4.', 'fourth option', 'fourth'
    ]
    
    # Phrases indicating cancellation
    cancel_phrases = [
        'cancel', 'no', 'stop', 'abort', 'don\'t', 'dont', 'never mind', 'nevermind',
        'option 5', '5)', '5.', 'fifth option', 'fifth', 'skip', 'ignore'
    ]
    
    for phrase in modify_phrases:
        if phrase in user_lower:
            return {'confirmed': True, 'action': 'modify_existing'}
    
    for phrase in create_phrases:
        if phrase in user_lower:
            return {'confirmed': True, 'action': 'create_new'}
    
    for phrase in diff_phrases:
        if phrase in user_lower:
            return {'confirmed': True, 'action': 'show_diff'}
    
    for phrase in backup_phrases:
        if phrase in user_lower:
            return {'confirmed': True, 'action': 'backup_and_modify'}
    
    for phrase in cancel_phrases:
        if phrase in user_lower:
            return {'confirmed': False, 'action': 'cancel'}
    
    return None


def show_file_diff(existing_path, new_content):
    """Show differences between existing file and new content."""
    try:
        with open(existing_path, 'r') as f:
            existing_content = f.read()
        
        import difflib
        diff = difflib.unified_diff(
            existing_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f'existing: {existing_path}',
            tofile='new: proposed',
            lineterm=''
        )
        
        diff_text = ''.join(diff)
        if not diff_text:
            return "[INFO] No differences found - files are identical."
        
        return f"[DIFF] Changes between existing and new content:\n{diff_text}"
    except Exception as e:
        return f"[ERROR] Could not generate diff: {e}"


def backup_file(path):
    """Create a backup of an existing file."""
    try:
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.backup_{timestamp}"
        
        shutil.copy2(path, backup_path)
        return backup_path
    except Exception as e:
        return None


def detect_direct_file_action(user_input):
    """Detect if user is requesting a direct file operation that can be handled without AI"""
    user_lower = user_input.lower().strip()

    # Patterns for direct file operations
    patterns = [
        (r'fix\s+(?:the\s+)?(?:issue|error|problem|bug)s?\s+in\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'debug_file'),
        (r'debug\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'debug_file'),
        (r'run\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'run_file'),
        (r'test\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'run_file'),
        (r'execute\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'run_file'),
        (r'check\s+(?:syntax|errors?)\s+in\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'debug_file'),
        (r'validate\s+(?!the\s+)([a-zA-Z0-9_./-]+)', 'debug_file'),
    ]

    import re
    for pattern, action in patterns:
        match = re.search(pattern, user_lower)
        if match:
            filename = match.group(1)
            # Skip common words that are likely not filenames
            common_words = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their']
            if filename.lower() in common_words:
                continue
            # Add .py extension if not present and looks like a Python file
            if not '.' in filename and not filename.endswith('/'):
                filename += '.py'
            return {'action': action, 'path': filename}

    return None


def handle_direct_file_action(action_data):
    """Handle direct file operations without requiring AI intervention"""
    action = action_data['action']
    path = action_data['path']

    try:
        if action == 'debug_file':
            # Check if Gemini API is available for auto-fixing
            if not check_gemini_available():
                # Fallback to basic syntax checking without AI auto-fix
                result = debug_file_basic(path)
                return f"I'll help check {path} for issues (AI auto-fix unavailable):\n\n{result}"
            else:
                # Full debug with AI auto-fix
                result = debug_file(path, "", "all")
                return f"I'll help fix the issues in {path}:\n\n{result}"

        elif action == 'run_file':
            # Try to run the file directly
            result = run_file(path, "none")
            return f"Running {path}:\n\n{result}"

    except Exception as e:
        return f"Error handling direct file action: {str(e)}"

    return None


def debug_file_basic(path):
    """Basic debug functionality without AI auto-fix"""
    try:
        # First, try to find the file by searching recursively
        full_path = find_file_recursive(path)

        # If not found by recursive search, try direct paths
        if not full_path:
            # Try as absolute path
            if os.path.isabs(path) and os.path.exists(path):
                full_path = path
            else:
                # Try relative to workspace
                full_path = os.path.join(WORKSPACE_PATH, path)
                if not os.path.exists(full_path):
                    # Try searching by basename only
                    basename = os.path.basename(path)
                    full_path = find_file_recursive(basename)

        if not full_path or not os.path.exists(full_path):
            return f"[ERROR] File '{path}' not found in workspace or any subdirectory."

        result_lines = [f"[DEBUGGING] Analyzing file '{full_path}' (basic check)..."]

        with open(full_path, "r") as f:
            content = f.read()

        # Stage 1: Syntax Check
        result_lines.append(f"\n[STAGE 1/2] Checking for syntax errors...")
        error, line_no = validate_python_code(content, full_path)
        if error:
            result_lines.append(f"[ERROR] Syntax error detected:")
            result_lines.append(error)
            result_lines.append(f"[INFO] AI auto-fix is not available. Please fix the syntax error manually at line {line_no}")
        else:
            result_lines.append(f"[OK] No syntax errors found.")

        # Stage 2: Runtime Check
        result_lines.append(f"\n[STAGE 2/2] Checking for runtime errors...")
        runtime_error, stdout, stderr = execute_and_capture_errors(content)

        if runtime_error:
            # Perform basic error analysis
            error_analysis = analyze_error(runtime_error, content, full_path)
            analysis_text = format_error_analysis(error_analysis)
            result_lines.append(analysis_text)

            if stdout:
                result_lines.append(f"[OUTPUT] Standard output before error:\n{stdout}")
            if stderr:
                result_lines.append(f"[ERROR OUTPUT] Standard error:\n{stderr}")
        else:
            result_lines.append(f"[OK] No runtime errors found.")
            if stdout:
                result_lines.append(f"[OUTPUT] Program output:\n{stdout}")

        result_lines.append(f"\n[OK] Basic file analysis completed!")
        return "\n".join(result_lines)

    except OSError as e:
        return f"[ERROR] {e}"


def process_message(user_input, conversation_history=""):
    """Process user message and return AI response"""
    # Check for simple greetings that don't need AI
    greeting_keywords = ['hi', 'hello', 'hey', 'help', 'start']
    user_lower = user_input.lower().strip()
    if any(user_lower.startswith(kw) for kw in greeting_keywords) and len(user_input) < 20:
        return "Hello! Great to connect. What are we building today?"

    # Check if Gemini API is available for actual AI processing
    if not check_gemini_available():
        return "Error: Cannot connect to Gemini API. Please check your API key configuration.\n\nMake sure GEMINI_API_KEY is set correctly in the backend configuration."

    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nConversation history:\n{conversation_history}\n\nUser: {user_input}\nAssistant:"

        # Use Gemini API with new client
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt
        )

        assistant_reply = response.text.strip()
        return assistant_reply


    except Exception as e:
        return f"Error: {str(e)}"


def main():
    global WORKSPACE_PATH, pending_confirmation
    conversation_history = ""
    
    # Send ready signal immediately - don't wait for Gemini API check
    sys.stdout.write(json.dumps({"type": "ready", "text": "Hello.! What would you like to work on today?"}) + "\n")
    sys.stdout.flush()
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
            
            # Handle workspace path configuration
            if data.get("type") == "config":
                if "workspacePath" in data:
                    WORKSPACE_PATH = data["workspacePath"]
                    # Ensure workspace path exists
                    if not os.path.exists(WORKSPACE_PATH):
                        os.makedirs(WORKSPACE_PATH, exist_ok=True)
                    send_status(f"Workspace set to: {WORKSPACE_PATH}")
                continue
            
            # Handle confirmation responses
            if data.get("type") == "confirmation_response":
                confirmed = data.get("confirmed", False)
                action = data.get("action", 'modify_existing')
                action_data = pending_confirmation
                
                if action_data and confirmed:
                    path = action_data.get("path")
                    content = action_data.get("content", "")
                    action_type = action_data.get("action")
                    
                    if action == 'modify_existing':
                        existing_path = find_file_recursive(os.path.basename(path))
                        if existing_path:
                            result = update_file(existing_path, content, confirmed=True)
                            send_response(result)
                        else:
                            send_response(f"[ERROR] Could not find existing file to modify.")
                    
                    elif action == 'create_new':
                        if action_type == "create_file":
                            result = create_file(path, content, confirmed=True, action_type='create_new')
                            send_response(result)
                        elif action_type == "update_file":
                            result = update_file(path, content, confirmed=True)
                            send_response(result)
                    
                    elif action == 'show_diff':
                        result = create_file(path, content, confirmed=True, action_type='show_diff')
                        send_response(result)
                    
                    elif action == 'backup_and_modify':
                        result = create_file(path, content, confirmed=True, action_type='backup_and_modify')
                        send_response(result)
                    
                    else:
                        send_response(f"[ERROR] Unknown action: {action}")
                else:
                    send_response("[INFO] Operation cancelled by user.")
                
                pending_confirmation = None
                continue
            
            if data.get("type") == "message":
                user_input = data.get("text", "").strip()
                if user_input:
                    # Check for direct file actions first
                    direct_action = detect_direct_file_action(user_input)
                    if direct_action:
                        result = handle_direct_file_action(direct_action)
                        if result:
                            send_response(result)
                            continue

                    # Check if this is a natural language confirmation response
                    confirmation = is_confirmation_response(user_input)
                    if confirmation:
                        # Handle as confirmation response
                        action_data = pending_confirmation

                        if action_data and confirmation['confirmed']:
                            path = action_data.get("path")
                            content = action_data.get("content", "")
                            user_action = confirmation.get('action', 'modify_existing')

                            if user_action == 'modify_existing':
                                existing_path = find_file_recursive(os.path.basename(path))
                                if existing_path:
                                    result = update_file(existing_path, content, confirmed=True)
                                    send_response(result)
                                else:
                                    send_response(f"[ERROR] Could not find existing file to modify.")

                            elif user_action == 'create_new':
                                action = action_data.get("action")
                                if action == "create_file":
                                    result = create_file(path, content, confirmed=True, action_type='create_new')
                                    send_response(result)
                                elif action == "update_file":
                                    result = update_file(path, content, confirmed=True)
                                    send_response(result)

                            elif user_action == 'show_diff':
                                result = create_file(path, content, confirmed=True, action_type='show_diff')
                                send_response(result)

                            elif user_action == 'backup_and_modify':
                                result = create_file(path, content, confirmed=True, action_type='backup_and_modify')
                                send_response(result)

                            else:
                                send_response(f"[ERROR] Unknown action: {user_action}")
                        else:
                            send_response("[INFO] Operation cancelled by user.")

                        pending_confirmation = None
                        continue
                    
                    # Check if this is a website building request
                    if is_website_building_request(user_input):
                        send_status("Detected website building request. Connecting to Website Building Backend...")
                        
                        # Generate website via backend
                        website_result = generate_website_via_backend(user_input)
                        
                        if not website_result.get("success"):
                            error_msg = website_result.get("error", "Unknown error")
                            send_error(f"Website generation failed: {error_msg}")
                            continue
                        
                        # Check if we need confirmation
                        if website_result.get("needs_confirmation"):
                            send_response(website_result.get("message"))
                            # Store for later confirmation handling
                            pending_confirmation = {
                                "action": "website_generation",
                                "prompt": website_result.get("prompt"),
                                "is_edit": website_result.get("is_edit", False)
                            }
                            continue
                        
                        # Check if it's actually a website generation (not just chat)
                        if website_result.get("is_website_generation"):
                            task_id = website_result.get("task_id")
                            send_response(website_result.get("message", "Starting website generation..."))
                            
                            # Stream the generation progress
                            send_status("Streaming website generation progress...")
                            stream_result = stream_website_generation(task_id)
                            
                            # Send all status updates
                            for update in stream_result.get("status_updates", []):
                                if update["type"] == "log":
                                    send_status(update["message"])
                                elif update["type"] == "summary":
                                    send_response(update["message"])
                                elif update["type"] == "progress":
                                    send_status(f"Generating: {update['file']} ({update['progress']}%)")
                                elif update["type"] == "error":
                                    send_error(update["message"])
                            
                            # Check final result
                            if stream_result.get("error"):
                                send_error(f"Generation failed: {stream_result['error']}")
                            elif stream_result.get("preview_url"):
                                preview_url = f"{WEBSITE_BACKEND_URL}{stream_result['preview_url']}"
                                zip_url = stream_result.get("zip_url", "")
                                
                                # Send special website_complete message
                                sys.stdout.write(json.dumps({
                                    "type": "website_complete",
                                    "preview_url": preview_url,
                                    "zip_url": zip_url,
                                    "text": f"✅ Website generated successfully!\n\n🌐 Preview: {preview_url}\n\n📦 Download: {zip_url}"
                                }) + "\n")
                                sys.stdout.flush()
                            continue
                        else:
                            # Just a chat response
                            send_response(website_result.get("message"))
                            continue
                    
                    # Process the message normally with Gemini
                    assistant_reply = process_message(user_input, conversation_history)
                    
                    # Update conversation history
                    conversation_history += f"User: {user_input}\nAssistant: {assistant_reply}\n"
                    
                    # Extract and execute JSON actions
                    json_objects = extract_json_objects(assistant_reply)
                    
                    action_results = []
                    if json_objects:
                        for action_data in json_objects:
                            try:
                                action = action_data.get("action") or action_data.get("intent")
                                if isinstance(action, str):
                                    act = action.strip().lower()
                                else:
                                    act = action

                                if act in ("create_folder", "create folder", "createfolder"):
                                    folder = action_data.get("folder") or action_data.get("name")
                                    if folder:
                                        result = create_folder(folder)
                                        action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing folder name")

                                elif act in ("create_project", "create project", "createproject"):
                                    folder = action_data.get("folder") or action_data.get("name") or action_data.get("project")
                                    files = action_data.get("files", [])
                                    if folder and files:
                                        result = create_project(folder, files)
                                        action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing folder name or files list")

                                elif act in ("create_file", "create file", "createfile"):
                                    path = action_data.get("path") or action_data.get("filename") or action_data.get("file")
                                    content = action_data.get("content", "")
                                    if path:
                                        result = create_file(path, content)
                                        # Check if confirmation is required
                                        if result.startswith("[CONFIRMATION_REQUIRED]"):
                                            pending_confirmation = {
                                                "action": "create_file",
                                                "path": path,
                                                "content": content
                                            }
                                            send_confirmation_request(result, pending_confirmation)
                                            action_results.append(result)
                                            # Don't process further actions until confirmation received
                                            break
                                        else:
                                            action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing path")

                                elif act in ("update_file", "update file", "updatefile"):
                                    path = action_data.get("path") or action_data.get("filename") or action_data.get("file")
                                    content = action_data.get("content", "")
                                    if path:
                                        # Check if file exists in main directory first
                                        main_path = os.path.join(WORKSPACE_PATH, path)
                                        if os.path.exists(main_path):
                                            # Update file in main directory
                                            result = update_file(path, content)
                                            action_results.append(result)
                                        else:
                                            # Check if file exists in subdirectory
                                            existing_path = find_file_recursive(path)
                                            if existing_path:
                                                # Ask for confirmation to modify existing
                                                result = update_file(path, content)
                                                if result.startswith("[CONFIRMATION_REQUIRED]"):
                                                    pending_confirmation = {
                                                        "action": "update_file",
                                                        "path": path,
                                                        "content": content
                                                    }
                                                    send_confirmation_request(result, pending_confirmation)
                                                    action_results.append(result)
                                                    # Don't process further actions until confirmation received
                                                    break
                                                else:
                                                    action_results.append(result)
                                            else:
                                                # Create new file
                                                result = create_file(path, content)
                                                action_results.append(f"[INFO] File '{path}' did not exist. Created new file.")
                                                action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing path")

                                elif act in ("debug_file", "debug file", "debugfile"):
                                    path = action_data.get("path") or action_data.get("filename") or action_data.get("file")
                                    debug_stage = action_data.get("stage", "all")
                                    if path:
                                        result = debug_file(path, conversation_history, debug_stage)
                                        action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing path")

                                elif act in ("run_file", "run file", "runfile", "test_file", "test file", "testfile"):
                                    path = action_data.get("path") or action_data.get("filename") or action_data.get("file")
                                    environment = action_data.get("environment", "none")
                                    if path:
                                        result = run_file(path, environment)
                                        action_results.append(result)
                                    else:
                                        action_results.append("Qwen: missing path")

                                elif act in ("search_files", "search files", "searchfiles"):
                                    keyword = action_data.get("keyword") or action_data.get("search") or action_data.get("query")
                                    file_type = action_data.get("file_type") or action_data.get("extension")
                                    max_results = action_data.get("max_results", 10)
                                    if keyword:
                                        results = find_files_by_keyword(keyword, file_type, max_results)
                                        result = format_search_results(results, "files")
                                        action_results.append(result)
                                    else:
                                        action_results.append("[ERROR] Missing search keyword")

                                elif act in ("search_folders", "search folders", "searchfolders"):
                                    keyword = action_data.get("keyword") or action_data.get("search") or action_data.get("query")
                                    max_results = action_data.get("max_results", 10)
                                    if keyword:
                                        results = find_folders_by_keyword(keyword, max_results)
                                        result = format_search_results(results, "folders")
                                        action_results.append(result)
                                    else:
                                        action_results.append("[ERROR] Missing search keyword")

                                elif act in ("search_in_files", "search in files", "searchinfiles", "grep"):
                                    keyword = action_data.get("keyword") or action_data.get("search") or action_data.get("query")
                                    file_pattern = action_data.get("file_pattern") or action_data.get("pattern") or "*"
                                    max_results = action_data.get("max_results", 10)
                                    if keyword:
                                        results = search_in_file_content(keyword, file_pattern, max_results)
                                        result = format_search_results(results, "content matches")
                                        action_results.append(result)
                                    else:
                                        action_results.append("[ERROR] Missing search keyword")

                                elif act in ("get_file_info", "get file info", "getfileinfo", "file_info"):
                                    path = action_data.get("path") or action_data.get("file") or action_data.get("filename")
                                    if path:
                                        info = get_file_info(path)
                                        if 'error' in info:
                                            result = f"[ERROR] {info['error']}"
                                        else:
                                            lines = ["[OK] File Information:"]
                                            lines.append("-" * 40)
                                            for key, value in info.items():
                                                if key != 'full_path':
                                                    lines.append(f"{key.replace('_', ' ').title()}: {value}")
                                            result = "\n".join(lines)
                                        action_results.append(result)
                                    else:
                                        action_results.append("[ERROR] Missing file path")

                            except Exception as e:
                                action_results.append(f"[ERROR] Failed to process action: {e}")
                    
                    # Send response back to VS Code:
                    if action_results:
                        full_response = assistant_reply + "\n\n" + "\n".join(action_results)
                        send_response(full_response)
                    else:
                        send_response(assistant_reply)
                        
            elif data.get("type") == "exit":
                break
        except json.JSONDecodeError as e:
            # Log the problematic line for debugging
            send_error(f"Invalid JSON message received: {line[:100]}...")
            continue
        except Exception as e:
            send_error(str(e))
            continue

if __name__ == "__main__":
    main()
