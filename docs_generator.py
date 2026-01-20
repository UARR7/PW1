

import os
import git
import shutil
from datetime import datetime
import logging
from pathlib import Path
import re
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pygments.formatters import HtmlFormatter
from config import Config
import requests
import json
import time
from typing import Dict, List, Optional, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

class HybridDocEnhancer:
    """Hybrid documentation enhancer using Microsoft CodeBERT + Google Gemini"""
    
    def __init__(self, gemini_api_key: str = None):
        # CodeBERT setup for one-sentence generation
        self.codebert_model_name = "microsoft/codebert-base"
        self.codebert_api_url = f"https://api-inference.huggingface.co/models/{self.codebert_model_name}"
        
        # Fallback to GPT-2 for text generation
        self.text_model_name = "gpt2"
        self.text_api_url = f"https://api-inference.huggingface.co/models/{self.text_model_name}"
        
        # Gemini setup for detailed explanations
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            self.gemini_available = True
            logger.info("Gemini API initialized successfully")
        else:
            self.gemini_available = False
            logger.warning("Gemini API key not provided. Detailed explanations will be disabled.")
        
        # Rate limiting
        self.last_hf_request_time = 0
        self.last_gemini_request_time = 0
        self.min_hf_request_interval = 1.0  # 1 second between HuggingFace requests
        self.min_gemini_request_interval = 2.0  # 2 seconds between Gemini requests
        
    def _make_hf_request(self, api_url: str, payload: Dict[str, Any], max_retries: int = 3) -> Optional[Dict]:
        """Make a request to Hugging Face API with error handling and rate limiting"""
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_hf_request_time
        if time_since_last_request < self.min_hf_request_interval:
            time.sleep(self.min_hf_request_interval - time_since_last_request)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                self.last_hf_request_time = time.time()
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    logger.info(f"HF Model loading, waiting... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5 * (attempt + 1))  # Progressive backoff
                    continue
                else:
                    logger.warning(f"HF API request failed with status {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"HF Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    
        return None
    
    def _make_gemini_request(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make a request to Gemini API with error handling and rate limiting"""
        if not self.gemini_available:
            return None
            
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_gemini_request_time
        if time_since_last_request < self.min_gemini_request_interval:
            time.sleep(self.min_gemini_request_interval - time_since_last_request)
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=500,
                        temperature=0.3,
                    )
                )
                self.last_gemini_request_time = time.time()
                
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                logger.warning(f"Gemini request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))  # Progressive backoff
                    
        return None
    
    def generate_function_description(self, function_name: str, function_code: str, docstring: str = "", language: str = "python") -> Dict[str, str]:
        """Generate both one-sentence and detailed descriptions for a function"""
        result = {
            'one_liner': '',
            'detailed_explanation': ''
        }
        
        # Step 1: Generate one-liner using CodeBERT/GPT-2
        if docstring.strip():
            prompt = f"Improve this {language} function documentation:\nFunction: {function_name}\nExisting docs: {docstring[:200]}..."
        else:
            prompt = f"Describe this {language} function in one clear sentence:\n{function_name}:\n{function_code[:300]}..."
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 100,
                "temperature": 0.3,
                "do_sample": True,
                "pad_token_id": 50256
            }
        }
        
        hf_result = self._make_hf_request(self.text_api_url, payload)
        if hf_result and isinstance(hf_result, list) and len(hf_result) > 0:
            generated_text = hf_result[0].get('generated_text', '')
            if prompt in generated_text:
                description = generated_text[len(prompt):].strip()
                description = description.split('\n')[0]  # Take first line
                description = description.strip('.,!?')  # Remove trailing punctuation
                if description and len(description) > 10:
                    result['one_liner'] = description
        
        # Fallback for one-liner
        if not result['one_liner']:
            result['one_liner'] = self._generate_simple_description(function_name, language)
        
        # Step 2: Generate detailed explanation using Gemini
        if self.gemini_available and result['one_liner']:
            gemini_prompt = f"""You are a technical documentation expert. Based on this function information, provide a detailed explanation (2-3 sentences) that expands on the one-sentence summary.

Function Name: {function_name}
Language: {language}
One-line Summary: {result['one_liner']}
Function Code: {function_code[:500]}
Existing Documentation: {docstring[:200] if docstring else 'None'}

Provide a detailed explanation that includes:
1. What the function does in more detail
2. Key parameters or inputs (if visible)
3. Expected behavior or use cases
4. Any important technical details

Keep it concise but informative (2-3 sentences maximum):"""

            detailed_explanation = self._make_gemini_request(gemini_prompt)
            if detailed_explanation:
                result['detailed_explanation'] = detailed_explanation
        
        return result
    
    def generate_class_description(self, class_name: str, class_code: str, docstring: str = "", language: str = "python") -> Dict[str, str]:
        """Generate both one-sentence and detailed descriptions for a class"""
        result = {
            'one_liner': '',
            'detailed_explanation': ''
        }
        
        # Step 1: Generate one-liner using CodeBERT/GPT-2
        if docstring.strip():
            prompt = f"Summarize this {language} class:\nClass: {class_name}\nDocs: {docstring[:200]}..."
        else:
            prompt = f"What does this {language} class do?\nclass {class_name}:\n{class_code[:300]}..."
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 80,
                "temperature": 0.3,
                "do_sample": True,
                "pad_token_id": 50256
            }
        }
        
        hf_result = self._make_hf_request(self.text_api_url, payload)
        if hf_result and isinstance(hf_result, list) and len(hf_result) > 0:
            generated_text = hf_result[0].get('generated_text', '')
            if prompt in generated_text:
                description = generated_text[len(prompt):].strip()
                description = description.split('\n')[0]
                description = description.strip('.,!?')
                if description and len(description) > 10:
                    result['one_liner'] = description
        
        # Fallback for one-liner
        if not result['one_liner']:
            result['one_liner'] = self._generate_simple_description(class_name, "class", is_class=True)
        
        # Step 2: Generate detailed explanation using Gemini
        if self.gemini_available and result['one_liner']:
            gemini_prompt = f"""You are a technical documentation expert. Based on this class information, provide a detailed explanation (2-3 sentences) that expands on the one-sentence summary.

Class Name: {class_name}
Language: {language}
One-line Summary: {result['one_liner']}
Class Code: {class_code[:500]}
Existing Documentation: {docstring[:200] if docstring else 'None'}

Provide a detailed explanation that includes:
1. The class's main purpose and responsibility
2. Key attributes or methods (if visible)
3. How it fits into the overall architecture
4. Common usage patterns or design patterns implemented

Keep it concise but informative (2-3 sentences maximum):"""

            detailed_explanation = self._make_gemini_request(gemini_prompt)
            if detailed_explanation:
                result['detailed_explanation'] = detailed_explanation
        
        return result
    
    def generate_file_summary(self, file_name: str, file_content: str, functions: List[Dict], classes: List[Dict], language: str = "unknown") -> Dict[str, str]:
        """Generate both basic and detailed summaries for the entire file"""
        result = {
            'basic_summary': '',
            'detailed_summary': ''
        }
        
        # Basic summary (existing logic)
        summary_parts = []
        
        if classes:
            summary_parts.append(f"Contains {len(classes)} class(es)")
        if functions:
            summary_parts.append(f"Contains {len(functions)} function(s)")
        
        file_purpose = self._infer_file_purpose(file_name, language)
        if file_purpose:
            summary_parts.insert(0, file_purpose)
        
        if summary_parts:
            result['basic_summary'] = ". ".join(summary_parts) + "."
        else:
            language_name = self._get_language_name(language)
            result['basic_summary'] = f"{language_name} module containing code for {file_name.replace('_', ' ').replace('.', ' ')}."
        
        # Detailed summary using Gemini
        if self.gemini_available:
            class_names = [cls['name'] for cls in classes]
            function_names = [func['name'] for func in functions]
            
            gemini_prompt = f"""You are a technical documentation expert. Analyze this code file and provide a comprehensive summary (3-4 sentences).

File Name: {file_name}
Language: {self._get_language_name(language)}
Basic Summary: {result['basic_summary']}
Classes Found: {', '.join(class_names) if class_names else 'None'}
Functions Found: {', '.join(function_names[:10]) if function_names else 'None'}  
File Content Preview: {file_content[:800]}

Provide a detailed summary that explains:
1. The file's main purpose and role in the project
2. Key functionality and architecture patterns
3. How the classes and functions work together
4. Important dependencies or integrations

Keep it informative but concise (3-4 sentences maximum):"""

            detailed_summary = self._make_gemini_request(gemini_prompt)
            if detailed_summary:
                result['detailed_summary'] = detailed_summary
        
        return result
    
    def _generate_simple_description(self, name: str, language: str = "python", is_class: bool = False) -> str:
        """Generate a simple description based on naming patterns"""
        # Convert camelCase and snake_case to words
        words = re.sub(r'([A-Z])', r' \1', name).replace('_', ' ').strip().lower()
        
        if is_class:
            if 'manager' in words:
                return f"Manages {words.replace('manager', '').strip()}"
            elif 'handler' in words:
                return f"Handles {words.replace('handler', '').strip()}"
            elif 'processor' in words:
                return f"Processes {words.replace('processor', '').strip()}"
            elif 'controller' in words:
                return f"Controls {words.replace('controller', '').strip()}"
            elif 'service' in words:
                return f"Provides services for {words.replace('service', '').strip()}"
            else:
                return f"A class that handles {words}"
        else:
            if words.startswith('get'):
                return f"Retrieves {words[3:].strip()}"
            elif words.startswith('set'):
                return f"Sets {words[3:].strip()}"
            elif words.startswith('create'):
                return f"Creates {words[6:].strip()}"
            elif words.startswith('delete'):
                return f"Deletes {words[6:].strip()}"
            elif words.startswith('update'):
                return f"Updates {words[6:].strip()}"
            elif words.startswith('process'):
                return f"Processes {words[7:].strip()}"
            elif words.startswith('generate'):
                return f"Generates {words[8:].strip()}"
            elif words.startswith('init') or words.startswith('initialize'):
                return f"Initializes {words[4:].strip() if words.startswith('init') else words[10:].strip()}"
            else:
                return f"Function that handles {words}"
    
    def _infer_file_purpose(self, filename: str, language: str) -> str:
        """Infer file purpose from filename"""
        base_name = filename.lower()
        for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go']:
            base_name = base_name.replace(ext, '')
        
        purpose_map = {
            'main': "Main application entry point",
            'app': "Application core functionality",
            'config': "Configuration and settings",
            'models': "Data models and structures",
            'views': "User interface views",
            'controllers': "Application controllers",
            'utils': "Utility functions and helpers",
            'helpers': "Helper functions and utilities",
            'tests': "Test cases and test utilities",
            'test': "Test cases and test utilities",
            'api': "API endpoints and handlers",
            'database': "Database operations and connections",
            'db': "Database operations and connections",
            'auth': "Authentication and authorization",
            'middleware': "Middleware components",
            'routes': "URL routing and endpoints",
            'server': "Server implementation and setup",
            'client': "Client-side implementation",
            'service': "Service layer implementation",
            'repository': "Data repository pattern implementation",
            'component': "Reusable component implementation",
            'interface': "Interface definitions",
            'abstract': "Abstract classes and interfaces",
            'factory': "Factory pattern implementation",
            'builder': "Builder pattern implementation",
            'adapter': "Adapter pattern implementation",
            'decorator': "Decorator pattern implementation"
        }
        
        for keyword, description in purpose_map.items():
            if keyword in base_name:
                return description
        
        return ""
    
    def _get_language_name(self, extension: str) -> str:
        """Get human-readable language name from extension"""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go'
        }
        return language_map.get(extension, 'Unknown')


class DocsGenerator:
    def __init__(self, gemini_api_key: str = None):
        self.config = Config()
        self.output_dir = self.config.DOCS_OUTPUT_DIR
        self.temp_dir = self.config.TEMP_DIR
        self.ai_enhancer = HybridDocEnhancer(gemini_api_key)
        
        # Multi-language support
        self.supported_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go']
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def generate_from_repo(self, repo_url, branch='main', use_ai_enhancement=True):
        """Generate documentation from a Git repository"""
        temp_repo_path = None
        try:
            # Clone repository
            logger.info(f"Cloning repository: {repo_url}")
            temp_repo_path = self._clone_repository(repo_url, branch)
            
            if not temp_repo_path:
                return False
            
            # Analyze repository structure
            repo_info = self._analyze_repository(temp_repo_path)
            
            # Store repo_info as instance attribute for XAI access
            self.repo_info = repo_info
            
            # Enhance with AI if enabled
            if use_ai_enhancement:
                logger.info("Enhancing documentation with Hybrid AI (CodeBERT + Gemini)...")
                self._enhance_with_hybrid_ai(repo_info)
            
            # Generate documentation
            self._generate_documentation(repo_info, temp_repo_path)
            
            logger.info("Documentation generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return False
        finally:
            # Cleanup
            if temp_repo_path and os.path.exists(temp_repo_path):
                shutil.rmtree(temp_repo_path, ignore_errors=True)
    
    def _enhance_with_hybrid_ai(self, repo_info):
        """Enhance documentation using hybrid AI approach"""
        try:
            total_files = len([f for f in repo_info['files'] if f['extension'] in self.supported_extensions])
            processed = 0
            
            for file_info in repo_info['files']:
                if file_info['extension'] in self.supported_extensions:
                    processed += 1
                    logger.info(f"Hybrid AI enhancing file {processed}/{total_files}: {file_info['name']}")
                    
                    # Generate file summary (basic + detailed)
                    functions = file_info.get('functions', [])
                    classes = file_info.get('classes', [])
                    language = file_info.get('language', 'unknown')
                    
                    file_summary = self.ai_enhancer.generate_file_summary(
                        file_info['name'], 
                        file_info.get('content', ''),
                        functions,
                        classes,
                        language
                    )
                    
                    file_info['ai_summary'] = file_summary['basic_summary']
                    file_info['ai_detailed_summary'] = file_summary.get('detailed_summary', '')
                    
                    # Enhance function descriptions
                    for func in functions:
                        func_desc = self.ai_enhancer.generate_function_description(
                            func['name'],
                            func.get('params', ''),
                            func.get('docstring', ''),
                            language
                        )
                        func['ai_description'] = func_desc['one_liner']
                        func['ai_detailed_description'] = func_desc.get('detailed_explanation', '')
                    
                    # Enhance class descriptions
                    for cls in classes:
                        class_desc = self.ai_enhancer.generate_class_description(
                            cls['name'],
                            '',  # We could extract class content if needed
                            cls.get('docstring', ''),
                            language
                        )
                        cls['ai_description'] = class_desc['one_liner']
                        cls['ai_detailed_description'] = class_desc.get('detailed_explanation', '')
                    
                    file_info['ai_enhanced'] = True
                    
                    # Small delay to be respectful to the APIs
                    time.sleep(0.5)
            
            repo_info['ai_enhanced'] = True
            repo_info['hybrid_ai_used'] = self.ai_enhancer.gemini_available
            logger.info("Hybrid AI enhancement completed")
            
        except Exception as e:
            logger.warning(f"Hybrid AI enhancement failed: {e}")
            repo_info['ai_enhanced'] = False
            repo_info['hybrid_ai_used'] = False

    # Include all the existing methods from the original class
    # (I'll include the key ones and indicate where the rest would go)
    
    def _clone_repository(self, repo_url, branch):
        """Clone repository to temporary directory"""
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            temp_path = os.path.join(self.temp_dir, f"{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Clone with timeout
            repo = git.Repo.clone_from(
                repo_url,
                temp_path,
                branch=branch,
                depth=1  # Shallow clone for efficiency
            )
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            return None

    def _analyze_repository(self, repo_path):
        """Analyze repository structure and extract information"""
        repo_info = {
            'name': os.path.basename(repo_path),
            'path': repo_path,
            'files': [],
            'structure': {},
            'readme': None,
            'languages': {},
            'ai_enhanced': False,
            'hybrid_ai_used': False
        }
        
        # Walk through repository
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git', 'target', 'build', 'dist', 'bin', 'obj']]
            
            rel_path = os.path.relpath(root, repo_path)
            if rel_path == '.':
                rel_path = ''
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, repo_path)
                
                # Check if it's a source file
                file_ext = Path(file).suffix.lower()
                if file_ext in self.supported_extensions or file.lower() in ['readme.md', 'readme.txt', 'readme']:
                    file_info = {
                        'name': file,
                        'path': file_path,
                        'rel_path': rel_file_path,
                        'extension': file_ext,
                        'size': os.path.getsize(file_path),
                        'ai_summary': '',
                        'ai_detailed_summary': '',
                        'ai_enhanced': False,
                        'language': self._get_language_from_extension(file_ext)
                    }
                    
                    # Extract file content for analysis
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            file_info['content'] = content
                            file_info['lines'] = len(content.split('\n'))
                            
                            # Extract functions/classes based on language
                            file_info['functions'], file_info['classes'] = self._extract_code_elements(content, file_ext)
                            
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        continue
                    
                    repo_info['files'].append(file_info)
                    
                    # Count languages
                    if file_ext:
                        repo_info['languages'][file_ext] = repo_info['languages'].get(file_ext, 0) + 1
                
                # Check for README
                if file.lower().startswith('readme'):
                    repo_info['readme'] = file_path
        
        return repo_info

    def _generate_functions_html(self, functions: List[Dict], language: str) -> str:
        """Generate HTML for functions documentation with hybrid AI descriptions"""
        html_parts = []
        language_info = self._get_language_info(f'.{language}')
        
        for func in functions:
            # One-liner description
            ai_desc = ''
            if func.get('ai_description'):
                ai_desc = f'<div class="ai-description"><strong>📝 Quick Summary:</strong> {func["ai_description"]}</div>'
            
            # Detailed explanation from Gemini
            detailed_desc = ''
            if func.get('ai_detailed_description'):
                detailed_desc = f'<div class="ai-detailed-description"><strong>🤖 Detailed Explanation:</strong> {func["ai_detailed_description"]}</div>'
            
            docstring_section = ''
            if func.get('docstring'):
                docstring_section = f'<div class="docstring"><strong>📖 Original Documentation:</strong><pre>{func["docstring"]}</pre></div>'
            
            html_parts.append(f'''
                <div class="function-item">
                    <div class="element-header">
                        <h4>{language_info["function_icon"]} {func['name']}({func['params']})</h4>
                        <span class="line-number">Line {func['line']}</span>
                    </div>
                    {ai_desc}
                    {detailed_desc}
                    {docstring_section}
                </div>
            ''')
        return ''.join(html_parts)
    
    def _generate_classes_html(self, classes: List[Dict], language: str) -> str:
        """Generate HTML for classes documentation with hybrid AI descriptions"""
        html_parts = []
        language_info = self._get_language_info(f'.{language}')
        
        for cls in classes:
            # One-liner description
            ai_desc = ''
            if cls.get('ai_description'):
                ai_desc = f'<div class="ai-description"><strong>📝 Quick Summary:</strong> {cls["ai_description"]}</div>'
            
            # Detailed explanation from Gemini
            detailed_desc = ''
            if cls.get('ai_detailed_description'):
                detailed_desc = f'<div class="ai-detailed-description"><strong>🤖 Detailed Explanation:</strong> {cls["ai_detailed_description"]}</div>'
            
            docstring_section = ''
            if cls.get('docstring'):
                docstring_section = f'<div class="docstring"><strong>📖 Original Documentation:</strong><pre>{cls["docstring"]}</pre></div>'
            
            html_parts.append(f'''
                <div class="class-item">
                    <div class="element-header">
                        <h4>{language_info["class_icon"]} {cls['name']}</h4>
                        <span class="line-number">Line {cls['line']}</span>
                    </div>
                    {ai_desc}
                    {detailed_desc}
                    {docstring_section}
                </div>
            ''')
        return ''.join(html_parts)

    def _get_language_from_extension(self, extension: str) -> str:
        """Get language identifier from file extension"""
        return extension.lstrip('.')
    
    def _extract_code_elements(self, content: str, extension: str) -> tuple:
        """Extract functions and classes from code based on language"""
        if extension == '.py':
            return self._extract_python_elements(content)
        elif extension in ['.js', '.ts']:
            return self._extract_javascript_elements(content)
        elif extension == '.java':
            return self._extract_java_elements(content)
        elif extension in ['.cpp', '.c', '.h']:
            return self._extract_c_cpp_elements(content)
        elif extension == '.cs':
            return self._extract_csharp_elements(content)
        elif extension == '.php':
            return self._extract_php_elements(content)
        elif extension == '.rb':
            return self._extract_ruby_elements(content)
        elif extension == '.go':
            return self._extract_go_elements(content)
        else:
            return [], []

    def _extract_python_elements(self, content: str) -> tuple:
        """Extract function and class definitions from Python code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract functions
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                func_match = re.match(r'^\s*def\s+(\w+)\s*\(([^)]*)\):', line)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    docstring = self._extract_python_docstring(lines, i + 1)
                    
                    functions.append({
                        'name': func_name,
                        'params': func_params,
                        'line': i + 1,
                        'docstring': docstring,
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract classes
            elif re.match(r'^\s*class\s+\w+', line):
                class_match = re.match(r'^\s*class\s+(\w+)\s*(?:\([^)]*\))?:', line)
                if class_match:
                    class_name = class_match.group(1)
                    docstring = self._extract_python_docstring(lines, i + 1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': docstring,
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_javascript_elements(self, content: str) -> tuple:
        """Extract function and class definitions from JavaScript/TypeScript code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract functions (function declarations and arrow functions)
            func_patterns = [
                r'^\s*function\s+(\w+)\s*\(([^)]*)\)',  # function name()
                r'^\s*const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>',  # const name = () =>
                r'^\s*(\w+)\s*:\s*function\s*\(([^)]*)\)',  # name: function()
                r'^\s*(\w+)\s*\(([^)]*)\)\s*{',  # name() { (method)
            ]
            
            for pattern in func_patterns:
                func_match = re.match(pattern, line)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2) if func_match.lastindex >= 2 else ''
                    
                    functions.append({
                        'name': func_name,
                        'params': func_params,
                        'line': i + 1,
                        'docstring': self._extract_js_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
                    break
            
            # Extract classes
            if re.match(r'^\s*class\s+\w+', line):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_js_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_java_elements(self, content: str) -> tuple:
        """Extract function and class definitions from Java code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract methods (public, private, protected)
            if re.search(r'\b(public|private|protected|static).*\w+\s*\([^)]*\)\s*{?', line) and 'class' not in line:
                method_match = re.search(r'\b(\w+)\s*\(([^)]*)\)', line)
                if method_match:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2)
                    
                    functions.append({
                        'name': method_name,
                        'params': method_params,
                        'line': i + 1,
                        'docstring': self._extract_java_javadoc(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract classes
            elif re.match(r'^\s*(public|private|protected)?\s*class\s+\w+', line):
                class_match = re.search(r'class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_java_javadoc(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_c_cpp_elements(self, content: str) -> tuple:
        """Extract function definitions from C/C++ code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract functions (basic pattern)
            if re.match(r'^[^/]*\w+\s+\w+\s*\([^)]*\)\s*{?', line) and not line.strip().startswith('//'):
                func_match = re.search(r'(\w+)\s*\(([^)]*)\)', line)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    
                    # Skip common keywords
                    if func_name not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                        functions.append({
                            'name': func_name,
                            'params': func_params,
                            'line': i + 1,
                            'docstring': self._extract_c_comment(lines, i),
                            'ai_description': '',
                            'ai_detailed_description': ''
                        })
            
            # Extract classes (C++)
            elif re.match(r'^\s*class\s+\w+', line):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_c_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_csharp_elements(self, content: str) -> tuple:
        """Extract function and class definitions from C# code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract methods
            if re.search(r'\b(public|private|protected|internal|static).*\w+\s*\([^)]*\)\s*{?', line) and 'class' not in line:
                method_match = re.search(r'\b(\w+)\s*\(([^)]*)\)', line)
                if method_match:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2)
                    
                    functions.append({
                        'name': method_name,
                        'params': method_params,
                        'line': i + 1,
                        'docstring': self._extract_csharp_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract classes
            elif re.match(r'^\s*(public|private|protected|internal)?\s*class\s+\w+', line):
                class_match = re.search(r'class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_csharp_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_php_elements(self, content: str) -> tuple:
        """Extract function and class definitions from PHP code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract functions
            if re.match(r'^\s*function\s+\w+\s*\(', line):
                func_match = re.match(r'^\s*function\s+(\w+)\s*\(([^)]*)\)', line)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    
                    functions.append({
                        'name': func_name,
                        'params': func_params,
                        'line': i + 1,
                        'docstring': self._extract_php_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract classes
            elif re.match(r'^\s*class\s+\w+', line):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_php_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_ruby_elements(self, content: str) -> tuple:
        """Extract function and class definitions from Ruby code"""
        functions = []
        classes = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract methods
            if re.match(r'^\s*def\s+\w+', line):
                method_match = re.match(r'^\s*def\s+(\w+)(?:\(([^)]*)\))?', line)
                if method_match:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2) if method_match.group(2) else ''
                    
                    functions.append({
                        'name': method_name,
                        'params': method_params,
                        'line': i + 1,
                        'docstring': self._extract_ruby_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract classes
            elif re.match(r'^\s*class\s+\w+', line):
                class_match = re.match(r'^\s*class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    
                    classes.append({
                        'name': class_name,
                        'line': i + 1,
                        'docstring': self._extract_ruby_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_go_elements(self, content: str) -> tuple:
        """Extract function definitions from Go code"""
        functions = []
        classes = []  # Go doesn't have classes, but we'll extract structs
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Extract functions
            if re.match(r'^\s*func\s+', line):
                func_match = re.match(r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)', line)
                if func_match:
                    func_name = func_match.group(1)
                    func_params = func_match.group(2)
                    
                    functions.append({
                        'name': func_name,
                        'params': func_params,
                        'line': i + 1,
                        'docstring': self._extract_go_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
            
            # Extract structs (Go's equivalent to classes)
            elif re.match(r'^\s*type\s+\w+\s+struct', line):
                struct_match = re.match(r'^\s*type\s+(\w+)\s+struct', line)
                if struct_match:
                    struct_name = struct_match.group(1)
                    
                    classes.append({
                        'name': struct_name,
                        'line': i + 1,
                        'docstring': self._extract_go_comment(lines, i),
                        'ai_description': '',
                        'ai_detailed_description': ''
                    })
        
        return functions, classes

    def _extract_python_docstring(self, lines: List[str], start_line: int) -> str:
        """Extract Python docstring from function/class definition"""
        docstring = ''
        for j in range(start_line, min(start_line + 10, len(lines))):
            if '"""' in lines[j] or "'''" in lines[j]:
                # Found docstring start
                doc_lines = []
                for k in range(j, len(lines)):
                    doc_lines.append(lines[k])
                    if k > j and ('"""' in lines[k] or "'''" in lines[k]):
                        break
                docstring = '\n'.join(doc_lines)
                break
        return docstring

    def _extract_js_comment(self, lines: List[str], line_index: int) -> str:
        """Extract JavaScript comment (JSDoc or regular comment)"""
        comment = ''
        # Look for comments above the function
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if line.startswith('/**') or line.startswith('/*'):
                # Start of block comment
                comment_lines = []
                for j in range(i, line_index):
                    comment_lines.append(lines[j])
                    if '*/' in lines[j]:
                        break
                comment = '\n'.join(comment_lines)
                break
            elif line.startswith('//'):
                # Single line comment
                comment = line
        return comment

    def _extract_java_javadoc(self, lines: List[str], line_index: int) -> str:
        """Extract Java Javadoc comment"""
        comment = ''
        # Look for Javadoc above the method/class
        for i in range(max(0, line_index - 10), line_index):
            line = lines[i].strip()
            if line.startswith('/**'):
                # Start of Javadoc
                comment_lines = []
                for j in range(i, line_index):
                    comment_lines.append(lines[j])
                    if '*/' in lines[j]:
                        break
                comment = '\n'.join(comment_lines)
                break
        return comment

    def _extract_c_comment(self, lines: List[str], line_index: int) -> str:
        """Extract C/C++ comment"""
        comment = ''
        # Look for comments above the function
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if line.startswith('/*') or line.startswith('//'):
                comment = line
                break
        return comment

    def _extract_csharp_comment(self, lines: List[str], line_index: int) -> str:
        """Extract C# XML documentation comment"""
        comment = ''
        # Look for XML doc comments above the method/class
        for i in range(max(0, line_index - 10), line_index):
            line = lines[i].strip()
            if line.startswith('///'):
                # Start of XML doc comment
                comment_lines = []
                for j in range(i, line_index):
                    if lines[j].strip().startswith('///'):
                        comment_lines.append(lines[j])
                    else:
                        break
                comment = '\n'.join(comment_lines)
                break
        return comment

    def _extract_php_comment(self, lines: List[str], line_index: int) -> str:
        """Extract PHP comment (PHPDoc or regular comment)"""
        comment = ''
        # Look for comments above the function/class
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if line.startswith('/**') or line.startswith('/*'):
                # Start of block comment
                comment_lines = []
                for j in range(i, line_index):
                    comment_lines.append(lines[j])
                    if '*/' in lines[j]:
                        break
                comment = '\n'.join(comment_lines)
                break
            elif line.startswith('//'):
                comment = line
        return comment

    def _extract_ruby_comment(self, lines: List[str], line_index: int) -> str:
        """Extract Ruby comment"""
        comment = ''
        # Look for comments above the method/class
        for i in range(max(0, line_index - 3), line_index):
            line = lines[i].strip()
            if line.startswith('#'):
                comment = line
                break
        return comment

    def _extract_go_comment(self, lines: List[str], line_index: int) -> str:
        """Extract Go comment"""
        comment = ''
        # Look for comments above the function/struct
        for i in range(max(0, line_index - 5), line_index):
            line = lines[i].strip()
            if line.startswith('//'):
                comment = line
                break
        return comment

    def _generate_documentation(self, repo_info, repo_path):
        """Generate HTML documentation with hybrid AI enhancements"""
        # Create output directory for this repository
        repo_output_dir = os.path.join(self.output_dir, repo_info['name'])
        os.makedirs(repo_output_dir, exist_ok=True)
        
        # Generate index page
        self._generate_index_page(repo_info, repo_output_dir)
        
        # Generate file documentation
        for file_info in repo_info['files']:
            self._generate_file_documentation(file_info, repo_output_dir)
        
        # Copy and process README if exists
        if repo_info['readme']:
            self._process_readme(repo_info['readme'], repo_output_dir)

    def _generate_index_page(self, repo_info, output_dir):
        """Generate repository index page with hybrid AI indicators"""
        ai_badge = ''
        if repo_info.get('ai_enhanced', False):
            if repo_info.get('hybrid_ai_used', False):
                ai_badge = ' Hybrid AI Enhanced (CodeBERT + Gemini)'
            else:
                ai_badge = ' AI Enhanced (CodeBERT)'
        
        template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{repo_name} - Documentation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-bottom: 10px; }}
        .ai-badge {{ display: inline-block; background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px; margin-left: 10px; }}
        .hybrid-ai-badge {{ display: inline-block; background: linear-gradient(45deg, #ff6b6b, #feca57); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px; margin-left: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .stat-card h3 {{ margin: 0 0 10px 0; font-size: 1.1em; }}
        .stat-card p {{ margin: 0; font-size: 1.5em; font-weight: bold; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-item {{ padding: 15px; margin: 10px 0; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px; transition: transform 0.2s; position: relative; }}
        .file-item:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .file-item a {{ text-decoration: none; color: #333; font-weight: bold; }}
        .language-tag {{ display: inline-block; padding: 4px 12px; margin: 3px; background: #667eea; color: white; border-radius: 15px; font-size: 12px; }}
        .file-language {{ position: absolute; top: 10px; right: 15px; background: rgba(102, 126, 234, 0.2); color: #333; padding: 4px 8px; border-radius: 10px; font-size: 12px; }}
        .generated-time {{ color: #666; font-size: 14px; margin-top: 30px; text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #667eea; }}
        .ai-detailed-summary {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ff6b6b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 {repo_name} Documentation {ai_badge_html}</h1>
            <p>Automatically generated documentation for the {repo_name} repository.</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>📄 Total Files</h3>
                <p>{total_files}</p>
            </div>
            <div class="stat-card">
                <h3>🔢 Total Lines</h3>
                <p>{total_lines}</p>
            </div>
            <div class="stat-card">
                <h3>🏷️ Languages</h3>
                <p>{language_count}</p>
            </div>
        </div>
        
        <div class="summary">
            <h3>📋 Repository Overview</h3>
            <p>This repository contains {total_files} documented files with {total_lines} lines of code across {language_count} different file types.</p>
            {ai_summary}
        </div>
        
        <h2>🔤 Languages Used</h2>
        <div style="margin: 20px 0;">
            {language_tags}
        </div>
        
        <h2>📁 File Structure</h2>
        <ul class="file-list">
            {file_list}
        </ul>
        
        <div class="generated-time">
            🕒 Generated on {timestamp}
            {ai_note}
        </div>
    </div>
</body>
</html>'''
        
        # Calculate statistics
        total_files = len(repo_info['files'])
        total_lines = sum(f.get('lines', 0) for f in repo_info['files'])
        language_count = len(repo_info['languages'])
        
        # Generate language tags with proper names
        language_tags = ''.join([
            f'<span class="language-tag">{self._get_language_display_name(ext)} ({count})</span>'
            for ext, count in repo_info['languages'].items()
        ])
        
        # Generate file list with AI summaries and language indicators
        file_items = []
        for file_info in sorted(repo_info['files'], key=lambda x: x['rel_path']):
            # Show both basic and detailed AI summaries if available
            ai_summary = ''
            if file_info.get('ai_summary'):
                ai_summary += f'<br><small><em>📝 {file_info["ai_summary"]}</em></small>'
            if file_info.get('ai_detailed_summary'):
                ai_summary += f'<br><small><strong>🤖 {file_info["ai_detailed_summary"]}</strong></small>'
            
            language_display = self._get_language_display_name(file_info['extension'])
            
            file_items.append(
                f'<li class="file-item">'
                f'<div class="file-language">{language_display}</div>'
                f'<a href="{self._sanitize_filename(file_info["name"])}.html">{file_info["rel_path"]}</a>'
                f' <small>({file_info.get("lines", 0)} lines)</small>'
                f'{ai_summary}'
                f'</li>'
            )
        
        # AI summary section
        ai_summary_section = ''
        if repo_info.get('ai_enhanced'):
            if repo_info.get('hybrid_ai_used'):
                ai_summary_section = '<p><strong>Hybrid AI Enhancement:</strong> This documentation has been enhanced with AI-generated descriptions using Microsoft CodeBERT for concise summaries and Google Gemini for detailed explanations.</p>'
            else:
                ai_summary_section = '<p><strong>AI Enhancement:</strong> This documentation has been enhanced with AI-generated descriptions and summaries for better understanding.</p>'
        
        # AI badge HTML
        ai_badge_html = ''
        if ai_badge:
            badge_class = 'hybrid-ai-badge' if 'Hybrid' in ai_badge else 'ai-badge'
            ai_badge_html = f'<span class="{badge_class}">{ai_badge}</span>'
        
        ai_note = ''
        if repo_info.get('hybrid_ai_used'):
            ai_note = '<br>Enhanced with Hybrid AI (CodeBERT + Gemini)'
        elif repo_info.get('ai_enhanced'):
            ai_note = '<br>Enhanced with AI-generated descriptions'
        
        # Render template
        html_content = template.format(
            repo_name=repo_info['name'],
            ai_badge_html=ai_badge_html,
            total_files=total_files,
            total_lines=total_lines,
            language_count=language_count,
            language_tags=language_tags,
            file_list='\n'.join(file_items),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ai_summary=ai_summary_section,
            ai_note=ai_note
        )
        
        # Write index file
        with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_file_documentation(self, file_info, output_dir):
        """Generate documentation for a single file with hybrid AI content"""
        try:
            # Syntax highlight the content
            highlighted_code = self._highlight_code(file_info['content'], file_info['extension'])
            
            # Generate function/class documentation for supported languages
            functions_html = ''
            classes_html = ''
            
            if file_info['extension'] in self.supported_extensions:
                if file_info.get('functions'):
                    functions_html = self._generate_functions_html(file_info['functions'], file_info['language'])
                if file_info.get('classes'):
                    classes_html = self._generate_classes_html(file_info['classes'], file_info['language'])
            
            # AI summary sections (both basic and detailed)
            ai_summary_section = ''
            if file_info.get('ai_summary'):
                ai_summary_section += f'''
                <div class="ai-summary">
                    <h3>📝 AI Summary</h3>
                    <p>{file_info['ai_summary']}</p>
                </div>
                '''
            
            if file_info.get('ai_detailed_summary'):
                ai_summary_section += f'''
                <div class="ai-detailed-summary">
                    <h3>🤖 Detailed AI Analysis</h3>
                    <p>{file_info['ai_detailed_summary']}</p>
                </div>
                '''
            
            # Language-specific icons and terms
            language_info = self._get_language_info(file_info['extension'])
            
            template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{file_name} - Documentation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 15px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .file-info {{ background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .ai-summary {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #667eea; }}
        .ai-detailed-summary {{ background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #ff6b6b; }}
        .code-container {{ background: #f8f8f8; border-radius: 10px; overflow-x: auto; margin: 20px 0; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1); }}
        .functions, .classes {{ margin: 20px 0; }}
        .function-item, .class-item {{ background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); padding: 20px; margin: 15px 0; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); }}
        .back-link {{ display: inline-block; margin-bottom: 20px; color: white; text-decoration: none; background: #667eea; padding: 10px 20px; border-radius: 25px; transition: background 0.3s; }}
        .back-link:hover {{ background: #764ba2; }}
        .language-badge {{ display: inline-block; background: #667eea; color: white; padding: 6px 12px; border-radius: 15px; font-size: 14px; margin-left: 10px; }}
        pre {{ margin: 0; padding: 15px; }}
        code {{ font-family: 'Consolas', 'Monaco', 'Courier New', monospace; }}
        .ai-description {{ background: rgba(102, 126, 234, 0.1); padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #667eea; }}
        .ai-detailed-description {{ background: rgba(255, 107, 107, 0.1); padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #ff6b6b; }}
        .docstring {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #28a745; }}
        .element-header {{ display: flex; align-items: center; margin-bottom: 10px; }}
        .element-header h4 {{ margin: 0; flex: 1; }}
        .line-number {{ background: #6c757d; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Back to Repository</a>
        
        <h1>{language_icon} {file_name} <span class="language-badge">{language_name}</span></h1>
        
        <div class="file-info">
            <p><strong>📍 Path:</strong> {file_path}</p>
            <p><strong>📏 Size:</strong> {file_size} bytes</p>
            <p><strong>📊 Lines:</strong> {file_lines}</p>
            <p><strong>🏷️ Language:</strong> {language_name}</p>
        </div>
        
        {ai_summary_section}
        
        {classes_section}
        {functions_section}
        
        <h2>📋 Source Code</h2>
        <div class="code-container">
            {highlighted_code}
        </div>
    </div>
</body>
</html>'''
            
            # Prepare sections with language-specific terms
            classes_section = ''
            functions_section = ''
            
            if file_info.get('classes'):
                classes_section = f'<h2>{language_info["class_icon"]} {language_info["class_term"]}</h2>{classes_html}'
            
            if file_info.get('functions'):
                functions_section = f'<h2>{language_info["function_icon"]} {language_info["function_term"]}</h2>{functions_html}'
            
            html_content = template.format(
                file_name=file_info['name'],
                language_icon=language_info['file_icon'],
                language_name=language_info['display_name'],
                file_path=file_info['rel_path'],
                file_size=file_info['size'],
                file_lines=file_info.get('lines', 0),
                ai_summary_section=ai_summary_section,
                classes_section=classes_section,
                functions_section=functions_section,
                highlighted_code=highlighted_code
            )
            
            # Write file documentation
            output_file = os.path.join(output_dir, f"{self._sanitize_filename(file_info['name'])}.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Failed to generate documentation for {file_info['name']}: {e}")

    def _get_language_info(self, extension: str) -> Dict[str, str]:
        """Get language-specific information for display"""
        language_info_map = {
            '.py': {
                'display_name': 'Python',
                'file_icon': '🐍',
                'function_icon': '⚙️',
                'function_term': 'Functions',
                'class_icon': '🏛️',
                'class_term': 'Classes'
            },
            '.js': {
                'display_name': 'JavaScript',
                'file_icon': '📜',
                'function_icon': '⚡',
                'function_term': 'Functions',
                'class_icon': '🏗️',
                'class_term': 'Classes'
            },
            '.ts': {
                'display_name': 'TypeScript',
                'file_icon': '📘',
                'function_icon': '⚡',
                'function_term': 'Functions',
                'class_icon': '🏗️',
                'class_term': 'Classes'
            },
            '.java': {
                'display_name': 'Java',
                'file_icon': '☕',
                'function_icon': '🔧',
                'function_term': 'Methods',
                'class_icon': '🏛️',
                'class_term': 'Classes'
            },
            '.cpp': {
                'display_name': 'C++',
                'file_icon': '⚡',
                'function_icon': '🔩',
                'function_term': 'Functions',
                'class_icon': '🏗️',
                'class_term': 'Classes'
            },
            '.c': {
                'display_name': 'C',
                'file_icon': '🔧',
                'function_icon': '🔩',
                'function_term': 'Functions',
                'class_icon': '📦',
                'class_term': 'Structures'
            },
            '.h': {
                'display_name': 'C/C++ Header',
                'file_icon': '📄',
                'function_icon': '📋',
                'function_term': 'Declarations',
                'class_icon': '📦',
                'class_term': 'Structures'
            },
            '.cs': {
                'display_name': 'C#',
                'file_icon': '🟣',
                'function_icon': '🔧',
                'function_term': 'Methods',
                'class_icon': '🏛️',
                'class_term': 'Classes'
            },
            '.php': {
                'display_name': 'PHP',
                'file_icon': '🐘',
                'function_icon': '⚙️',
                'function_term': 'Functions',
                'class_icon': '🏛️',
                'class_term': 'Classes'
            },
            '.rb': {
                'display_name': 'Ruby',
                'file_icon': '💎',
                'function_icon': '🔧',
                'function_term': 'Methods',
                'class_icon': '🏛️',
                'class_term': 'Classes'
            },
            '.go': {
                'display_name': 'Go',
                'file_icon': '🐹',
                'function_icon': '⚙️',
                'function_term': 'Functions',
                'class_icon': '📦',
                'class_term': 'Structs'
            }
        }
        
        return language_info_map.get(extension, {
            'display_name': extension.upper(),
            'file_icon': '📄',
            'function_icon': '⚙️',
            'function_term': 'Functions',
            'class_icon': '🏛️',
            'class_term': 'Classes'
        })

    def _get_language_display_name(self, extension: str) -> str:
        """Get human-readable language name from extension"""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go'
        }
        return language_map.get(extension, extension.upper())

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in HTML file names"""
        # Replace special characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return sanitized

    def _highlight_code(self, content, extension):
        """Apply syntax highlighting to code"""
        try:
            # Map extensions to lexer names
            lexer_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.h': 'c',
                '.cs': 'csharp',
                '.php': 'php',
                '.rb': 'ruby',
                '.go': 'go'
            }
            
            lexer_name = lexer_map.get(extension)
            if lexer_name:
                lexer = get_lexer_by_name(lexer_name, stripall=True)
            else:
                lexer = guess_lexer_for_filename('', content)
            
            formatter = HtmlFormatter(
                style='github', 
                linenos=True, 
                linenostart=1,
                cssclass="highlight"
            )
            return highlight(content, lexer, formatter)
        except Exception as e:
            logger.warning(f"Failed to highlight code: {e}")
            # Fallback to plain text with line numbers
            lines = content.split('\n')
            numbered_lines = []
            for i, line in enumerate(lines, 1):
                numbered_lines.append(f'<span style="color: #666; margin-right: 10px;">{i:4d}</span>{line}')
            return f'<pre><code>{"<br>".join(numbered_lines)}</code></pre>'

    def _process_readme(self, readme_path, output_dir):
        """Process and copy README file"""
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Convert markdown to HTML if it's a .md file
            if readme_path.lower().endswith('.md'):
                html_content = markdown.markdown(content, extensions=['codehilite', 'fenced_code', 'tables'])
                
                template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .back-link {{ display: inline-block; margin-bottom: 20px; color: white; text-decoration: none; background: #667eea; padding: 10px 20px; border-radius: 25px; transition: background 0.3s; }}
        .back-link:hover {{ background: #764ba2; }}
        h1, h2, h3, h4, h5, h6 {{ color: #333; }}
        h1 {{ border-bottom: 3px solid #667eea; padding-bottom: 15px; }}
        h2 {{ border-bottom: 2px solid #a8edea; padding-bottom: 10px; }}
        code {{ background: #f8f8f8; padding: 3px 6px; border-radius: 4px; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; }}
        pre {{ background: #f8f8f8; padding: 20px; border-radius: 10px; overflow-x: auto; border-left: 4px solid #667eea; }}
        blockquote {{ background: #f0f7ff; padding: 15px; border-left: 4px solid #667eea; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Back to Repository</a>
        {content}
    </div>
</body>
</html>'''
                
                final_content = template.format(content=html_content)
                
                with open(os.path.join(output_dir, 'README.html'), 'w', encoding='utf-8') as f:
                    f.write(final_content)
            else:
                # Copy as plain text
                shutil.copy2(readme_path, os.path.join(output_dir, 'README.txt'))
                
        except Exception as e:
            logger.error(f"Failed to process README: {e}")

    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages and their display names"""
        return {ext: self._get_language_display_name(ext) for ext in self.supported_extensions}

    def generate_language_report(self, repo_info) -> str:
        """Generate a detailed language breakdown report"""
        language_stats = {}
        total_lines = 0
        
        for file_info in repo_info['files']:
            ext = file_info['extension']
            lines = file_info.get('lines', 0)
            
            if ext not in language_stats:
                language_stats[ext] = {
                    'files': 0,
                    'lines': 0,
                    'functions': 0,
                    'classes': 0
                }
            
            language_stats[ext]['files'] += 1
            language_stats[ext]['lines'] += lines
            language_stats[ext]['functions'] += len(file_info.get('functions', []))
            language_stats[ext]['classes'] += len(file_info.get('classes', []))
            total_lines += lines
        
        # Generate HTML report
        report_html = '<div class="language-report">'
        for ext, stats in sorted(language_stats.items(), key=lambda x: x[1]['lines'], reverse=True):
            percentage = (stats['lines'] / total_lines * 100) if total_lines > 0 else 0
            language_name = self._get_language_display_name(ext)
            
            report_html += f'''
            <div class="language-stat">
                <h4>{language_name} ({ext})</h4>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage:.1f}%"></div>
                </div>
                <p>{stats['files']} files, {stats['lines']} lines ({percentage:.1f}%)</p>
                <p>{stats['functions']} functions/methods, {stats['classes']} classes/structs</p>
            </div>
            '''
        
        report_html += '</div>'
        return report_html


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generator = DocsGenerator()
    
    print("Hybrid AI Documentation Generator")
    print("- CodeBERT: One-sentence summaries")
    print("- Gemini: Detailed explanations")
    print("- Supported Languages:")
    for ext, name in generator.get_supported_languages().items():
        print(f"  {ext} - {name}")
    
    # Test with a sample repository
    repo_url = "https://github.com/your-username/your-repo.git"
    
    print("\nStarting hybrid AI documentation generation...")
    success = generator.generate_from_repo(
        repo_url=repo_url,
        branch="main",
        use_ai_enhancement=True
    )
    
    if success:
        print(f"Documentation generated successfully!")
        print(f"Check the output directory: {generator.output_dir}")
        if generator.ai_enhancer.gemini_available:
            print("✅ Hybrid AI Enhancement: CodeBERT + Gemini")
        else:
            print("⚠️ Only CodeBERT enhancement available (Gemini API key not configured)")
    else:
        print("Documentation generation failed. Check the logs for details.")
