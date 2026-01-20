import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set")

    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # GitHub webhook settings
    GITHUB_WEBHOOK_SECRET = os.environ.get('GITHUB_WEBHOOK_SECRET', '')
    
    # Documentation settings
    DOCS_OUTPUT_DIR = os.environ.get('DOCS_OUTPUT_DIR', 'docs_output')
    TEMP_DIR = os.environ.get('TEMP_DIR', 'temp')
    
    # Supported file extensions for documentation
    DOC_EXTENSIONS = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go']
    
    # Documentation formats
    OUTPUT_FORMATS = ['html', 'markdown']
    
    # Repository settings
    MAX_REPO_SIZE = int(os.environ.get('MAX_REPO_SIZE', 100))  # MB
    CLONE_TIMEOUT = int(os.environ.get('CLONE_TIMEOUT', 300))  # seconds
