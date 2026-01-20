import os
import shutil
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def cleanup_temp_files(temp_dir, max_age_hours=24):
    try:
        current_time = datetime.now()
        
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            
            # Get file/directory creation time
            creation_time = datetime.fromtimestamp(os.path.getctime(item_path))
            age_hours = (current_time - creation_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
                logger.info(f"Cleaned up old temp file/dir: {item}")
                
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")

def validate_repo_url(url):
    if not url:
        return False
    
    # Check for common Git hosting patterns
    valid_patterns = [
        'github.com',
        'gitlab.com',
        'bitbucket.org',
        '.git'
    ]
    
    return any(pattern in url.lower() for pattern in valid_patterns)

def safe_filename(filename):
    import re
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    return safe_name[:255]
