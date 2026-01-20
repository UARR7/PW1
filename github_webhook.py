import hashlib
import hmac
import logging
from config import Config

logger = logging.getLogger(__name__)

class GitHubWebhook:
    def __init__(self):
        self.config = Config()
        self.secret = self.config.GITHUB_WEBHOOK_SECRET.encode('utf-8') if self.config.GITHUB_WEBHOOK_SECRET else None
    
    def verify_signature(self, payload, signature):
    
        if not self.secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return False
        
        if not signature:
            return False
        
        # Remove 'sha256=' prefix
        signature = signature.replace('sha256=', '')
        
        # Calculate expected signature
        expected_signature = hmac.new(
            self.secret,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)

