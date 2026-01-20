// Auto Documentation Tool JavaScript

// Global variables
let systemStatus = 'online';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    loadSystemStatus();
    setupEventListeners();
    updateWebhookUrl();
    
    // Refresh status every 30 seconds
    setInterval(loadSystemStatus, 30000);
}

function setupEventListeners() {
    // Manual form submission
    const manualForm = document.getElementById('manual-form');
    if (manualForm) {
        manualForm.addEventListener('submit', handleManualGeneration);
    }
}

function loadSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateStatusIndicator(data.status);
            updateDocsCount(data.docs_count);
        })
        .catch(error => {
            console.error('Failed to load system status:', error);
            updateStatusIndicator('offline');
        });
}

function updateStatusIndicator(status) {
    const statusElement = document.getElementById('status');
    const statusDot = statusElement.querySelector('.status-dot');
    
    if (status === 'running') {
        statusDot.className = 'status-dot online';
        statusElement.innerHTML = '<span class="status-dot online"></span>System Online';
        systemStatus = 'online';
    } else {
        statusDot.className = 'status-dot offline';
        statusElement.innerHTML = '<span class="status-dot offline"></span>System Offline';
        systemStatus = 'offline';
    }
}

function updateDocsCount(count) {
    const docsCountElement = document.getElementById('docs-count');
    if (docsCountElement) {
        docsCountElement.textContent = count || '0';
    }
}

function handleManualGeneration(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const data = {
        repo_url: formData.get('repo_url'),
        branch: formData.get('branch') || 'main'
    };
    
    // Validate input
    if (!data.repo_url) {
        showStatusMessage('Please enter a repository URL', 'error');
        return;
    }
    
    if (!isValidRepoUrl(data.repo_url)) {
        showStatusMessage('Please enter a valid repository URL', 'error');
        return;
    }
    
    // Show loading state
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.innerHTML = '<span class="spinner"></span>Generating...';
    submitButton.disabled = true;
    
    showStatusMessage('Starting documentation generation...', 'info');
    
    // Send request
    fetch('/api/regenerate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.status === 'started') {
            showStatusMessage('Documentation generation started! This may take a few minutes.', 'success');
            // Poll for completion (optional)
            pollGenerationStatus();
        } else {
            showStatusMessage('Failed to start documentation generation', 'error');
        }
    })
    .catch(error => {
        console.error('Generation failed:', error);
        showStatusMessage('Failed to generate documentation. Please try again.', 'error');
    })
    .finally(() => {
        // Reset button
        submitButton.textContent = originalText;
        submitButton.disabled = false;
    });
}

function pollGenerationStatus() {
    // This could poll an endpoint to check generation status
    // For now, we'll just refresh the page after a delay
    setTimeout(() => {
        location.reload();
    }, 10000); // Refresh after 10 seconds
}

function isValidRepoUrl(url) {
    // Basic URL validation for Git repositories
    const gitUrlPatterns = [
        /^https:\/\/github\.com\/[\w-]+\/[\w-]+\.git$/,
        /^https:\/\/github\.com\/[\w-]+\/[\w-]+$/,
        /^https:\/\/gitlab\.com\/[\w-]+\/[\w-]+\.git$/,
        /^https:\/\/gitlab\.com\/[\w-]+\/[\w-]+$/,
        /^https:\/\/bitbucket\.org\/[\w-]+\/[\w-]+\.git$/,
        /^https:\/\/bitbucket\.org\/[\w-]+\/[\w-]+$/
    ];
    
    return gitUrlPatterns.some(pattern => pattern.test(url));
}

function showStatusMessage(message, type) {
    const statusElement = document.getElementById('manual-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        
        // Hide after 5 seconds for non-error messages
        if (type !== 'error') {
            setTimeout(() => {
                statusElement.className = 'status-message';
            }, 5000);
        }
    }
}

function updateWebhookUrl() {
    const webhookUrlElement = document.getElementById('webhook-url');
    if (webhookUrlElement) {
        const baseUrl = window.location.origin;
        const webhookUrl = `${baseUrl}/webhook`;
        webhookUrlElement.textContent = webhookUrl;
    }
}

function copyWebhookUrl() {
    const webhookUrlElement = document.getElementById('webhook-url');
    if (webhookUrlElement) {
        const url = webhookUrlElement.textContent;
        
        // Try to use the Clipboard API
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(url).then(() => {
                showNotification('Webhook URL copied to clipboard!');
            }).catch(err => {
                fallbackCopyTextToClipboard(url);
            });
        } else {
            fallbackCopyTextToClipboard(url);
        }
    }
}

function fallbackCopyTextToClipboard(text) {
    // Fallback method for copying text
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showNotification('Webhook URL copied to clipboard!');
    } catch (err) {
        showNotification('Failed to copy URL. Please copy manually.');
    }
    
    document.body.removeChild(textArea);
}

function showNotification(message) {
    // Create a temporary notification
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 15px 20px;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 1000;
        font-weight: 600;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Add animation keyframes if not already added
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Utility functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global use
window.copyWebhookUrl = copyWebhookUrl;