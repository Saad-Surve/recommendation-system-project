async function handleInteraction(articleId, interactionType) {
    const button = document.getElementById(`${interactionType}-btn-${articleId}`);
    const statusDiv = document.getElementById(`status-${articleId}`);
    
    try {
        button.disabled = true;
        button.classList.add('opacity-50');
        statusDiv.textContent = 'Processing...';
        statusDiv.className = 'mt-2 text-sm text-gray-500';
        
        const response = await fetch(`/interact/${articleId}/${interactionType}`, {
            method: 'POST',
            credentials: 'include'
        });
        
        if (response.ok) {
            statusDiv.textContent = `Successfully ${interactionType}d!`;
            statusDiv.className = 'mt-2 text-sm text-green-600';
            
            // Update interaction count if it exists
            const countElement = document.getElementById(`${interactionType}-count`);
            if (countElement) {
                const currentCount = parseInt(countElement.textContent);
                countElement.textContent = currentCount + 1;
            }
        } else {
            statusDiv.textContent = 'Failed to process interaction';
            statusDiv.className = 'mt-2 text-sm text-red-600';
        }
    } catch (error) {
        console.error('Interaction error:', error);
        statusDiv.textContent = 'Error processing interaction';
        statusDiv.className = 'mt-2 text-sm text-red-600';
    } finally {
        setTimeout(() => {
            button.disabled = false;
            button.classList.remove('opacity-50');
            setTimeout(() => {
                statusDiv.textContent = '';
            }, 2000);
        }, 500);
    }
}

async function handleShare(articleId, articleTitle) {
    const url = `${window.location.origin}/articles/${articleId}`;
    const statusDiv = document.getElementById(`status-${articleId}`);
    
    try {
        if (navigator.share) {
            await navigator.share({
                title: articleTitle,
                text: 'Check out this article!',
                url: url
            });
            
            // Record share interaction
            await fetch(`/interact/${articleId}/share`, {
                method: 'POST',
                credentials: 'include'
            });
            
            statusDiv.textContent = 'Shared successfully!';
        } else {
            await navigator.clipboard.writeText(url);
            
            // Record share interaction
            await fetch(`/interact/${articleId}/share`, {
                method: 'POST',
                credentials: 'include'
            });
            
            statusDiv.textContent = 'Link copied to clipboard!';
        }
        statusDiv.className = 'mt-2 text-sm text-green-600';
    } catch (error) {
        console.error('Share error:', error);
        statusDiv.textContent = 'Error sharing article';
        statusDiv.className = 'mt-2 text-sm text-red-600';
    }
    
    setTimeout(() => {
        statusDiv.textContent = '';
    }, 2000);
}

async function recordView(articleId) {
    try {
        await fetch(`/interact/${articleId}/view`, {
            method: 'POST',
            credentials: 'include'
        });
    } catch (error) {
        console.error('View recording error:', error);
    }
}

async function handleNotInterested(articleId) {
    const button = document.getElementById(`not-interested-${articleId}`);
    const statusDiv = document.getElementById(`status-${articleId}`);
    
    try {
        button.disabled = true;
        button.classList.add('opacity-50');
        statusDiv.textContent = 'Processing...';
        statusDiv.className = 'mt-4 text-lg font-bold text-black';
        
        const response = await fetch(`/interact/${articleId}/not_interested`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include'
        });
        
        if (response.ok) {
            statusDiv.textContent = 'Marked as not interested. Redirecting...';
            statusDiv.className = 'mt-4 text-lg font-bold text-green-600';
            
            // Redirect to dashboard after a short delay
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1500);
        } else {
            statusDiv.textContent = 'Failed to process interaction';
            statusDiv.className = 'mt-4 text-lg font-bold text-red-600';
            button.disabled = false;
            button.classList.remove('opacity-50');
        }
    } catch (error) {
        console.error('Not Interested error:', error);
        statusDiv.textContent = 'Error processing interaction';
        statusDiv.className = 'mt-4 text-lg font-bold text-red-600';
        button.disabled = false;
        button.classList.remove('opacity-50');
    }
} 