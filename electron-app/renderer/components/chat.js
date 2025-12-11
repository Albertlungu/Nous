// Chat functionality
let isGenerating = false;

document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const instructionInput = document.getElementById('instruction-input');
    const contextInput = document.getElementById('context-input');
    const messagesContainer = document.getElementById('messages');

    if (!sendBtn || !instructionInput || !messagesContainer) {
        console.error('Chat elements not found');
        return;
    }

    // Use ESC to stop generation (like in Claude)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isGenerating) {
            isGenerating = false;

            if (stopBtn && sendBtn) {
                stopBtn.style.opacity = '0';
                stopBtn.style.transform = 'scale(0.8)';

                setTimeout(() => {
                    stopBtn.style.display = 'none';
                    sendBtn.style.display = 'block';

                    requestAnimationFrame(() => {
                        sendBtn.style.opacity = '1';
                        sendBtn.style.transform = 'scale(1)';
                    });
                }, 150);
            }
        }
    });

    sendBtn.addEventListener('click', handleSend);

    // Send message on Enter (Shift+Enter for new line)
    instructionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Stop generation
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            isGenerating = false;

            stopBtn.style.opacity = '0';
            stopBtn.style.transform = 'scale(0.8)';

            setTimeout(() => {
                stopBtn.style.display = 'none';
                sendBtn.style.display = 'block';

                requestAnimationFrame(() => {
                    sendBtn.style.opacity = '1';
                    sendBtn.style.transform = 'scale(1)';
                });
            }, 150);
        });
    }

    async function handleSend() {
        if (isGenerating) return;

        // Check if server is healthy and model is loaded
        try {
            const healthCheck = await fetch('http:127.0.0.1:5000/api/health');
            const health = await healthCheck.json();
            if (!health.model_loaded) {
                showError('Please load a model first from the Model Config page');
                return;
            }
        } catch (e) {
            showError('Cannot connect to server. Make sure Flask is running.');
            return;
        }

        const instruction = instructionInput.value.trim();
        const context = contextInput.value.trim();

        if (!instruction) {
            showError('Please enter a prompt');
            return;
        }

        // Get generation parameters
        const maxTokens = parseInt(document.getElementById('max-tokens')?.value || 100);
        const temperature = parseFloat(document.getElementById('temperature')?.value || 0.7)
        const topK = parseInt(document.getElementById('top-k')?.value || 40);

        // Show user message with context if given
        const userMessage = context
            ? `${instruction}\n\nContext: ${context}`
            : instruction;
        addMessage('user', instruction);

        // Clear input
        instructionInput.value = '';
        contextInput.value = '';

        // Show stop button
        isGenerating = true;
        if (sendBtn && stopBtn) {
            sendBtn.style.opacity = '0';
            sendBtn.style.transform = 'scale(0.8)';

            setTimeout(() => {
                sendBtn.style.display = 'none';
                stopBtn.style.display = 'block';

                requestAnimationFrame(() => {
                    stopBtn.style.opacity = '1';
                    stopBtn.style.transform = 'scale(1)';
                });
            }, 150);
        }

        // Show loading indicator
        const loadingId = showLoading();

        try {
            const response = await fetch('http://127.0.0.1:5000/api/chat/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: instruction,
                    context: context || undefined,
                    max_tokens: maxTokens,
                    temperature: temperature,
                    top_k: topK
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            // Remove loading indicator
            removeLoading(loadingId);

            // Handle streaming response
            await handleStreamingResponse(response);

        } catch (error) {
            console.error('Chat error:', error);
            removeLoading(loadingId);
            showError(`Failed to generate response: ${error.message}`);
        } finally {
            isGenerating = false;
            if (sendBtn && stopBtn) {
                stopBtn.style.opacity = '0';
                stopBtn.style.transform = 'scale(0.8)';

                setTimeout(() => {
                    stopBtn.style.display = 'none';
                    sendBtn.style.display = 'block';

                    requestAnimationFrame(() => {
                        sendBtn.style.opacity = '1';
                        sendBtn.style.transform = 'scale(1)';
                    });
                }, 150);
            }
        }
    }

    async function handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessage = '';
        let messageElement = null;

        try {
            while (isGenerating) {
                const { done, value } = await reader.read();

                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.trim() || !line.startsWith('data: ')) continue;

                    const data = line.slice(6); // Remove 'data' prefix

                    try {
                        const parsed = JSON.parse(data);

                        if (parsed.error) {
                            throw new Error(parsed.error);
                        }

                        if (parsed.done) {
                            isGenerating = false;
                            break;
                        }

                        if (parsed.token) {
                            assistantMessage += parsed.token;

                            // Create/update message element
                            if (!messageElement) {
                                messageElement = addMessage('assistant', assistantMessage);
                            } else {
                                updateMessage(messageElement, assistantMessage);
                            }
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            throw error;
        }
    }

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const roleLabel = document.createElement('div');
        roleLabel.className = `message-role`;
        roleLabel.textContent = role === 'user' ? 'You' : 'Assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(roleLabel);
        messageDiv.appendChild(contentDiv);

        messagesContainer.appendChild(messageDiv);

        // Hide background logo when first message is sent
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.classList.add('has-messages');

            setTimeout(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }, 10);
        }

        return messageDiv
    }

    function updateMessage(messageElement, content) {
        const contentDiv = messageElement.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.textContent = content;

            // Scroll the chat container
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    }

    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message-loading';
        loadingDiv.innerHTML = '<span></span><span></span><span></span>';
        loadingDiv.id = `loading-${Date.now()}`;

        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return loadingDiv.id;
    }

    function removeLoading(loadingId) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;

        messagesContainer.appendChild(errorDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        setTimeout(() => errorDiv.remove(), 5000);
    }
});
