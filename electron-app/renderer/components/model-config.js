// Model configuration functionality

document.addEventListener('DOMContentLoaded', () => {
    const loadModelBtn = document.getElementById('load-model-btn');
    const listModelsBtn = document.getElementById('list-models-btn');
    const unloadModelBtn = document.getElementById('unload-model-btn');
    const modelInfo = document.getElementById('model-info');

    if (loadModelBtn) {
        loadModelBtn.addEventListener('click', async () => {
            await listAndSelectModel();
        });
    }

    if (listModelsBtn) {
        listModelsBtn.addEventListener('click', async () => {
            await listModels();
        });
    }

    if (unloadModelBtn) {
        unloadModelBtn.addEventListener('click', async () => {
            await unloadModel();
        });
    }

    // Auto-load model info on page load
    updateModelInfo();

    async function listAndSelectModel() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/models/list');
            const data = await response.json();

            if (!data.models || data.models.length === 0) {
                showModelError('No models found in artifacts/models/');
                return;
            }

            // Show model selection UI
            showModelSelector(data.models);

        } catch (error) {
            console.error('List models error:', error);
            showModelError(`Failed to list models: ${error.message}`);
        }
    }

    function showModelSelector(models) {
        const container = document.createElement('div');
        container.style.marginTop = '20px';

        const title = document.createElement('h3');
        title.style.cssText = 'font-size: 16px; margin-bottom: 12px; color: var(--text-primary);';
        title.textContent = 'Select a model to load:';

        const listContainer = document.createElement('div');
        listContainer.style.cssText = 'display: flex; flex-direction: column; gap: 8px;';

        models.forEach(model => {
            const item = document.createElement('div');
            item.className = 'list-item';
            item.style.cursor = 'pointer';

            const itemContent = document.createElement('div');

            const itemTitle = document.createElement('div');
            itemTitle.className = 'list-item-title';
            itemTitle.textContent = model.name;

            const itemMeta = document.createElement('div');
            itemMeta.className = 'list-item-meta';
            itemMeta.textContent = `${model.size_mb.toFixed(2)} MB • ${new Date(model.modified * 1000).toLocaleString()}`;

            itemContent.appendChild(itemTitle);
            itemContent.appendChild(itemMeta);
            item.appendChild(itemContent);

            // Add click handler directly
            item.addEventListener('click', () => {
                selectModel(model.path);
            });

            listContainer.appendChild(item);
        });

        container.appendChild(title);
        container.appendChild(listContainer);

        modelInfo.innerHTML = '';
        modelInfo.appendChild(container);
    }

    // Global function for model selection
    async function selectModel(modelPath) {
        try {
            // Show persistent loading message
            modelInfo.innerHTML = `
                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); padding: 16px; border-radius: 8px; color: var(--text-primary);">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="message-loading" style="padding: 0;">
                            <span></span><span></span><span></span>
                        </div>
                        <span>Loading model... This may take a moment.</span>
                    </div>
                </div>
            `;

            const response = await fetch('http://127.0.0.1:5000/api/models/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_path: modelPath
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to load model');
            }

            // Show success message briefly
            modelInfo.innerHTML = `
                <div style="background: var(--bg-tertiary); border: 1px solid var(--accent-primary); padding: 12px; border-radius: 8px; color: var(--accent-primary);">
                    Model loaded successfully!
                </div>
            `;

            // Wait a moment then show model info
            setTimeout(async () => {
                await updateModelInfo();
            }, 1500);

        } catch (error) {
            console.error('Load model error:', error);
            showModelError(`Failed to load model: ${error.message}`);
        }
    };

    async function listModels() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/models/list');
            const data = await response.json();

            if (!data.models || data.models.length === 0) {
                modelInfo.innerHTML = '<p style="color: var(--text-tertiary);">No models found</p>';
                return;
            }

            const html = `
                <h3 style="font-size: 16px; margin-bottom: 12px; color: var(--text-primary);">Available Models</h3>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    ${data.models.map(model => `
                        <div class="list-item">
                            <div>
                                <div class="list-item-title">${model.name}</div>
                                <div class="list-item-meta">
                                    ${model.size_mb.toFixed(2)} MB "
                                    ${new Date(model.modified * 1000).toLocaleString()}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;

            modelInfo.innerHTML = html;

        } catch (error) {
            console.error('List models error:', error);
            showModelError(`Failed to list models: ${error.message}`);
        }
    }

    async function unloadModel() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/models/unload', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to unload model');
            }

            showModelInfo('Model unloaded successfully');
            modelInfo.innerHTML = '';

        } catch (error) {
            console.error('Unload model error:', error);
            showModelError(`Failed to unload model: ${error.message}`);
        }
    }

    async function updateModelInfo() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/models/info');
            const data = await response.json();

            if (response.status === 400 || !data || data.error) {
                modelInfo.innerHTML = '<p style="color: var(--text-tertiary);">No model loaded</p>';
                return;
            }

            const html = `
                <div class="info-card">
                    <div class="info-card-label">Model Path</div>
                    <div class="info-card-value">${data.model_path || 'Unknown'}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Total Parameters</div>
                    <div class="info-card-value">${(data.total_parameters / 1000000).toFixed(2)}M</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Configuration</div>
                    <div class="info-card-value" style="font-size: 13px; font-weight: 400;">
                        Blocks: ${data.config?.num_blocks || 'N/A'} "
                        Heads: ${data.config?.num_heads || 'N/A'} "
                        Embed Dim: ${data.config?.embedding_dim || 'N/A'}<br>
                        Vocab: ${data.config?.vocab_size || 'N/A'} "
                        Max Seq: ${data.config?.max_seq_len || 'N/A'}
                    </div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Tokenizer</div>
                    <div class="info-card-value">${data.tokenizer || 'Unknown'}</div>
                </div>
            `;

            modelInfo.innerHTML = html;

        } catch (error) {
            console.error('Get model info error:', error);
        }
    }

    function showModelInfo(message, autoRemove = false) {
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'background: var(--bg-tertiary); border: 1px solid var(--border-primary); padding: 12px; border-radius: 8px; margin-bottom: 12px; color: var(--text-primary);';
        infoDiv.textContent = message;

        modelInfo.innerHTML = '';
        modelInfo.appendChild(infoDiv);

        if (autoRemove) {
            setTimeout(() => infoDiv.remove(), 3000);
        }
    }

    function showModelError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;

        modelInfo.innerHTML = '';
        modelInfo.appendChild(errorDiv);
    }
});
