// Training functionality

document.addEventListener('DOMContentLoaded', () => {
    const startTrainingBtn = document.getElementById('start-training-btn');
    const pauseTrainingBtn = document.getElementById('pause-training-btn');
    const resumeTrainingBtn = document.getElementById('resume-training-btn');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const resetConfigBtn = document.getElementById('reset-config-btn');
    const calcParamsBtn = document.getElementById('calc-params-btn');
    const extendTrainingBtn = document.getElementById('extend-training-btn');
    const trainingStatus = document.getElementById('training-status');

    // Tokenizer elements
    const tokenizerToggle = document.getElementById('tokenizer-toggle');
    const tokenizerTypeToggleLabels = document.querySelectorAll('#tokenizer-panel > .tokenizer-type-select > .toggle-label');
    const tiktokenOptions = document.getElementById('tiktoken-options');
    const bpeOptions = document.getElementById('bpe-options');
    const bpeChoiceToggle = document.getElementById('bpe-choice-toggle');
    const bpeChoiceLabels = document.querySelectorAll('#bpe-options .bpe-choice > .toggle-label');
    const bpeExisting = document.getElementById('bpe-existing');
    const bpeNew = document.getElementById('bpe-new');
    const setTiktokenBtn = document.getElementById('set-tiktoken-btn');
    const loadBpeBtn = document.getElementById('load-bpe-btn');
    const trainBpeBtn = document.getElementById('train-bpe-btn');
    const bpeDatasetSelect = document.getElementById('bpe-dataset-select');
    const tokenizerStatus = document.getElementById('tokenizer-status');

    let statusInterval = null;

    // Load saved config on page load
    loadConfig();

    // Load datasets for BPE training
    loadDatasets();

    // Auto-select all text on focus for input fields
    const allInputs = document.querySelectorAll('#training-view input[type="number"], #training-view input[type="text"]');
    allInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.select();
        });
    });

    if (startTrainingBtn) {
        startTrainingBtn.addEventListener('click', async () => {
            await startTraining();
        });
    }

    if (pauseTrainingBtn) {
        pauseTrainingBtn.addEventListener('click', async () => {
            await pauseTraining();
        });
    }

    if (resumeTrainingBtn) {
        resumeTrainingBtn.addEventListener('click', async () => {
            await resumeTraining();
        });
    }

    if (stopTrainingBtn) {
        stopTrainingBtn.addEventListener('click', async () => {
            await stopTraining();
        });
    }

    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', () => {
            saveConfig();
        });
    }

    if (resetConfigBtn) {
        resetConfigBtn.addEventListener('click', () => {
            resetConfig();
        });
    }

    if (calcParamsBtn) {
        calcParamsBtn.addEventListener('click', () => {
            calculateParameters();
        });
    }

    if (extendTrainingBtn) {
        extendTrainingBtn.addEventListener('click', () => {
            showExtendTrainingModal();
        });
    }

    // Tokenizer type toggle
    if (tokenizerToggle) {
        tokenizerToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                // BPE selected
                tiktokenOptions.style.display = 'none';
                bpeOptions.style.display = 'flex';
                tokenizerTypeToggleLabels[0].classList.remove('active');
                tokenizerTypeToggleLabels[1].classList.add('active');
            } else {
                // TikToken selected
                tiktokenOptions.style.display = 'flex';
                bpeOptions.style.display = 'none';
                tokenizerTypeToggleLabels[0].classList.add('active');
                tokenizerTypeToggleLabels[1].classList.remove('active');
            }
        });

        // Set initial state
        tokenizerTypeToggleLabels[0].classList.add('active');
    }

    // BPE choice toggle
    if (bpeChoiceToggle) {
        bpeChoiceToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                // Train New selected
                bpeExisting.style.display = 'none';
                bpeNew.style.display = 'flex';
                bpeChoiceLabels[0].classList.remove('active');
                bpeChoiceLabels[1].classList.add('active');
            } else {
                // Use Existing selected
                bpeExisting.style.display = 'flex';
                bpeNew.style.display = 'none';
                bpeChoiceLabels[0].classList.add('active');
                bpeChoiceLabels[1].classList.remove('active');
            }
        });

        // Set initial state
        bpeChoiceLabels[0].classList.add('active');
    }

    // Set TikToken button
    if (setTiktokenBtn) {
        setTiktokenBtn.addEventListener('click', async () => {
            await setTikToken();
        });
    }

    // Load BPE button
    if (loadBpeBtn) {
        loadBpeBtn.addEventListener('click', async () => {
            await loadBPE();
        });
    }

    // Train BPE button
    if (trainBpeBtn) {
        trainBpeBtn.addEventListener('click', async () => {
            await trainBPE();
        });
    }

    async function startTraining() {
        const config = getConfigFromForm();

        try {
            const response = await fetch('http://127.0.0.1:5000/api/training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ config })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to start training');
            }

            showTrainingInfo('Training started successfully');
            startStatusPolling();

        } catch (error) {
            console.error('Start training error:', error);
            showTrainingError(`Failed to start training: ${error.message}`);
        }
    }

    async function pauseTraining() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/training/pause', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to pause training');
            }

            showTrainingInfo('Training paused');

        } catch (error) {
            console.error('Pause training error:', error);
            showTrainingError(`Failed to pause training: ${error.message}`);
        }
    }

    async function resumeTraining() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/training/resume', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to resume training');
            }

            showTrainingInfo('Training resumed');
            startStatusPolling();

        } catch (error) {
            console.error('Resume training error:', error);
            showTrainingError(`Failed to resume training: ${error.message}`);
        }
    }

    async function stopTraining() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/training/stop', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to stop training');
            }

            showTrainingInfo('Training stopped');
            stopStatusPolling();

        } catch (error) {
            console.error('Stop training error:', error);
            showTrainingError(`Failed to stop training: ${error.message}`);
        }
    }

    function getConfigFromForm() {
        return {
            num_blocks: parseInt(document.getElementById('num-blocks')?.value || 8),
            num_heads: parseInt(document.getElementById('num-heads')?.value || 8),
            embedding_dim: parseInt(document.getElementById('embedding-dim')?.value || 512),
            max_seq_len: parseInt(document.getElementById('max-seq-len')?.value || 256),
            dropout: parseFloat(document.getElementById('dropout')?.value || 0.0),
            vocab_size: parseInt(document.getElementById('vocab-size')?.value || 50257),
            num_epochs: parseInt(document.getElementById('num-epochs')?.value || 75),
            batch_size: parseInt(document.getElementById('batch-size')?.value || 64),
            learning_rate: parseFloat(document.getElementById('learning-rate')?.value || 0.0011),
            min_lr: parseFloat(document.getElementById('min-lr')?.value || 0.000005),
            warmup_steps: parseInt(document.getElementById('warmup-steps')?.value || 500),
            save_every: parseInt(document.getElementById('save-every')?.value || 1)
        };
    }

    function saveConfig() {
        const config = getConfigFromForm();
        localStorage.setItem('training_config', JSON.stringify(config));
        showTrainingInfo('Configuration saved', true);
    }

    function loadConfig() {
        const saved = localStorage.getItem('training_config');
        if (!saved) return;

        try {
            const config = JSON.parse(saved);

            if (document.getElementById('num-blocks')) document.getElementById('num-blocks').value = config.num_blocks || 8;
            if (document.getElementById('num-heads')) document.getElementById('num-heads').value = config.num_heads || 8;
            if (document.getElementById('embedding-dim')) document.getElementById('embedding-dim').value = config.embedding_dim || 512;
            if (document.getElementById('max-seq-len')) document.getElementById('max-seq-len').value = config.max_seq_len || 256;
            if (document.getElementById('dropout')) document.getElementById('dropout').value = config.dropout || 0.0;
            if (document.getElementById('num-epochs')) document.getElementById('num-epochs').value = config.num_epochs || 75;
            if (document.getElementById('batch-size')) document.getElementById('batch-size').value = config.batch_size || 64;
            if (document.getElementById('learning-rate')) document.getElementById('learning-rate').value = config.learning_rate || 0.0011;
            if (document.getElementById('min-lr')) document.getElementById('min-lr').value = config.min_lr || 0.000005;
            if (document.getElementById('warmup-steps')) document.getElementById('warmup-steps').value = config.warmup_steps || 500;
            if (document.getElementById('save-every')) document.getElementById('save-every').value = config.save_every || 1;

        } catch (e) {
            console.error('Failed to load config:', e);
        }
    }

    function resetConfig() {
        const defaults = {
            'num-blocks': 8,
            'num-heads': 8,
            'embedding-dim': 512,
            'max-seq-len': 256,
            'dropout': 0.0,
            'num-epochs': 75,
            'batch-size': 64,
            'learning-rate': 0.0011,
            'min-lr': 0.000005,
            'warmup-steps': 500,
            'save-every': 1
        };

        Object.keys(defaults).forEach(key => {
            const element = document.getElementById(key);
            if (element) element.value = defaults[key];
        });

        localStorage.removeItem('training_config');
        showTrainingInfo('Configuration reset to defaults', true);
    }

    function startStatusPolling() {
        if (statusInterval) return;

        statusInterval = setInterval(async () => {
            try {
                const response = await fetch('http://127.0.0.1:5000/api/training/status');
                const status = await response.json();

                updateStatusDisplay(status);

                if (!status.is_training && !status.is_paused) {
                    stopStatusPolling();
                }

            } catch (error) {
                console.error('Status poll error:', error);
            }
        }, 2000);
    }

    function stopStatusPolling() {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
    }

    function updateStatusDisplay(status) {
        if (!status.is_training && !status.is_paused) {
            trainingStatus.innerHTML = '<p style="color: var(--text-tertiary);">No active training</p>';
            return;
        }

        const state = status.is_paused ? 'Paused' : 'Training';
        const epochProgress = status.current_epoch && status.total_epochs
            ? (status.current_epoch / status.total_epochs * 100).toFixed(1)
            : 0;

        const html = `
            <div class="info-card">
                <div class="info-card-label">Status</div>
                <div class="info-card-value" style="color: ${status.is_paused ? 'var(--text-secondary)' : 'var(--accent-primary)'};">${state}</div>
            </div>
            <div class="info-card">
                <div class="info-card-label">Progress</div>
                <div class="info-card-value">
                    Epoch ${status.current_epoch || 0} / ${status.total_epochs || 0}
                    <div style="width: 100%; height: 4px; background: var(--bg-tertiary); border-radius: 2px; margin-top: 8px;">
                        <div style="width: ${epochProgress}%; height: 100%; background: var(--accent-primary); border-radius: 2px; transition: width 0.3s;"></div>
                    </div>
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-label">Loss</div>
                <div class="info-card-value">${status.current_loss?.toFixed(4) || 'N/A'}</div>
            </div>
            <div class="info-card">
                <div class="info-card-label">Learning Rate</div>
                <div class="info-card-value">${status.learning_rate?.toFixed(6) || 'N/A'}</div>
            </div>
        `;

        trainingStatus.innerHTML = html;
    }

    function showTrainingInfo(message, autoRemove = false) {
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'background: var(--bg-tertiary); border: 1px solid var(--border-primary); padding: 12px; border-radius: 8px; margin-bottom: 12px; color: var(--text-primary);';
        infoDiv.textContent = message;

        trainingStatus.innerHTML = '';
        trainingStatus.appendChild(infoDiv);

        if (autoRemove) {
            setTimeout(() => infoDiv.remove(), 3000);
        }
    }

    function showTrainingError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;

        trainingStatus.innerHTML = '';
        trainingStatus.appendChild(errorDiv);

        setTimeout(() => errorDiv.remove(), 5000);
    }

    async function loadDatasets() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/list');

            if (!response.ok) {
                console.warn('Could not load datasets');
                return;
            }

            const data = await response.json();

            if (data.datasets && data.datasets.length > 0) {
                data.datasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset.path;
                    option.textContent = `${dataset.name} (${dataset.examples} examples)`;
                    bpeDatasetSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.warn('Failed to load datasets:', error);
        }
    }

    async function setTikToken() {
        const tokenizerName = document.getElementById('tiktoken-name').value.trim();

        if (!tokenizerName) {
            showTokenizerError('Please enter a tokenizer name');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/api/tokenizers/set-tiktoken', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tokenizer_name: tokenizerName })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to set TikToken');
            }

            showTokenizerInfo(`TikToken set to ${tokenizerName}`, true);
        } catch (error) {
            console.error('Set TikToken error:', error);
            showTokenizerError(`Failed to set TikToken: ${error.message}`);
        }
    }

    async function loadBPE() {
        const fileInput = document.getElementById('bpe-file-input');
        const file = fileInput.files[0];

        if (!file) {
            showTokenizerError('Please select a tokenizer file');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/api/tokenizers/load-bpe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tokenizer_path: file.path })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to load BPE');
            }

            showTokenizerInfo('BPE tokenizer loaded successfully', true);
        } catch (error) {
            console.error('Load BPE error:', error);
            showTokenizerError(`Failed to load BPE: ${error.message}`);
        }
    }

    async function trainBPE() {
        const datasetPath = bpeDatasetSelect.value;
        const vocabSize = parseInt(document.getElementById('bpe-vocab-size').value);

        if (!datasetPath) {
            showTokenizerError('Please select a dataset');
            return;
        }

        // Create progress bar
        const progressHTML = `
            <div id="bpe-progress-container" style="margin-top: 12px;">
                <div style="color: var(--text-secondary); margin-bottom: 8px; font-size: 13px;">
                    <span id="bpe-progress-text">Starting BPE training...</span>
                </div>
                <div style="width: 100%; height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden;">
                    <div id="bpe-progress-bar" style="width: 0%; height: 100%; background: var(--accent-primary); transition: width 0.3s;"></div>
                </div>
                <div style="color: var(--text-tertiary); margin-top: 4px; font-size: 12px;" id="bpe-progress-details">
                    Merges: 0 / ${vocabSize - 256}
                </div>
            </div>
        `;

        tokenizerStatus.innerHTML = progressHTML;

        try {
            // Start BPE training (non-blocking)
            const response = await fetch('http://127.0.0.1:5000/api/tokenizers/train-bpe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_path: datasetPath,
                    vocab_size: vocabSize,
                    output_path: `artifacts/tokenizer/bpe_${vocabSize}.pkl`
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to start BPE training');
            }

            // Poll for progress
            const progressInterval = setInterval(async () => {
                try {
                    const statusResponse = await fetch('http://127.0.0.1:5000/api/tokenizers/bpe-progress');
                    const status = await statusResponse.json();

                    const progressBar = document.getElementById('bpe-progress-bar');
                    const progressText = document.getElementById('bpe-progress-text');
                    const progressDetails = document.getElementById('bpe-progress-details');

                    if (progressBar && status.current_merge !== undefined) {
                        const totalMerges = vocabSize - 256;
                        const progress = (status.current_merge / totalMerges) * 100;

                        progressBar.style.width = `${progress}%`;
                        progressText.textContent = `Training BPE tokenizer... ${progress.toFixed(1)}%`;
                        progressDetails.textContent = `Merges: ${status.current_merge} / ${totalMerges}`;

                        if (status.is_complete) {
                            clearInterval(progressInterval);
                            progressBar.style.width = '100%';
                            progressText.textContent = 'BPE tokenizer trained successfully!';
                            progressText.style.color = 'var(--success)';
                            progressDetails.textContent = `Saved to ${status.output_path || data.path}`;

                            setTimeout(() => {
                                showTokenizerInfo(`BPE tokenizer trained successfully! Saved to ${status.output_path || data.path}`, true);
                            }, 2000);
                        }

                        if (status.error) {
                            clearInterval(progressInterval);
                            throw new Error(status.error);
                        }
                    }
                } catch (pollError) {
                    console.error('Progress poll error:', pollError);
                }
            }, 500); // Poll every 500ms

        } catch (error) {
            console.error('Train BPE error:', error);
            showTokenizerError(`Failed to train BPE: ${error.message}`);
        }
    }

    function showTokenizerInfo(message, autoRemove = false) {
        const infoDiv = document.createElement('div');
        infoDiv.style.cssText = 'background: var(--bg-tertiary); border: 1px solid var(--border-primary); padding: 12px; border-radius: 8px; color: var(--text-primary);';
        infoDiv.textContent = message;

        tokenizerStatus.innerHTML = '';
        tokenizerStatus.appendChild(infoDiv);

        if (autoRemove) {
            setTimeout(() => infoDiv.remove(), 3000);
        }
    }

    function showTokenizerError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;

        tokenizerStatus.innerHTML = '';
        tokenizerStatus.appendChild(errorDiv);

        setTimeout(() => errorDiv.remove(), 5000);
    }

    function calculateParameters() {
        // Get model configuration values
        const numBlocks = parseInt(document.getElementById('num-blocks')?.value || 8);
        const numHeads = parseInt(document.getElementById('num-heads')?.value || 8);
        const embeddingDim = parseInt(document.getElementById('embedding-dim')?.value || 512);
        const vocabSize = parseInt(document.getElementById('vocab-size')?.value || 50257);
        const maxSeqLen = parseInt(document.getElementById('max-seq-len')?.value || 256);

        // Calculate parameters for a transformer model
        // Token embedding: vocab_size * embedding_dim
        const tokenEmbedding = vocabSize * embeddingDim;

        // Position embedding: max_seq_len * embedding_dim
        const positionEmbedding = maxSeqLen * embeddingDim;

        // Per transformer block:
        // - Multi-head attention: 4 * embedding_dim^2 (Q, K, V, O projections)
        // - Feed-forward: 2 * embedding_dim * (4 * embedding_dim) = 8 * embedding_dim^2
        // - Layer norms: 4 * embedding_dim (2 layer norms per block, each has 2 params per dim)
        const attentionParams = 4 * embeddingDim * embeddingDim;
        const ffnParams = 8 * embeddingDim * embeddingDim;
        const layerNormParams = 4 * embeddingDim;
        const paramsPerBlock = attentionParams + ffnParams + layerNormParams;
        const totalBlockParams = numBlocks * paramsPerBlock;

        // Final layer norm: 2 * embedding_dim
        const finalLayerNorm = 2 * embeddingDim;

        // Output projection (LM head): embedding_dim * vocab_size
        const outputProjection = embeddingDim * vocabSize;

        // Total parameters
        const totalParams = tokenEmbedding + positionEmbedding + totalBlockParams + finalLayerNorm + outputProjection;

        // Format the number with commas
        const formattedParams = totalParams.toLocaleString();

        // Display the result
        const paramDisplay = document.getElementById('param-display');
        const paramCount = document.getElementById('param-count');

        const millions = totalParams / 1_000_000;
        const billions = totalParams / 1_000_000_000;
        const trillions = totalParams / 1_000_000_000_000;

        if (trillions >= 1) {
            paramCount.innerHTML =
                `${formattedParams}<br><span style="font-size: 16px; color: var(--text-secondary);">(${trillions.toFixed(2)}T parameters)</span>`;
        } else if (billions >= 1) {
            paramCount.innerHTML =
                `${formattedParams}<br><span style="font-size: 16px; color: var(--text-secondary);">(${billions.toFixed(2)}B parameters)</span>`;
        } else {
            paramCount.innerHTML =
                `${formattedParams}<br><span style="font-size: 16px; color: var(--text-secondary);">(${millions.toFixed(2)}M parameters)</span>`;
        }

        paramDisplay.style.display = 'block';

        // Auto-hide after 10 seconds
        setTimeout(() => {
            paramDisplay.style.display = 'none';
        }, 10000);
    }
});
