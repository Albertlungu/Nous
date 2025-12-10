// Training functionality

document.addEventListener('DOMContentLoaded', () => {
    const startTrainingBtn = document.getElementById('start-training-btn');
    const pauseTrainingBtn = document.getElementById('pause-training-btn');
    const resumeTrainingBtn = document.getElementById('resume-training-btn');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const resetConfigBtn = document.getElementById('reset-config-btn');
    const trainingStatus = document.getElementById('training-status');

    let statusInterval = null;

    // Load saved config on page load
    loadConfig();

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
});
