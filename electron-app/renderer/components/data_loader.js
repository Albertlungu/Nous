// Data Loader - handles dataset creation, combination, tokenization and file uploads

document.addEventListener('DOMContentLoaded', () => {
    const createDatasetBtn = document.getElementById('create-dataset-btn');
    const listDatasetsBtn = document.getElementById('list-datasets-btn');
    const combineDatasetBtn = document.getElementById('combine-datasets-btn');
    const removeDatasetBtn = document.getElementById('remove-dataset-btn');
    const datasetList = document.getElementById('dataset-list');
    const datasetInfo = document.getElementById('dataset-info');

    // Event listeners
    if (createDatasetBtn) {
        createDatasetBtn.addEventListener('click', () => {
            showCreateDatasetModal();
        });
    }

    if (listDatasetsBtn) {
        listDatasetsBtn.addEventListener('click', () => {
            loadAndDisplayDatasets();
        });
    }

    if (combineDatasetBtn) {
        combineDatasetBtn.addEventListener('click', () => {
            showCombineDatasetsModal();
        });
    }

    if (removeDatasetBtn) {
        removeDatasetBtn.addEventListener('click', () => {
            showRemoveDatasetModal();
        });
    }

    // Load datasets on page load
    loadAndDisplayDatasets();

    async function loadAndDisplayDatasets() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/list');

            if (!response.ok) {
                throw new Error('Failed to load datasets');
            }

            const data = await response.json();
            displayDatasets(data.datasets || []);
        } catch (e) {
            console.error('Load dataset error:', e);
            showError(`Failed to load datasets: ${e.message}`);
        }
    }

    function displayDatasets(datasets) {
        if (!datasets || datasets.length === 0) {
            datasetList.innerHTML = '<p style="color: var(--text-tertiary); text-align: center; padding: 20px;">No datasets available</p>';
            return;
        }

        const html = datasets.map(dataset => `
            <div class="list-item" data-path="${dataset.path}">
                <div>
                    <div class="list-item-title">${dataset.name}</div>
                    <div class="list-item-meta">${dataset.examples} examples • ${dataset.size || 'Unknown size'}</div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="mini-btn preview-btn" data-path="${dataset.path}">Preview</button>
                    <button class="mini-btn tokenize-btn" data-path="${dataset.path}">Tokenize</button>
                </div>
            </div>
        `).join('');

        datasetList.innerHTML = html;

        document.querySelectorAll('.preview-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                previewDataset(btn.dataset.path);
            });
        });

        document.querySelectorAll('.tokenize-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                showTokenizeModal(btn.dataset.path);
            });
        });
    }

    function showCreateDatasetModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Create Dataset</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <h4>Dataset Source</h4>

                        <div class="source-tabs">
                            <button class="source-tab active" data-source="preset">Preset Datasets</button>
                            <button class="source-tab" data-source="huggingface">HuggingFace</button>
                            <button class="source-tab" data-source="file">Local Files</button>
                        </div>

                        <div id="preset-source" class="source-panel active">
                            <div class="checkbox-grid">
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="alpaca">
                                    <span>Alpaca (51,974 examples)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="wizardlm">
                                    <span>WizardLM (70,004 examples)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="flan">
                                    <span>FLAN (50,000 examples)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="gpt-teacher">
                                    <span>GPT Teacher (89,260 examples)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="dolly">
                                    <span>Dolly 15k (15,000 examples)</span>
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" name="preset-dataset" value="trivia-qa">
                                    <span>TriviaQA (20,000 examples)</span>
                                </label>
                            </div>
                        </div>

                        <div id="huggingface-source" class="source-panel">
                            <label style="display: flex; flex-direction: column; gap: 8px;">
                                <span>HuggingFace Dataset Path:</span>
                                <input type="text" id="hf-dataset-path" placeholder="e.g., tatsu-lab/alpaca"
                                    style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                            </label>
                            <label style="display: flex; flex-direction: column; gap: 8px; margin-top: 12px;">
                                <span>Split (optional):</span>
                                <input type="text" id="hf-split" value="train" placeholder="train"
                                    style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                            </label>
                            <label style="display: flex; flex-direction: column; gap: 8px; margin-top: 12px;">
                                <span>Max Examples (0 for all):</span>
                                <input type="number" id="hf-max-examples" value="0" min="0"
                                    style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                            </label>
                        </div>

                        <div id="file-source" class="source-panel">
                            <div style="border: 2px dashed var(--border-primary); border-radius: 8px; padding: 32px; text-align: center; cursor: pointer; transition: all 0.2s;" id="file-drop-zone">
                                <div style="font-size: 48px; margin-bottom: 12px;">📁</div>
                                <div style="color: var(--text-primary); margin-bottom: 8px;">Drop .txt or .pkl files here or click to browse</div>
                                <div style="color: var(--text-tertiary); font-size: 12px;">Files will be imported as-is to training_data/</div>
                                <input type="file" id="file-input" accept=".txt,.pkl" multiple style="display: none;">
                            </div>
                            <div id="selected-files" style="margin-top: 12px;"></div>
                        </div>
                    </div>

                    <div class="form-section" id="format-section">
                        <h4>Format Template</h4>
                        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                            <button class="format-preset-btn" data-format="instruction">Instruction/Input/Output</button>
                            <button class="format-preset-btn" data-format="qa">Question/Answer</button>
                            <button class="format-preset-btn" data-format="conversation">Conversation</button>
                            <button class="format-preset-btn" data-format="custom">Custom</button>
                        </div>
                        <textarea id="format-template" rows="6" placeholder="Instruction: {instruction}&#10;Input: {input}&#10;Output: {output}&#10;&#10;Available placeholders: {instruction}, {input}, {output}, {question}, {answer}, {context}, {response}"
                            style="width: 100%; background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 12px; color: var(--text-primary); font-family: monospace; font-size: 13px; resize: vertical;">Instruction: {instruction}
Input: {input}
Output: {output}
</textarea>
                        <div style="margin-top: 8px; font-size: 12px; color: var(--text-secondary);">
                            Use {field_name} placeholders. Leave fields empty if not in your dataset.
                        </div>
                    </div>

                    <div class="form-section" id="output-section">
                        <h4>Output Settings</h4>
                        <label style="display: flex; flex-direction: column; gap: 8px;">
                            <span>Output Filename:</span>
                            <input type="text" id="output-filename" placeholder="my_dataset.txt"
                                style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                        </label>
                    </div>

                    <div id="dataset-progress" style="display: none; margin-top: 16px;">
                        <div style="color: var(--text-secondary); margin-bottom: 8px;" id="progress-text">Processing...</div>
                        <div style="width: 100%; height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden;">
                            <div id="progress-bar" style="width: 0%; height: 100%; background: var(--accent-primary); transition: width 0.3s;"></div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="config-btn secondary modal-cancel">Cancel</button>
                    <button class="config-btn" id="create-dataset-confirm">Create Dataset</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
        modal._escHandler = escHandler;

        // Auto-select text on focus
        modal.querySelectorAll('input[type="text"], input[type="number"]').forEach(input => {
            input.addEventListener('focus', function() { this.select(); });
        });

        // Source tab switching
        modal.querySelectorAll('.source-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                modal.querySelectorAll('.source-tab').forEach(t => t.classList.remove('active'));
                modal.querySelectorAll('.source-panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                modal.querySelector(`#${tab.dataset.source}-source`).classList.add('active');

                // Hide format template and output for file uploads
                const formatSection = modal.querySelector('#format-section');
                const outputSection = modal.querySelector('#output-section');
                if (tab.dataset.source === 'file') {
                    formatSection.style.display = 'none';
                    outputSection.style.display = 'none';
                } else {
                    formatSection.style.display = 'block';
                    outputSection.style.display = 'block';
                }
            });
        });

        // Format preset buttons
        modal.querySelectorAll('.format-preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.dataset.format;
                const textarea = modal.querySelector('#format-template');

                const formats = {
                    instruction: 'Instruction: {instruction}\nInput: {input}\nOutput: {output}\n',
                    qa: 'Question: {question}\nAnswer: {answer}\n',
                    conversation: '{prompt}\n{response}\n',
                    custom: ''
                };

                textarea.value = formats[format] || '';
            });
        });

        // File drop zone
        const dropZone = modal.querySelector('#file-drop-zone');
        const fileInput = modal.querySelector('#file-input');
        const selectedFilesDiv = modal.querySelector('#selected-files');
        let selectedFiles = [];

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--accent-primary)';
            dropZone.style.background = 'var(--bg-hover)';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-primary)';
            dropZone.style.background = 'transparent';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-primary)';
            dropZone.style.background = 'transparent';

            const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.txt') || f.name.endsWith('.pkl'));
            selectedFiles = files;
            displaySelectedFiles();
        });

        fileInput.addEventListener('change', (e) => {
            selectedFiles = Array.from(e.target.files);
            displaySelectedFiles();
        });

        function displaySelectedFiles() {
            if (selectedFiles.length === 0) {
                selectedFilesDiv.innerHTML = '';
                return;
            }

            selectedFilesDiv.innerHTML = `
                <div style="color: var(--text-primary); font-size: 13px; margin-bottom: 8px;">Selected files:</div>
                ${selectedFiles.map(f => `
                    <div style="background: var(--bg-tertiary); padding: 8px 12px; border-radius: 6px; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px;">
                        ${f.name} (${(f.size / 1024).toFixed(1)} KB)
                    </div>
                `).join('')}
            `;
        }

        // Close modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.querySelector('.modal-cancel').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        });

        // Create dataset
        modal.querySelector('#create-dataset-confirm').addEventListener('click', async () => {
            await createDataset(modal, selectedFiles);
        });
    }

    async function createDataset(modal, selectedFiles) {
        const activeSource = modal.querySelector('.source-tab.active').dataset.source;
        const formatTemplate = modal.querySelector('#format-template').value;
        const outputFilename = modal.querySelector('#output-filename').value.trim();

        // For file uploads, we don't need format template or output filename
        if (activeSource !== 'file' && !outputFilename) {
            showModalError(modal, 'Please enter an output filename');
            return;
        }

        const progressDiv = modal.querySelector('#dataset-progress');
        const progressBar = modal.querySelector('#progress-bar');
        const progressText = modal.querySelector('#progress-text');

        progressDiv.style.display = 'block';

        try {
            let requestData = {
                format_template: formatTemplate,
                output_filename: outputFilename
            };

            if (activeSource == 'preset') {
                const selectedPresets = Array.from(modal.querySelectorAll('input[name="preset-dataset"]:checked')).map(cb => cb.value);

                if (selectedPresets.length === 0) {
                    throw new Error('Please select at least one preset dataset');
                }

                requestData.source_type = 'preset';
                requestData.presets = selectedPresets;
            } else if (activeSource === 'huggingface') {
                const hfPath = modal.querySelector('#hf-dataset-path').value.trim();
                const hfSplit = modal.querySelector('#hf-split').value.trim() || 'train';
                const maxExamples = parseInt(modal.querySelector('#hf-max-examples').value) || 0;

                if (!hfPath) {
                    throw new Error('Please enter a HuggingFace dataset path');
                }

                requestData.source_type = 'huggingface';
                requestData.hf_path = hfPath;
                requestData.hf_split = hfSplit;
                requestData.max_examples = maxExamples;
            } else if (activeSource === 'file') {
                if (selectedFiles.length === 0) {
                    throw new Error('Please select at least one file')
                }

                // For file uploads, use FormData to upload files directly
                const formData = new FormData();
                selectedFiles.forEach(file => {
                    formData.append('files', file);
                });

                progressText.textContent = 'Importing files...';

                const response = await fetch('http://127.0.0.1:5000/api/datasets/import-files', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to import files');
                }

                progressBar.style.width = '100%';
                progressText.textContent = `Files imported successfully!`;
                progressText.style.color = 'var(--success)';

                setTimeout(() => {
                    modal.remove();
                    document.removeEventListener('keydown', escHandler);
                    loadAndDisplayDatasets();
                }, 2000);
                return;
            }

            progressText.textContent = 'Creating dataset...';

            const response = await fetch('http://127.0.0.1:5000/api/datasets/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to create dataset');
            }

            progressBar.style.width = '100%';
            progressText.textContent = `Dataset created successfully! (${data.examples} examples)`;
            progressText.style.color = 'var(--success)';

            setTimeout(() => {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
                loadAndDisplayDatasets();
            }, 2000);
        } catch (e) {
            console.error('Create dataset error:', e);
            showModalError(modal, e.message);
            progressDiv.style.display = 'none';
        }
    }

    function showCombineDatasetsModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Combine Datasets</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <h4>Select Datasets to Combine</h4>
                        <div id="combine-dataset-list" style="max-height: 300px; overflow-y: auto;">
                            <p style="color: var(--text-tertiary); text-align: center;">Loading datasets...</p>
                        </div>
                    </div>

                    <div class="form-section">
                        <h4>Output Settings</h4>
                        <label style="display: flex; flex-direction: column; gap: 8px;">
                            <span>Combined Dataset Name:</span>
                            <input type="text" id="combined-output-name" placeholder="combined_dataset.txt"
                                style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                        </label>
                        <label style="display: flex; align-items: center; gap: 8px; margin-top: 12px;">
                            <input type="checkbox" id="shuffle-combined">
                            <span>Shuffle combined dataset</span>
                        </label>
                    </div>

                    <div id="combine-info" style="display: none; margin-top: 16px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px;">
                        <div style="color: var(--text-secondary); font-size: 13px;">Total examples: <span id="total-examples" style="color: var(--accent-primary); font-weight: 500;">0</span></div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="config-btn secondary modal-cancel">Cancel</button>
                    <button class="config-btn" id="combine-datasets-confirm">Combine Datasets</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
        modal._escHandler = escHandler;

        // Auto-select text on focus
        modal.querySelector('#combined-output-name').addEventListener('focus', function() { this.select(); });

        // Load datasets for combining
        loadDatasetsForCombine(modal);

        // Close modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.querySelector('.modal-cancel').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        });

        // Combine datasets
        modal.querySelector('#combine-datasets-confirm').addEventListener('click', async () => {
            await combineDatasets(modal);
        });
    }


    async function loadDatasetsForCombine(modal) {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/list');
            const data = await response.json();

            const listDiv = modal.querySelector('#combine-dataset-list');
            const combineInfo = modal.querySelector('#combine-info');
            const totalExamplesSpan = modal.querySelector('#total-examples');

            if (!data.datasets || data.datasets.length === 0) {
                listDiv.innerHTML = '<p style="color: var(--text-tertiary); text-align: center;">No datasets available</p>';
                return;
            }

            listDiv.innerHTML = data.datasets.map(dataset => `
                <label class="checkbox-label" style="display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 8px; cursor: pointer;">
                    <input type="checkbox" name="combine-dataset" value="${dataset.path}" data-examples="${dataset.examples}">
                    <div style="flex: 1;">
                        <div style="color: var(--text-primary); font-weight: 500;">${dataset.name}</div>
                        <div style="color: var(--text-tertiary); font-size: 12px;">${dataset.examples} examples</div>
                    </div>
                </label>
            `).join('');

            modal.querySelectorAll('input[name="combine-dataset"]').forEach(cb => {
                cb.addEventListener('change', () => {
                    const checked = modal.querySelectorAll('input[name="combine-dataset"]:checked');
                    const total = Array.from(checked).reduce((sum, cb) => sum + parseInt(cb.dataset.examples), 0);

                    if (checked.length > 0) {
                        combineInfo.style.display = 'block';
                        totalExamplesSpan.textContent = total.toLocaleString();
                    } else {
                        combineInfo.style.display = 'none';
                    }
                });
            });
        } catch (e) {
            console.error('Load datasets error:', e);
            modal.querySelector('#combine-dataset-list').innerHTML = '<p style="color: var(--error); text-align: center;">Failed to load datasets</p>';

        }
    }

    async function combineDatasets(modal) {
        const selectedDatasets = Array.from(modal.querySelectorAll('input[name="combine-dataset"]:checked')).map(cb => cb.value);
        const outputName = modal.querySelector('#combined-output-name').value.trim();
        const shuffle = modal.querySelector('#shuffle-combined').checked;

        if (selectedDatasets.length < 2) {
            showModalError(modal, 'Please select at least 2 datasets to combine');
            return;
        }

        if (!outputName) {
            showModalError(modal, 'Please enter an output filename');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/combine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_paths: selectedDatasets,
                    output_filename: outputName,
                    shuffle: shuffle
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to combine datasets');
            }

            showModalSuccess(modal, `Datasets combined successfully! (${data.total_examples} examples)`);

            setTimeout(() => {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
                loadAndDisplayDatasets();
            }, 2000);

        } catch (error) {
            console.error('Combine datasets error:', error);
            showModalError(modal, error.message);
        }
    }

    function showTokenizeModal(datasetPath) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Tokenize Dataset</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <h4>Dataset</h4>
                        <div style="background: var(--bg-tertiary); padding: 12px; border-radius: 8px; color: var(--text-secondary);">
                            ${datasetPath.split('/').pop()}
                        </div>
                    </div>

                    <div class="form-section">
                        <h4>Tokenizer Type</h4>
                        <div style="display: flex; align-items: center; justify-content: center; gap: 16px; padding: 16px; background: var(--bg-tertiary); border-radius: 8px;">
                            <span class="toggle-label">TikToken</span>
                            <label class="toggle-switch">
                                <input type="checkbox" id="tokenize-tokenizer-toggle">
                                <span class="toggle-slider"></span>
                            </label>
                            <span class="toggle-label">BPE</span>
                        </div>
                    </div>

                    <div id="tokenize-tiktoken" class="form-section">
                        <h4>TikToken Settings</h4>
                        <label style="display: flex; flex-direction: column; gap: 8px;">
                            <span>Tokenizer Name:</span>
                            <input type="text" id="tokenize-tiktoken-name" value="gpt2" placeholder="e.g., gpt2, cl100k_base"
                                style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                        </label>
                    </div>

                    <div id="tokenize-bpe" class="form-section" style="display: none;">
                        <h4>BPE Settings</h4>
                        <label style="display: flex; flex-direction: column; gap: 8px;">
                            <span>Select BPE Tokenizer File:</span>
                            <input type="file" id="tokenize-bpe-file" accept=".pkl"
                                style="padding: 8px; color: var(--text-secondary);">
                        </label>
                    </div>

                    <div class="form-section">
                        <h4>Output Settings</h4>
                        <label style="display: flex; flex-direction: column; gap: 8px;">
                            <span>Output Filename:</span>
                            <input type="text" id="tokenize-output-name" placeholder="tokenized_dataset.pkl"
                                style="background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 8px; padding: 10px; color: var(--text-primary);">
                        </label>
                    </div>

                    <div id="tokenize-progress" style="display: none; margin-top: 16px;">
                        <div style="color: var(--text-secondary); margin-bottom: 8px;" id="tokenize-progress-text">Tokenizing...</div>
                        <div style="width: 100%; height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden;">
                            <div id="tokenize-progress-bar" style="width: 0%; height: 100%; background: var(--accent-primary); transition: width 0.3s;"></div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="config-btn secondary modal-cancel">Cancel</button>
                    <button class="config-btn" id="tokenize-confirm">Tokenize Dataset</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
        modal._escHandler = escHandler;

        // Auto-select text on focus
        modal.querySelectorAll('input[type="text"]').forEach(input => {
            input.addEventListener('focus', function() { this.select(); });
        });

        // Toggle tokenizer type
        const tokenizerToggle = modal.querySelector('#tokenize-tokenizer-toggle');
        const tiktokenDiv = modal.querySelector('#tokenize-tiktoken');
        const bpeDiv = modal.querySelector('#tokenize-bpe');
        const toggleLabels = modal.querySelectorAll('.toggle-label');

        tokenizerToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                tiktokenDiv.style.display = 'none';
                bpeDiv.style.display = 'block';
                toggleLabels[0].classList.remove('active');
                toggleLabels[1].classList.add('active');
            } else {
                tiktokenDiv.style.display = 'block';
                bpeDiv.style.display = 'none';
                toggleLabels[0].classList.add('active');
                toggleLabels[1].classList.remove('active');
            }
        });

        toggleLabels[0].classList.add('active');

        // Close modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.querySelector('.modal-cancel').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        });

        // Tokenize dataset
        modal.querySelector('#tokenize-confirm').addEventListener('click', async () => {
            await tokenizeDataset(modal, datasetPath);
        });
    }

    async function tokenizeDataset(modal, datasetPath) {
        const tokenizerType = modal.querySelector('#tokenize-tokenizer-toggle').checked ? 'bpe' : 'tiktoken';
        const outputName = modal.querySelector('#tokenize-output-name').value.trim();

        if (!outputName) {
            showModalError(modal, 'Please enter an output filename');
            return;
        }

        const progressDiv = modal.querySelector('#tokenize-progress');
        const progressBar = modal.querySelector('#tokenize-progress-bar');
        const progressText = modal.querySelector('#tokenize-progress-text');

        progressDiv.style.display = 'block';

        try {
            let requestData = {
                dataset_path: datasetPath,
                output_filename: outputName,
                tokenizer_type: tokenizerType
            };

            if (tokenizerType === 'tiktoken') {
                const tokenizerName = modal.querySelector('#tokenize-tiktoken-name').value.trim();
                if (!tokenizerName) {
                    throw new Error('Please enter a TikToken tokenizer name');
                }
                requestData.tokenizer_name = tokenizerName;
            } else {
                const bpeFile = modal.querySelector('#tokenize-bpe-file').files[0];
                if (!bpeFile) {
                    throw new Error('Please select a BPE tokenizer file');
                }
                requestData.tokenizer_path = bpeFile.path;
            }

            progressText.textContent = 'Tokenizing dataset... This may take a while.';

            const response = await fetch('http://127.0.0.1:5000/api/datasets/tokenize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to tokenize dataset');
            }

            progressBar.style.width = '100%';
            progressText.textContent = `Dataset tokenized successfully! Saved to ${data.output_path}`;
            progressText.style.color = 'var(--success)';

            setTimeout(() => {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }, 3000);

        } catch (error) {
            console.error('Tokenize dataset error:', error);
            showModalError(modal, error.message);
            progressDiv.style.display = 'none';
        }
    }

    async function previewDataset(datasetPath) {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_path: datasetPath, num_examples: 3 })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to preview dataset');
            }

            showPreviewModal(datasetPath, data.examples);
        } catch (e) {
            console.error('Preview dataset error:', e);
            showError(`Failed to preview dataset: ${e.message}`);
        }
    }

    function showPreviewModal(datasetPath, examples) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 800px;">
                <div class="modal-header">
                    <h3>Dataset Preview</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div style="background: var(--bg-tertiary); padding: 12px; border-radius: 8px; margin-bottom: 16px; color: var(--text-secondary);">
                        ${datasetPath.split('/').pop()}
                    </div>
                    ${examples.map((ex, i) => `
                        <div style="background: var(--bg-tertiary); padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                            <div style="color: var(--accent-primary); font-size: 12px; font-weight: 600; margin-bottom: 8px;">EXAMPLE ${i + 1}</div>
                            <pre style="white-space: pre-wrap; font-family: monospace; font-size: 13px; color: var(--text-primary); margin: 0;">${ex}</pre>
                        </div>
                    `).join('')}
                </div>
                <div class="modal-footer">
                    <button class="config-btn" onclick="this.closest('.modal-overlay').remove()">Close</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
        modal._escHandler = escHandler;

        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        });
    }

    function showRemoveDatasetModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Remove Dataset</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-section">
                        <h4>Select Dataset to Remove</h4>
                        <div id="remove-dataset-list" style="max-height: 400px; overflow-y: auto;">
                            <p style="color: var(--text-tertiary); text-align: center;">Loading datasets...</p>
                        </div>
                    </div>
                    <div style="padding: 12px; background: rgba(244, 67, 54, 0.1); border: 1px solid var(--error); border-radius: 8px; color: var(--error); font-size: 13px;">
                        Warning: This action cannot be undone. The dataset file will be permanently deleted.
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="config-btn secondary modal-cancel">Cancel</button>
                    <button class="config-btn" style="background: var(--error);" id="remove-dataset-confirm">Remove Dataset</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // ESC key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
        modal._escHandler = escHandler;

        // Load datasets for removal
        loadDatasetsForRemoval(modal);

        // Close modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.querySelector('.modal-cancel').addEventListener('click', () => {
            modal.remove();
            if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
            }
        });

        // Remove dataset
        modal.querySelector('#remove-dataset-confirm').addEventListener('click', async () => {
            await removeDataset(modal);
        });
    }

    async function loadDatasetsForRemoval(modal) {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/list');
            const data = await response.json();

            const listDiv = modal.querySelector('#remove-dataset-list');

            if (!data.datasets || data.datasets.length === 0) {
                listDiv.innerHTML = '<p style="color: var(--text-tertiary); text-align: center;">No datasets available</p>';
                return;
            }

            listDiv.innerHTML = data.datasets.map(dataset => `
                <label class="checkbox-label" style="display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 8px; cursor: pointer;">
                    <input type="radio" name="remove-dataset" value="${dataset.path}">
                    <div style="flex: 1;">
                        <div style="color: var(--text-primary); font-weight: 500;">${dataset.name}</div>
                        <div style="color: var(--text-tertiary); font-size: 12px;">${dataset.path}</div>
                    </div>
                </label>
            `).join('');

        } catch (error) {
            console.error('Load datasets error:', error);
            modal.querySelector('#remove-dataset-list').innerHTML =
                '<p style="color: var(--error); text-align: center;">Failed to load datasets</p>';
        }
    }

    async function removeDataset(modal) {
        const selected = modal.querySelector('input[name="remove-dataset"]:checked');

        if (!selected) {
            showModalError(modal, 'Please select a dataset to remove');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/api/datasets/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_path: selected.value })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to remove dataset');
            }

            showModalSuccess(modal, 'Dataset removed successfully');

            setTimeout(() => {
                modal.remove();
                if (modal._escHandler) document.removeEventListener('keydown', modal._escHandler);
                loadAndDisplayDatasets();
            }, 1500);

        } catch (error) {
            console.error('Remove dataset error:', error);
            showModalError(modal, error.message);
        }
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000; max-width: 400px;';

        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    function showModalError(modal, message) {
        let errorDiv = modal.querySelector('.modal-error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'modal-error error-message';
            errorDiv.style.marginTop = '16px';
            modal.querySelector('.modal-body').appendChild(errorDiv);
        }
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }

    function showModalSuccess(modal, message) {
        let successDiv = modal.querySelector('.modal-success');
        if (!successDiv) {
            successDiv = document.createElement('div');
            successDiv.className = 'modal-success';
            successDiv.style.cssText = 'background: rgba(76, 175, 80, 0.1); border: 1px solid var(--success); color: var(--success); padding: 12px; border-radius: 8px; margin-top: 16px;';
            modal.querySelector('.modal-body').appendChild(successDiv);
        }
        successDiv.textContent = message;
        successDiv.style.display = 'block';
    }
});
