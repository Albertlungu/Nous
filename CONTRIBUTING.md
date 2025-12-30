# Contributing to Nous
**Hey everyone!**
First of all, I would like to thank you for visiting my repository. As you can tell, it isn't a very well-known one, which is why it makes it so much more exciting! This is a passion project of mine and every single person passing through here means a lot.

I am a single student working on this, and although it is quite fun and educative, at times, doing everything myself gets tiresome, and it starts feeling less like an interaction with the GitHub community, and more like a chore.

## Prerequisites
Nous is completely based on the JAX library, which I used to be able to have full control of how my model works. Because of this, there are a couple of requirements that you must be aware of before cloning the repository.

1. This repo is built to run on Python 3.10, where JAX-Metal actually works.

    If you are on Mac:

        ```bash
        chmod +x metal_setup.sh
        ./metal_setup.sh
        ```

    If you're using a Unix computer with a CUDA-enabled GPU (somehow) or are on a remote GPU:

        ```bash
        chmod +x cuda_setup.sh
        cuda_setup.sh
        ```

    And if you're on Windows:

        ```bash
        windows_setup.ps1
        ```

2. You need quite a bit of storage.

    As any solid ML project, you can expect to need around 5GB dedicated to this (if you want to install the Nous electron app). This is mostly for model weights and training data.

3. Most importantly, you must have access to a powerful computer if you want to be able to train.

    Again, due to the nature of ML, you need a monster of a PC to be able to train it seriously. For inference, it's not that much of an issue (on an M4 MBA, it works just fine).

## Ways to Contribute

There are many ways you can help make Nous better:

- **Bug Reports & Fixes**: Found something broken? Let me know or fix it yourself.
- **Feature Implementations**: New architectures, better MoE, vision model improvements, etc.
- **Dataset Contributions**: Help expand or improve training datasets.
- **Tokenizer Improvements**: Optimize BPE or add support for other tokenization methods.
- **Electron App UI/UX**: Make the desktop app more intuitive and beautiful.
- **Documentation**: Improve code comments, write tutorials, or expand the README.
- **Performance Optimizations**: JAX optimization, memory efficiency, faster training.
- **Testing**: Add unit tests, integration tests, or test on different platforms.

## Project Structure

Understanding the codebase will help you contribute effectively:

```
Nous/
├── src/
│   ├── transformer/        # Attention mechanisms, FFN, transformer blocks
│   ├── embeddings/         # Token and positional embeddings
│   ├── training/           # Training loops and loss functions
│   ├── tokenizer/          # BPE and TikToken implementations
│   ├── vision/             # Vision transformer (ViT) components
│   ├── optimizers/         # Adam optimizer implementation
│   └── utils/              # Data loaders, config, helper functions
├── electron-app/           # Desktop application (Node.js + Electron)
├── artifacts/
│   ├── models/             # Saved model checkpoints (.pkl)
│   └── tokenizer/          # Trained tokenizers
├── training_data/          # Datasets (not committed to repo)
└── concepts/               # Documentation and explanations
```

## Development Workflow

### Branch Naming
Please use descriptive branch names with these prefixes:
- `feature/your-feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/what-youre-documenting` - Documentation updates
- `refactor/what-youre-refactoring` - Code refactoring
- `perf/optimization-area` - Performance improvements

### Commit Messages
Write clear, concise commit messages. Prefixes like `feat:`, `fix:`, `docs:`, etc. are nice to have but not required. I'll be honest, I don't use them myself, so it would be hypocritical of me to require them from you. Just make your commits descriptive and understandable.

Examples:
```
Add rotary positional embeddings
Fix attention mask broadcasting issue
Update transformer architecture explanation
Optimize matrix multiplication in FFN
Simplify tokenizer encode method
```

### Before Submitting
1. Test your changes locally
2. Ensure code follows the style guidelines
3. Update documentation if needed
4. Make sure you didn't commit large files (.pkl, .txt datasets)

## Code Style & Standards

- **Follow PEP 8**: Python code should be clean and readable.
- **Type Hints**: Encouraged for function signatures, especially for complex types.
- **Docstrings**: Use clear docstrings for classes and non-trivial functions.
  ```python
  def expert_fwd(x: jnp.ndarray,
                 expert_params: dict,
                 activation='gelu'
                 ) -> jnp.ndarray:
      """
      Forward pass through a single expert (normal FFN)

      Args:
          x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
          expert_params (dict): Dictionary with W1, B1, W2, B2
          activation (str, optional): Activation, 'gelu' or 'relu'. Defaults to 'gelu'.

      Returns:
          jnp.ndarray: Output tensor (batch, seq_len, embedding_dim)
      ```
  For complex returns (tuples, dicts with multiple parts):
  ```python
  Returns:
      tuple
          - output (jnp.ndarray): MoE output (batch, seq_len, embedding_dim)
          - aux_loss (jnp.float16): Load balancing auxiliary loss
  ```
- **JAX Conventions**: Write pure functions when possible, use `jax.jit` appropriately.
- **Comments**: Explain complex mathematical operations and non-obvious design decisions.
- **No Generated Files**: Don't commit `*.pyc`, `__pycache__/`, model files, or datasets.

## Testing Your Changes

Before submitting a pull request, please test your changes:

### For Model/Training Changes
```bash
# Test inference
python src/main.py
# Select 'i' for inference and try a prompt

# Test training on a small batch (optional)
python src/main.py
# Select 't' for training, use small epoch count
```

### For Electron App Changes
```bash
cd electron-app
npm start
# Test all UI interactions and verify API communication
```

### Verify JAX Setup
Make sure JAX detects your GPU/Metal device:
```python
import jax
print(jax.devices())  # Should show GPU or Metal device
```

### Run Existing Tests
```bash
# If applicable
python src/tokenizer/test_tokenizer.py
```

## Submitting Pull Requests

### The Process
1. **Fork** the repository to your GitHub account
2. **Clone** your fork locally
3. **Create a branch** from `main` (or current development branch)
4. **Make your changes** with clear, focused commits
5. **Test** thoroughly
6. **Push** to your fork
7. **Open a Pull Request** with a clear description

### PR Guidelines
- **Keep PRs focused**: One feature or fix per PR makes review easier
- **Write a clear description**: Explain what you changed and why
- **Reference issues**: Use "Fixes #issue-number" or "Closes #issue-number" if applicable
- **Include examples**: For model changes, show before/after behavior or performance
- **Be patient**: I'm a single maintainer, so reviews might take a few days

### PR Template
```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
How did you test these changes?

## Related Issues
Fixes #issue-number (if applicable)
```

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- **System information**: OS, Python version, JAX version
- **Steps to reproduce**: Clear steps to trigger the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error logs**: Full error messages and stack traces
- **Code snippet**: Minimal code that reproduces the issue

### Feature Requests
For feature requests:
- **Describe the feature**: What do you want to add?
- **Use case**: Why is this useful?
- **Alternatives considered**: Other approaches you thought about
- **Willingness to implement**: Can you help build it?

### Performance Details
If you've been training the model, I'd love to gather performance data from different devices. Please share:
- **Hardware specs**: CPU/GPU model, RAM, Metal/CUDA version
- **Training performance**:
  - Time per epoch
  - Iterations per second
  - Memory usage
- **Configuration used**:
  - Batch size
  - Sequence length
  - Model size (embedding_dim, num_blocks, num_heads)
  - Dataset size
- **Any other details** that might affect training speed

This helps me understand how Nous performs across different hardware and optimize accordingly.

## Specific Contribution Areas

Here are areas where contributions would be especially valuable:

### High Priority
- **Performance optimization**: Faster training, better memory efficiency
- **Dataset handling**: More flexible data pipeline, streaming support
- **Testing suite**: Unit tests and integration tests
- **Error handling**: Better error messages and validation

### Architecture Improvements
- **Attention mechanisms**: Flash attention, sparse attention, etc.
- **MoE enhancements**: Better expert routing, load balancing
- **Vision models**: Complete ViT implementation, add other architectures
- **Multimodal fusion**: Better integration between text and vision

### UI/UX
- **Electron app improvements**: Better design, more features, settings page
- **Real-time training visualization**: Live loss curves, token generation preview
- **Model comparison tools**: Compare different checkpoints
- **Dataset management UI**: Better interface for dataset creation

### Documentation
- **Code documentation**: Add docstrings and comments
- **Tutorial notebooks**: Jupyter notebooks explaining components
- **Architecture diagrams**: Visual explanations of the model
- **Video tutorials**: Walkthrough of how things work

## Recognition

I truly appreciate every contribution:
- **All contributors** will be added to a Contributors section in the README
- **Significant contributions** get special mentions and thanks
- **Regular contributors** might get collaborator access to the repo

Your work helps make this project better for everyone, and I'm genuinely grateful.

## Questions & Communication

### How to Ask Questions
- **GitHub Discussions**: Best for general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Pull Request comments**: For questions about specific code changes

### Response Time
I'm a student working on this in my free time, so please be patient:
- Issues/PRs: Usually within 2-5 days
- Discussions: When I can
- Urgent bugs: I'll try to respond faster

If I haven't responded in a week, feel free to ping me politely.

## Things to Avoid

To keep the project maintainable, please avoid:

- **Breaking changes without discussion**: Open an issue first for major architectural changes
- **Heavy dependencies**: Avoid adding dependencies that conflict with JAX or bloat the project
- **Scope creep**: Keep features focused on the core LLM functionality
- **Uncommitted generated files**: No `*.pyc`, `__pycache__/`, `.pkl` models, or large datasets
- **Reformatting entire files**: Focus changes on relevant code only
- **Unrelated changes in one PR**: Keep each PR focused on one thing

## Code of Conduct

Be respectful, kind, and constructive. This is a learning project and everyone is here to improve their skills and knowledge. We're all learning together.

## Thank You

Thank you for considering contributing to Nous. Whether you're fixing a typo, adding a feature, or just exploring the code, your interest and effort mean a lot. Let's build something cool together!
