# Contributing to Creator Catalyst

Thank you for your interest in contributing to Creator Catalyst! This document provides guidelines and information for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome new contributors and help them learn
- Focus on constructive feedback and collaboration
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots or error messages if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome. Please provide:

- A clear description of the proposed feature
- Use cases and benefits
- Potential implementation approach (if known)

### Pull Requests

We welcome contributions through pull requests! Some areas where you can help:

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- UI/UX improvements

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- FFmpeg (for video processing)

### Installation Steps

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Creator-Catalyst.git
   cd Creator-Catalyst
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. Run the application:
   ```bash
   streamlit run app/app.py
   ```

### Project Structure

```
Creator-Catalyst/
├── app/               # Streamlit application
├── src/
│   ├── core/          # Core AI logic
│   ├── database/      # Data persistence
│   ├── utils/         # Utility functions
│   └── config/        # Configuration
└── tests/             # Test files
```

## Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Write docstrings for functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Code Organization

- Keep modules focused on single responsibility
- Avoid circular imports
- Use relative imports for internal modules
- Keep the main application logic separate from UI code

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Use descriptive test names
- Mock external API calls in tests

## Commit Guidelines

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: add new feature for video processing
fix: resolve issue with SRT caption generation
docs: update README with installation instructions
refactor: improve database query performance
test: add unit tests for engagement scorer
```

### Commit Best Practices

- Make commits atomic (one logical change per commit)
- Write commit messages in imperative mood
- Reference related issues in commit messages
- Avoid including unrelated changes in a single commit

## Pull Request Process

### Before Submitting

1. Ensure your code follows the coding standards
2. Write/update tests as needed
3. Update documentation if applicable
4. Run tests and ensure they pass
5. Sync your branch with the latest main branch

### Submitting a PR

1. Create a descriptive title for your PR
2. Describe the changes and their purpose
3. Reference related issues (e.g., "Fixes #123")
4. Include screenshots for UI changes
5. Wait for code review feedback

### PR Review Process

- Maintainers will review your PR
- Address review comments promptly
- Make requested changes or discuss alternatives
- PRs need approval before merging

### After Merging

- Your contribution will be credited
- Consider helping review other PRs
- Join discussions in issues

## Getting Help

- Check existing documentation and issues
- Ask questions in GitHub Discussions
- Reach out to maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to Creator Catalyst!
