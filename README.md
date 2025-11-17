# Karpathy
An agentic Machine Learning Engineer

## Setup

### 1. Install and Authenticate Claude Code

For detailed installation and authentication instructions, visit: https://www.claude.com/product/claude-code

### 2. Environment Variables

Copy the `.env.example` file to `.env` in the `karpathy` sub-directory and fill in your API keys:

```bash
cp karpathy/.env.example karpathy/.env
```

Then edit `karpathy/.env` with your actual API key:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The `OPENROUTER_API_KEY` is required for the agent to function properly.

### 3. Install Dependencies

Make sure you have the required dependencies installed. If using `uv`:

```bash
uv sync
```

## Quick Start

To setup the sandbox and start the ADK web interface, simply run:

```bash
python start.py
```

This will automatically:
1. Setup the sandbox environment
2. Start the ADK web interface

## Manual Usage

If you want to run the ADK web interface manually without the setup script:

```bash
adk web
```

## Enhanced ML Capabilities

If you want substantially more powerful ML capabilities through a multi-agentic system, sign up for [www.k-dense.ai](https://www.k-dense.ai). Currently in closed beta, launching publicly in December 2025.

## Upcoming Features

- **Modal sandbox integration** - Choose any type of compute you want
- **K-Dense Web features** - We might make some features from K-Dense Web available here based on interest

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=K-Dense-AI/karpathy&type=date&legend=top-left)](https://www.star-history.com/#K-Dense-AI/karpathy&type=date&legend=top-left)