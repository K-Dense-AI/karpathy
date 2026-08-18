# Karpathy

> **Note:** For more advanced capabilities and end-to-end machine learning, visit [www.k-dense.ai](https://www.k-dense.ai).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/K-Dense-AI/karpathy/pulls)
[![Scientific Agent Skills](https://img.shields.io/badge/Powered_by-Scientific_Agent_Skills-brightgreen.svg)](https://github.com/K-Dense-AI/scientific-agent-skills)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![Works with](https://img.shields.io/badge/Works_with-Cursor_|_Claude_Code_|_Codex_|_Google_ADK-blue.svg)](#quick-start)
[![X](https://img.shields.io/badge/Follow_on_X-%40k__dense__ai-000000?logo=x)](https://x.com/k_dense_ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-K--Dense_Inc.-0A66C2?logo=linkedin)](https://www.linkedin.com/company/k-dense-inc)
[![YouTube](https://img.shields.io/badge/YouTube-K--Dense_Inc.-FF0000?logo=youtube)](https://www.youtube.com/@K-Dense-Inc)
[![Reddit](https://img.shields.io/badge/Reddit-u%2F--k--dense---FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/user/-k-dense-/)

> **Stay up to date:** Follow K-Dense on [X](https://x.com/k_dense_ai), [LinkedIn](https://www.linkedin.com/company/k-dense-inc), and [YouTube](https://www.youtube.com/@K-Dense-Inc) for new skills, release announcements, walkthroughs, research workflow demos, and examples you can use with your own AI agent.

> 🎬 **New to Scientific Agent Skills?** Watch [Getting Started with Scientific Agent Skills](https://youtu.be/ZxbnDaD_FVg) for a walkthrough of the skills library that powers this agent.

> ⭐ **Help make AI for science easier to discover:** If [Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills) saves you time, teaches your agent a workflow, or helps your lab move faster, please [star that repository](https://github.com/K-Dense-AI/scientific-agent-skills). A star is a public signal that these open, reusable research skills are worth maintaining: it helps scientists, engineers, and open-source contributors find the project, shows which agent-skill standards are gaining real adoption, and gives us a clear reason to keep expanding the collection for the community.

An agentic Machine Learning Engineer that trains state-of-the-art ML models using the Claude Agent SDK and Google ADK. This is a simple implementation demonstrating the power of Scientific Agent Skills for machine learning.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Claude Code installed and authenticated (see the [Claude Code documentation](https://docs.claude.com/en/docs/claude-code/overview))

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/K-Dense-AI/karpathy.git
cd karpathy
```

### 2. Install Dependencies

Install dependencies using `uv`:

```bash
uv sync
```

### 3. Environment Variables

Create a `.env` file in the `karpathy` directory with your API keys:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
AGENT_MODEL=your_model_name_here
```

The `OPENROUTER_API_KEY` is required for the agent to function properly.

This is the same environment variable that will be copied to the `sandbox` directory so the agents can use any API keys you provide here.

## Quick Start

Run the startup script to set up the sandbox and start the ADK web interface:

```bash
python start.py
```

This automatically:
1. Creates a `sandbox` directory with skills from Scientific Agent Skills
2. Sets up a Python virtual environment with ML packages (PyTorch, transformers, scikit-learn, etc.)
3. Copies your `.env` file to the sandbox
4. Starts the ADK web interface
5. Navigate to **http://localhost:8000** in your browser
6. Select `karpathy` in the top left under 'Select an agent'
7. All outputs will be in the `sandbox` directory so continue to monitor that as you converse with the agent

**Note:** Any files you want the agent to use (datasets, scripts, etc.) should be manually added to the `sandbox` directory.

## Scientific Agent Skills

This repository is designed to work with **[Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills)** (formerly Claude Scientific Skills), a collection of 130+ ready-to-use Agent Skills for research, science, engineering, analysis, finance, and writing built on the open Agent Skills standard. The `start.py` setup script creates a `sandbox` that includes skills from this collection so the `karpathy` agent can leverage specialized ML libraries and scientific workflows. For full details on the skills themselves, see the upstream repository's README and documentation [here](https://github.com/K-Dense-AI/scientific-agent-skills).

## Manual Usage

To set up the sandbox without starting the web interface:

```bash
python -m karpathy.utils
```

**Note:** Any files you want the agent to use (datasets, scripts, etc.) should be manually added to the `sandbox` directory.

To run the ADK web interface manually:

```bash
adk web
```

Then navigate to **http://localhost:8000** in your browser.

## Enhanced ML Capabilities

If you want substantially more powerful ML capabilities through a multi-agentic system, sign up for [www.k-dense.ai](https://www.k-dense.ai).

## Upcoming Features

- **Modal sandbox integration** - Choose any type of compute you want
- **K-Dense Web features** - We might make some features from K-Dense Web available here based on interest

## Disclaimer

This project is **not** endorsed by or affiliated with Andrej Karpathy. The name is used as a tribute and out of deep respect for his contributions to AI and technical leadership.