# Comprehensive Codebase Study: Karpathy Project

## Project Overview: Karpathy

This is an **autonomous ML Engineer system** that can design, train, and improve machine learning models by orchestrating multiple specialized AI agents. It's a tribute to Andrej Karpathy (though not affiliated) and demonstrates Claude's scientific capabilities for machine learning tasks.

---

## Core Architecture

The project uses a **multi-agent orchestration** pattern with 4 key layers:

### 1. Main Agent Layer (`karpathy/agent.py`)
- High-level orchestrator using Google ADK's `LlmAgent`
- Uses Gemini 3 Pro Preview by default (configurable via OpenRouter)
- Single tool: `delegate_task` - delegates work to specialized expert agents

### 2. Tool/Delegation Layer (`karpathy/tools.py`)
- Bridges Google ADK ↔ Claude Code SDK
- The `delegate_task()` function translates strategic decisions into code execution
- Uses async streaming for real-time progress updates

### 3. Expert Agent Layer (defined in `instructions.yaml`)
- 12+ predefined specialists:
  - **Planning**: Plan Creator, Plan Reviewer, Code Planner
  - **Research**: Research Agent (with perplexity search)
  - **Data**: Data Discoverer, Data Engineer
  - **Execution**: Experiment Manager, Code Writer, Code Executor, Evaluation Agent
  - **Infrastructure**: Infra & Modal Operator

### 4. Sandbox Environment (`sandbox/`)
- Isolated workspace with:
  - **120+ scientific skills** from `K-Dense-AI/claude-scientific-skills`
  - **Python venv** with 11 ML packages (PyTorch, transformers, scikit-learn, pandas, etc.)
  - **Example dataset**: `Iris.csv` (150 rows, 5 columns)

---

## Directory Structure

```
D:\Projects\karpathy/
├── karpathy/                    # Main package source code
│   ├── __init__.py             # Package exports (exposes root_agent)
│   ├── agent.py                # Main agent definition (LlmAgent configuration)
│   ├── tools.py                # delegate_task function for Claude Code integration
│   ├── utils.py                # Setup utilities (sandbox, skills, environment)
│   ├── instructions.yaml       # Agent behavior instructions (main & common)
│   ├── .env.example            # Template for API keys configuration
│   └── .env                    # Actual API keys (gitignored)
├── sandbox/                     # Isolated execution environment
│   ├── .claude/                # Claude Code configuration
│   │   └── skills/             # 120+ scientific skills from claude-scientific-skills
│   ├── .venv/                  # Python virtual environment with ML packages
│   ├── .env                    # Copied from karpathy/.env for agent access
│   └── Iris.csv                # Example dataset (150 rows)
├── start.py                     # Main entry point (setup + launch ADK web)
├── pyproject.toml              # Project dependencies and metadata
├── uv.lock                     # Locked dependencies (uv package manager)
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git exclusions
└── .python-version             # Python 3.13 requirement
```

---

## Key Files Explained

### Entry Point

**`start.py`**
- **Purpose**: Simple startup orchestrator
- **Functionality**:
  1. Calls `setup_sandbox()` to prepare the execution environment
  2. Launches the ADK web interface at `http://localhost:8000`
- **Key Features**:
  - Error handling with proper exit codes
  - Keyboard interrupt handling for graceful shutdown
- **Intuition**: "One-command setup" that abstracts environment complexity

---

### Main Agent Definition

**`karpathy/agent.py`**
- **Purpose**: Defines the main orchestrator agent using Google ADK's `LlmAgent` class

**Key Components**:
```python
main_agent = LlmAgent(
    name="MainAgent",
    model=LiteLlm(model=MODEL),  # Uses OpenRouter for model flexibility
    description="The main agent that makes sure ML requests are successfully fulfilled",
    instruction=load_instructions("main_agent"),  # From instructions.yaml
    tools=[delegate_task],  # Single tool: delegate to Claude Code experts
    output_key="final_output"
)
```

**Configuration**:
- Model: Configured via `AGENT_MODEL` env var (default: `openrouter/google/gemini-3-pro-preview`)
- Uses LiteLLM for multi-provider model support through OpenRouter
- Single tool: `delegate_task` for orchestrating sub-agents

**Intuition**: This is the "brain" that decides how to break down user requests into concrete ML tasks and which experts to delegate to. It operates at a high strategic level, not executing code directly.

---

### Delegation Bridge

**`karpathy/tools.py`**
- **Purpose**: Provides the `delegate_task` function that bridges Google ADK agents to Claude Code SDK for actual code execution

**Key Function**:
```python
async def delegate_task(
    prompt: str,
    append_system_prompt: str
) -> dict
```

**How It Works**:
1. Takes a task description (`prompt`) and expert role description (`append_system_prompt`)
2. Combines with common instructions from `instructions.yaml`
3. Creates a Claude Code SDK query with:
   - System preset: `"claude_code"` (Claude's built-in coding expertise)
   - Working directory: `sandbox` (isolated environment)
   - Permission mode: `"bypassPermissions"` (automated execution)
4. Streams results, printing tool usage for observability
5. Returns structured result with completion status

**Architecture Pattern**: **Delegation adapter** pattern - translates high-level agent decisions into low-level Claude Code execution requests.

**Intuition**: Acts as the "communication protocol" between strategic planning (ADK agents) and tactical execution (Claude Code). Like a project manager translating business requirements into developer tasks.

---

### Environment Setup

**`karpathy/utils.py`**
- **Purpose**: Provides utilities for setting up the sandbox environment with scientific skills and ML packages

**Key Functions**:

**1. `load_instructions(agent_name: str) -> str`**
- Loads agent behavior instructions from `instructions.yaml`
- Used by both main agent and common instructions

**2. `download_scientific_skills(...)`**
- Clones the `K-Dense-AI/claude-scientific-skills` GitHub repository
- Extracts 120+ skills from the `scientific-skills/` folder
- Places them in `sandbox/.claude/skills/` for Claude Code to discover
- Uses temporary directory for clean cloning

**Scientific Skills Categories**:
- Chemistry & Molecular (60+ formats): pdb, mol2, sdf, xyz, gro, etc.
- Bioinformatics (50+ formats): fasta, fastq, bam, vcf, bed, gtf, etc.
- Microscopy & Imaging (45+ formats): tiff, czi, svs, dicom, etc.
- Spectroscopy & Analytical: NMR, mass spec, chromatography formats
- Proteomics & Metabolomics: mzML, mzXML, pepXML, etc.
- General Scientific: HDF5, NetCDF, MATLAB, etc.

**3. `setup_uv_environment(sandbox_path, ml_packages=None)`**
- Creates a Python virtual environment using `uv` (fast Rust-based package manager)
- Installs comprehensive ML package suite:
  - **Core**: numpy, pandas, scipy
  - **ML/DL**: scikit-learn, torch, torchvision, torchaudio
  - **Advanced**: transformers, datasets, pytorch-lightning, torch-geometric
  - **Visualization**: matplotlib, seaborn
  - **Utilities**: requests
- Uses Windows-specific path: `venv_path / "Scripts" / "python.exe"`

**4. `copy_env_file()`**
- Copies `.env` from `karpathy/` to `sandbox/` so agents can access API keys
- Maintains security by keeping secrets out of git

**5. `setup_sandbox()`**
- Main orchestrator that calls all setup functions in sequence
- Creates the complete sandbox environment ready for ML work

**Intuition**: This is the "infrastructure provisioning" module. It automates what would normally take 30+ minutes of manual setup (cloning repos, creating venvs, installing packages, configuring paths) into a single function call.

---

### Agent Behavior Configuration

**`karpathy/instructions.yaml`**
- **Purpose**: Defines the behavior, personality, and workflow for agents using structured YAML

#### Section 1: main_agent Instructions

**Core Philosophy**:
- High-level orchestrator focused on ML experiment lifecycle
- Operates in cycles: plan → delegate → inspect → iterate
- All work happens in `sandbox/` directory (filesystem isolation)

**Decision Framework**:
1. Restate user's goal to clarify understanding
2. Classify request type:
   - **Conceptual question** → Answer directly with snippets
   - **Project/experiment task** → Create plan and orchestrate experts

**Workflow Loop**:
1. Check existing artifacts (plan.md, research.md, datasets, logs)
2. Use Plan Creator/Reviewer to create step-by-step plan
3. Delegate to appropriate experts with clear inputs/outputs
4. Inspect results (files, logs, metrics)
5. Decide next actions or completion

**Expert Team** (12+ predefined specialists):
- **Planning**: Plan Creator, Plan Reviewer, Code Planner, Code Reviewer
- **Research**: Research Agent (uses perplexity-search)
- **Data**: Data Discoverer, Data Engineer
- **Execution**: Experiment Manager, Evaluation Agent, Code Writer, Code Executor
- **Infrastructure**: Infra & Modal Operator
- **Quality**: Reviewer (monitors progress)
- **Extensibility**: Create new experts as needed

**Communication Guidelines**:
- Concise but informative
- Mention which expert and their task
- Provide concrete file paths and commands
- Summarize artifacts and reproducibility steps on completion

#### Section 2: common_instructions

**Universal Guidelines for All Experts**:
- Always use available Skills when relevant (120+ scientific tools)
- Prefer ML-focused skills: pytorch, pytorch-lightning, transformers, torch-geometric, scikit-learn
- Use Python environment in `sandbox/`
- Manage dependencies with `uv` (add to local `pyproject.toml`)
- Read environment variables from `sandbox/.env`
- Never print full secrets
- Use `markitdown` skill for reading PDFs, DOCX, PPTX, etc.
- Use `exploratory-data-analysis` skill for data exploration
- Check resources with `get-available-resources` before code planning/execution
- Design experiments with resource limits in mind (CPU, memory, GPU, time)
- Prefer small, well-defined steps
- Verify outputs frequently
- Inspect errors and iterate rather than giving up
- Be explicit about limitations and uncertainties

**Intuition**: These instructions encode **best practices for ML engineering** in a form that LLMs can follow. They create a systematic, iterative, quality-focused workflow similar to how experienced ML engineers approach projects.

---

### Sandbox Environment

**`sandbox/`**
- **Purpose**: Isolated workspace where all agent code execution and file I/O happens

**Contents**:

**a) `.claude/skills/` (120+ Scientific Skills)**
- Downloaded from `K-Dense-AI/claude-scientific-skills` repository
- Each skill is a directory with:
  - `SKILL.md`: Comprehensive documentation (when to use, capabilities, examples)
  - `scripts/`: Python scripts demonstrating usage
  - `references/`: Detailed API documentation
  - `assets/`: Images, diagrams, example outputs

**Notable Skills**:
- `exploratory-data-analysis`: Analyzes 200+ scientific file formats
- `markitdown`: Converts PDF/Office/images/audio to Markdown
- `perplexity-search`: AI web search with source citations
- `pytorch-lightning`: Scalable deep learning with multi-GPU/TPU
- `transformers`: Hugging Face models and pipelines
- `scikit-learn`: Classical ML algorithms
- `matplotlib`, `seaborn`: Visualization
- `get-available-resources`: System resource checking
- Domain-specific: biopython, rdkit, scanpy, deepchem, etc.
- Database access: PubMed, ChEMBL, COSMIC, UniProt, PDB, etc.

**b) `.venv/` (Python Virtual Environment)**
- Created via `uv venv`
- Contains 11 core ML packages installed via `uv pip install`
- Isolated from system Python to avoid conflicts
- Windows structure: `Scripts/python.exe` instead of Unix `bin/python`

**c) `.env` (Environment Variables)**
- Copied from `karpathy/.env`
- Contains `OPENROUTER_API_KEY` for API access
- Contains `AGENT_MODEL` for model selection
- Accessible to all agents during execution

**d) `Iris.csv` (Example Dataset)**
- Classic Iris flower dataset (150 rows, 5 columns)
- Columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
- Three species: Iris-setosa, Iris-versicolor, Iris-virginica
- Used for demonstrating ML capabilities

**Intuition**: The sandbox is a **production-grade ML workspace** with everything pre-installed. It's like having a fully-configured Jupyter environment or Colab notebook, but accessible programmatically to AI agents. The skills library provides "recipes" for common ML tasks.

---

## Configuration Files

### pyproject.toml
**Purpose**: Python project metadata and dependencies

**Dependencies** (9 core packages):
1. `claude-agent-sdk>=0.1.6`: Claude Code SDK for code execution
2. `google-adk>=1.18.0`: Google Agent Development Kit for agent orchestration
3. `litellm>=1.79.3`: Unified API for 100+ LLM providers
4. `markitdown[all]>=0.1.3`: Document conversion to Markdown
5. `mcp>=1.21.1`: Model Context Protocol
6. `modal>=1.2.2`: Cloud compute platform integration
7. `openai>=2.8.0`: OpenAI API client
8. `pydantic>=2.12.4`: Data validation
9. `python-dotenv>=1.2.1`: Environment variable management

**Requirements**: Python 3.13+ (specified in `.python-version`)

### uv.lock
**Purpose**: Dependency lock file for reproducible builds
- Version: 1, Revision: 3
- Contains exact versions and hashes for all dependencies and transitive dependencies
- Ensures consistent environment across machines
- Generated by `uv` package manager

### .gitignore
**Key Exclusions**:
- Environment files: `.env`, `.env.*`
- Python artifacts: `__pycache__/`, `*.pyc`, `*.egg-info/`
- Virtual environments: `venv/`, `.venv/`, `env/`
- Lock file: `uv.lock` (platform-specific)
- Testing: `.pytest_cache/`, `.coverage`, `htmlcov/`
- IDEs: `.vscode/`, `.idea/`, `.DS_Store`
- Project-specific: `sandbox/` (generated), `.modal/`

---

## The Workflow

**User Interaction Flow**:
```
User (Browser)
  → ADK Web Interface (http://localhost:8000)
    → MainAgent (Google ADK LlmAgent)
      → delegate_task tool
        → Claude Code SDK query
          → Expert Agent Execution (in sandbox/)
            → Skills + Python Environment + User Data
              → Results (files, logs, metrics)
            ← Results returned
          ← Task completion
        ← Structured result
      ← Tool output
    ← Agent response
  ← Display to user
```

**Example Workflow**:
1. **User request**: "Train a sentiment classifier on this dataset"
2. **MainAgent** analyzes and creates plan
3. **Delegates** to experts in sequence:
   - Data Discoverer → analyzes dataset
   - Plan Creator → designs ML approach
   - Code Writer → implements PyTorch model
   - Code Executor → runs training script
   - Evaluation Agent → measures performance
4. **Returns** summary with file paths, metrics, and reproducibility steps

**Filesystem Isolation**:
- All code execution happens in `sandbox/` directory
- Main agent never writes files directly - only delegates
- This provides security and organization

**Multi-Expert Orchestration**:
1. User: "Train a model on Iris dataset"
2. MainAgent analyzes request
3. Delegates to Data Discoverer: "Analyze Iris.csv"
4. Delegates to Plan Creator: "Create ML plan for Iris classification"
5. Delegates to Code Writer: "Implement sklearn model per plan"
6. Delegates to Code Executor: "Run the training script"
7. Delegates to Evaluation Agent: "Evaluate model performance"
8. Returns summary with artifact locations

**Parallel Execution**:
- Independent tasks can run in parallel (e.g., multiple hyperparameter sweeps)
- Reduces total execution time
- Mentioned in instructions as optimization strategy

---

## Scientific Skills Library

### Overview
120+ pre-built tools covering:

### Categories

**ML/DL Frameworks**:
- pytorch-lightning: Scalable deep learning with multi-GPU/TPU
- transformers: Hugging Face models and pipelines
- scikit-learn: Classical ML algorithms
- torch-geometric: Graph neural networks

**Data Science**:
- exploratory-data-analysis: Analyzes 200+ scientific file formats
- pandas, numpy: Data manipulation
- scipy: Scientific computing

**Document Processing**:
- markitdown: PDF, Office, images, audio → Markdown conversion

**Research Tools**:
- perplexity-search: AI web search with source citations

**Bioinformatics** (50+ formats):
- biopython: Sequence analysis
- File formats: fasta, fastq, bam, vcf, bed, gtf, gff

**Chemistry**:
- rdkit: Cheminformatics
- File formats: pdb, mol2, sdf, xyz, gro

**Microscopy** (45+ formats):
- File formats: tiff, czi, svs, dicom

**Visualization**:
- matplotlib, seaborn: Static plots
- plotly: Interactive visualizations

**Database Access**:
- PubMed: Biomedical literature
- ChEMBL: Drug discovery data
- UniProt: Protein sequences
- PDB: Protein structures
- COSMIC: Cancer genomics

### Skill Structure

Each skill directory contains:
- `SKILL.md`: Comprehensive documentation with metadata YAML header
- `scripts/`: Demonstration Python code
- `references/`: API documentation
- `assets/`: Images, diagrams, example outputs

---

## Key Design Patterns

### 1. Delegation Pattern
- MainAgent doesn't execute code - it delegates to specialists
- Similar to "Chain of Responsibility" pattern
- Enables specialization and scalability

### 2. Sandbox Isolation
- All work in isolated `sandbox/` directory
- Prevents pollution of main codebase
- Easy to reset/clean between projects

### 3. Skills as Tools
- 120+ pre-built "recipes" for scientific tasks
- Claude Code discovers skills via `.claude/skills/` directory
- Each skill has SKILL.md with structured metadata (name, description, usage)

### 4. Async Streaming
- `delegate_task` uses async generators for real-time progress
- Streams tool usage for observability
- Better UX than blocking calls

### 5. Environment Cloning
- `.env` copied to sandbox so agents have API access
- Maintains security (parent .env not exposed to agents)

### 6. Iterative Refinement Loop
```
Check artifacts → Plan → Execute → Inspect results → Decide next action
                    ↑                                        ↓
                    └────────────────────────────────────────┘
```

---

## Problem Being Solved

### Core Problem
Building state-of-the-art ML models requires:
1. Deep expertise across multiple domains (data engineering, model architecture, training, evaluation, deployment)
2. Iterative experimentation and refinement
3. Managing complex toolchains and dependencies
4. Integrating research (papers, documentation) with implementation
5. Significant time investment (days to weeks per project)

### Solution Approach
- **Multi-agent specialization**: Each expert focuses on one domain (like a team of specialists)
- **Scientific skills library**: Pre-built solutions for common tasks (like having senior engineer templates)
- **Automated environment setup**: No manual dependency hell
- **Iterative workflow**: Plan → Execute → Evaluate → Refine (mimics human ML engineering)
- **Claude's coding expertise**: Leverages Claude Code's strong scientific/coding capabilities

### Use Cases
1. **Rapid prototyping**: "Build a text classifier for sentiment analysis"
2. **Research reproduction**: "Implement the Vision Transformer paper"
3. **Data exploration**: "Analyze this genomics dataset and suggest models"
4. **Experiment management**: "Run a hyperparameter sweep for learning rate"
5. **Model comparison**: "Compare LSTM vs Transformer on time series data"

---

## Integration Points

### Claude Code SDK
- Version: 0.1.6+
- Provides: Code execution, file I/O, bash commands, skills discovery
- Preset: `"claude_code"` (Claude's built-in coding expertise)

### Google ADK
- Version: 1.18.0+
- Provides: Agent orchestration, web UI, tool calling, message streaming
- LlmAgent: High-level agent abstraction

### LiteLLM
- Version: 1.79.3+
- Provides: Unified interface to 100+ LLM providers
- Used via: `LiteLlm(model=MODEL)` in agent.py

### OpenRouter
- API gateway for accessing multiple LLM providers
- Single API key for all models
- Handles rate limiting, load balancing, billing

### Modal (Future)
- Cloud compute platform for scalable ML training
- Listed in dependencies and upcoming features
- Will enable "choose any compute you want"

---

## Security Considerations

### API Key Management
- Keys stored in `.env` files (gitignored)
- Copied to sandbox for agent access
- Instructions warn: "never print full secrets"

### Sandbox Isolation
- All code execution in `sandbox/` directory
- Prevents agents from modifying main codebase
- Can be reset/deleted without affecting core system

### Permission Mode
- Uses `"bypassPermissions"` for automation
- Trade-off: Speed vs manual review
- Appropriate for trusted code generation tasks

---

## Performance and Scalability

### Fast Package Management
- Uses `uv` (Rust-based) instead of pip
- 10-100x faster dependency resolution
- Reduces setup time from minutes to seconds

### Parallel Execution
- Instructions explicitly mention parallel delegation
- Multiple independent experiments can run simultaneously
- Better resource utilization

### Lightweight Skills
- Skills are markdown documentation + Python scripts
- No heavy frameworks or containers
- Fast discovery and execution

### Resource Awareness
- `get-available-resources` skill for checking CPU/GPU/memory
- Instructions mandate checking resources before execution
- Prevents OOM errors and crashes

---

## Configuration and Extensibility

### Model Selection
- Configured via `AGENT_MODEL` environment variable
- Supports any model via OpenRouter (100+ providers)
- Default: `openrouter/google/gemini-3-pro-preview`
- Can use: GPT-4, Claude, Mistral, Llama, etc.

### Custom Experts
- Instructions explicitly state: "Create new experts as needed"
- Simply add new role description to `append_system_prompt`
- Examples: "Database Engineer", "Deployment Specialist", "Documentation Writer"

### Custom Skills
- Add directories to `sandbox/.claude/skills/`
- Follow SKILL.md format with metadata YAML header
- Include scripts/, references/, assets/ subdirectories

### Custom ML Packages
- Modify `ml_packages` list in `setup_uv_environment()`
- Add to sandbox with: `uv pip install --python sandbox/.venv/Scripts/python.exe <package>`

---

## Recent Development History

Based on git log analysis:

### Most Recent Changes
1. **README improvements**: Added badges (License, Python version, PRs Welcome)
2. **.gitignore expansion**: Added comprehensive exclusions
3. **Model change**: Default to Gemini 3 Pro Preview
4. **Instructions refinement**: Enhanced expert descriptions and workflows
5. **Data Discoverer role**: Added specialist for dataset management
6. **Community building**: Added Slack community link
7. **Markitdown integration**: Explicit skill usage for document conversion
8. **Documentation clarity**: Improved setup instructions

### Current Work-in-Progress
- `.env.example`: API key changed (possibly for testing)
- `utils.py`: Windows path compatibility fix (Scripts vs bin)

---

## Community and Ecosystem

### K-Dense AI Organization
- Main developer/maintainer
- Also maintains Claude Scientific Skills repository
- Building k-dense.ai platform (closed beta, launching Dec 2025)

### Open Source
- MIT License (permissive)
- PRs welcome badge
- Active development (15 commits in recent history)

### Claude Scientific Skills
- Upstream dependency (120+ skills)
- Regularly updated
- Covers: bioinformatics, chemistry, physics, ML, data science

### Slack Community
- K-Dense Community Slack
- Support and idea sharing
- User engagement

---

## Future Roadmap

### Upcoming Features (from README)
1. **Modal sandbox integration**: Choose any compute type (CPU, GPU, TPU)
2. **K-Dense Web features**: Possible backporting from commercial product
3. **Multi-agentic system**: More powerful ML via agent swarms (k-dense.ai)

---

## Summary: The Big Picture

### What is Karpathy?
An autonomous ML engineer that can take natural language requests like "train a sentiment classifier" and:
1. Research best practices
2. Explore and prepare data
3. Design model architecture
4. Write training code
5. Run experiments
6. Evaluate results
7. Iterate to improve performance

### How does it work?
- **Strategic Layer** (Google ADK): Main agent decides what needs to be done
- **Tactical Layer** (Claude Code SDK): Expert agents execute code and file operations
- **Knowledge Layer** (Skills): 120+ pre-built scientific tools and workflows
- **Infrastructure Layer** (Sandbox): Isolated environment with ML packages pre-installed

### Why is it powerful?
- Combines planning intelligence (ADK agent) with coding expertise (Claude Code)
- Specialization through expert agents (divide and conquer)
- Extensive scientific knowledge via skills library
- Iterative refinement (plan → execute → evaluate → improve)
- Production-grade setup (proper dependency management, isolation, security)

### Who is it for?
- ML researchers wanting to prototype ideas quickly
- Data scientists needing to explore new datasets
- Engineers reproducing papers or baselines
- Anyone who wants an AI pair programmer for ML tasks

### Key Innovation
Treating ML engineering as an **orchestration problem** rather than a monolithic task. By breaking down complex ML projects into specialized sub-agents with clear interfaces, the system can tackle projects that would normally require human expertise across multiple domains.

---

## File Inventory (Absolute Paths)

### Root Level
- `D:\Projects\karpathy\README.md` - Project documentation
- `D:\Projects\karpathy\LICENSE` - MIT license
- `D:\Projects\karpathy\pyproject.toml` - Project metadata and dependencies
- `D:\Projects\karpathy\uv.lock` - Locked dependencies
- `D:\Projects\karpathy\start.py` - Main entry point
- `D:\Projects\karpathy\.gitignore` - Git exclusions
- `D:\Projects\karpathy\.python-version` - Python 3.13 requirement

### karpathy/ Package
- `D:\Projects\karpathy\karpathy\__init__.py` - Package exports
- `D:\Projects\karpathy\karpathy\agent.py` - Main agent definition
- `D:\Projects\karpathy\karpathy\tools.py` - Delegation bridge
- `D:\Projects\karpathy\karpathy\utils.py` - Setup utilities
- `D:\Projects\karpathy\karpathy\instructions.yaml` - Agent behavior config
- `D:\Projects\karpathy\karpathy\.env.example` - API key template
- `D:\Projects\karpathy\karpathy\.env` - Actual API keys (gitignored)

### sandbox/ Environment
- `D:\Projects\karpathy\sandbox\.claude\skills\` - 120+ scientific skills (subdirectories)
- `D:\Projects\karpathy\sandbox\.venv\` - Python virtual environment
- `D:\Projects\karpathy\sandbox\.env` - Environment variables (copied)
- `D:\Projects\karpathy\sandbox\Iris.csv` - Example dataset (150 rows)

### Key Skills (examples from 120+)
- `D:\Projects\karpathy\sandbox\.claude\skills\exploratory-data-analysis\SKILL.md`
- `D:\Projects\karpathy\sandbox\.claude\skills\pytorch-lightning\SKILL.md`
- `D:\Projects\karpathy\sandbox\.claude\skills\markitdown\SKILL.md`
- `D:\Projects\karpathy\sandbox\.claude\skills\perplexity-search\SKILL.md`
- `D:\Projects\karpathy\sandbox\.claude\skills\transformers\SKILL.md`
- `D:\Projects\karpathy\sandbox\.claude\skills\scikit-learn\SKILL.md`

---

## Conclusion

This is a **well-architected system** that demonstrates sophisticated software engineering practices: separation of concerns, dependency injection, async programming, sandbox isolation, and extensibility through skills. The codebase is clean, documented, and follows modern Python best practices.

The project successfully bridges the gap between high-level ML task planning and low-level code execution, creating an autonomous system capable of tackling complex machine learning projects with minimal human intervention.
