# LLMunix Starter: A Self-Evolving OS Template

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLMunix** is a Pure Markdown Operating System where AI agents are created dynamically to solve complex problems. This starter template is optimized for Claude Code on the web, providing a "fork-and-run" experience.

## What is LLMunix?

LLMunix is a revolutionary approach to AI-powered problem solving:

- **Pure Markdown OS**: Agents, memory, and tools are human-readable markdown files
- **Dynamic Agent Creation**: The system creates specialized agents on-the-fly for your specific task
- **Self-Evolving**: Learns from each project and improves over time
- **Multi-Agent Orchestration**: Complex problems are decomposed and solved by coordinated specialist agents
- **Version-Controlled Intelligence**: Everything is in Git - agents, memory, learnings

## Quick Start with Claude Code on the Web (Recommended)

This template is designed for the web version of Claude Code. Get started in 3 simple steps:

### 1. Use this Template
Click the "**Use this template**" button at the top of this repository page and select "**Create a new repository**". This will create a copy of `llmunix-starter` in your own GitHub account.

### 2. Connect to Claude Code
- Go to [claude.ai/code](https://claude.ai/code)
- Connect your GitHub account
- Authorize the Claude GitHub app for the new repository you just created

### 3. Start Your Project
Select your new repository in the Claude Code interface and give Claude your high-level goal. For example:

> "Create a machine learning pipeline to predict customer churn using the data in `data/customer_data.csv`. The pipeline should include data cleaning, feature engineering, model training with scikit-learn, and evaluation."

Claude will:
1. Read the LLMunix kernel instructions from `CLAUDE.md`
2. Start working on a new branch
3. Create a project folder under `projects/`
4. Dynamically write specialized agent definitions (e.g., `DataEngineerAgent`, `MLEngineerAgent`, `DocumentationAgent`)
5. Orchestrate these agents to complete your goal
6. Save all outputs and memory logs to the project folder

## Example Use Cases

### Scientific Research
> "Create a tutorial explaining chaos theory and bifurcation in discrete prey-predator models. Include mathematical foundations, Python simulations, and visualizations."

### Software Development
> "Build a REST API for a task management system with user authentication, CRUD operations, and PostgreSQL database. Include tests and API documentation."

### Data Analysis
> "Analyze sales data to identify trends, create visualizations, and generate a comprehensive report with actionable insights."

### Quantum Computing
> "Design a quantum algorithm for solving the traveling salesman problem. Include the quantum circuit design, implementation in Qiskit, and performance analysis."

## How It Works

### The LLMunix Architecture

```
llmunix-starter/
├── CLAUDE.md                 # The kernel/bootloader (auto-read by Claude Code)
├── .claude/
│   └── agents/              # Pre-populated system agents
│       ├── SystemAgent.md
│       ├── MemoryAnalysisAgent.md
│       ├── MemoryConsolidationAgent.md
│       └── ...
├── system/
│   ├── agents/              # Source definitions for core agents
│   ├── tools/               # Tool definitions
│   └── SmartMemory.md       # Memory system documentation
├── projects/                # Your projects live here
│   └── Project_[name]/
│       ├── components/
│       │   └── agents/      # Project-specific agents
│       ├── output/          # Final deliverables
│       └── memory/
│           ├── short_term/  # Session logs
│           └── long_term/   # Consolidated learnings
└── doc/                     # Documentation and guides
```

### The Workflow

When you give Claude a goal:

1. **SystemAgent analyzes** your goal and creates a plan
2. **Dynamic agents are created** - specialized for your specific domain
3. **Agents collaborate** - each handles their area of expertise
4. **Everything is logged** - full memory of the process
5. **Outputs are organized** - in a structured project folder
6. **System learns** - consolidates insights for future use

### Core System Agents

LLMunix comes with these pre-configured agents:

- **SystemAgent** - Master orchestrator (the brain)
- **MemoryAnalysisAgent** - Analyzes logs and extracts patterns
- **MemoryConsolidationAgent** - Consolidates learnings
- **VisionaryAgent** - Strategic thinking and conceptualization
- **MathematicianAgent** - Mathematical modeling
- **QuantumEngineerAgent** - Quantum computing expertise

You can create unlimited specialized agents for any domain!

## Advanced Features

### Memory System

LLMunix has a sophisticated memory system:

- **Short-term memory**: Detailed logs of every action in a project session
- **Long-term memory**: Consolidated learnings, patterns, and reusable templates
- **Memory queries**: Agents can learn from past projects

### Self-Evolution

After each project, the system:
1. Analyzes what worked well
2. Identifies reusable patterns
3. Stores agent templates for similar future tasks
4. Continuously improves its problem-solving capabilities

## Alternative Runtimes

### Local CLI Usage (Advanced)

For users who want to run LLMunix locally with the Claude CLI:

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/your-llmunix-project.git
cd your-llmunix-project

# The agents are already pre-populated in .claude/agents/
# You can start using Claude CLI directly
claude "Your goal here"
```

### Lightweight Qwen Runtime

LLMunix also supports a lightweight runtime using Qwen models:

```bash
# Install dependencies
pip install openai

# Set up your API key
export OPENAI_API_KEY=your_key_here

# Run with Qwen
python qwen_runtime.py "Your goal here"
```

See `QWEN.md` for detailed instructions.

## Project Structure Best Practices

When LLMunix creates a project, it follows this structure:

```
projects/Project_your_goal/
├── components/
│   └── agents/                    # Agent definitions
│       ├── DomainExpertAgent.md
│       └── ImplementationAgent.md
├── output/                        # Final deliverables
│   ├── documentation.md
│   ├── code/
│   └── results/
└── memory/
    ├── short_term/                # Session logs
    │   ├── 2025-10-25_planning.md
    │   └── 2025-10-25_execution.md
    └── long_term/                 # Learnings
        └── project_learnings.md
```

## Example Project: Chaos & Bifurcation Tutorial

This repository includes a complete example project:

```
projects/Project_chaos_bifurcation_tutorial_v2/
```

Explore it to see:
- How agents are defined
- Memory logging structure
- Output organization
- The final tutorial and code

## Philosophy

LLMunix is built on these principles:

1. **Markdown is Code** - Human-readable, version-controllable, LLM-interpretable
2. **Dynamic over Static** - Create agents for the problem, not the other way around
3. **Learning by Doing** - The system improves with every project
4. **Orchestration over Monoliths** - Specialized agents collaborate
5. **Transparency** - Every decision is logged and auditable

## Contributing

Contributions are welcome! Areas of interest:

- New system agent templates
- Runtime implementations for other LLMs
- Tools and utilities
- Documentation and examples

## License

MIT License - see `LICENSE` file for details.

## Architecture Documentation

For deep technical details, see:
- `CLAUDE_CODE_ARCHITECTURE.md` - How LLMunix integrates with Claude Code
- `system/SmartMemory.md` - Memory system design
- `doc/` - Additional guides and documentation

## Support & Community

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share your projects and ideas in GitHub Discussions

---

**Start building intelligent, self-evolving systems today. Fork this template and let LLMunix orchestrate your next breakthrough.**
