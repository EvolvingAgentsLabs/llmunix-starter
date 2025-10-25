# LLMunix Starter: Pure Markdown Operating System Template

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**LLMunix** is a Pure Markdown Operating System that doesn't ship pre-built solutions—it provides the **factory to build them**. This starter template is optimized for Claude Code on the web, enabling dynamic agent creation and self-evolving problem-solving.

## Philosophy: The Factory, Not the Products

Traditional AI systems pre-define agents for specific domains (e.g., "Python Developer", "Data Scientist"). This has fundamental limitations:

❌ **Domain coverage is bounded** - Can't handle novel combinations
❌ **Agents are generic** - Not tailored to specific problem nuances
❌ **Systems grow bloated** - Shipping hundreds of pre-built agents
❌ **No learning feedback loop** - Each execution starts from scratch

LLMunix inverts this model:

✅ **Infinite domain coverage** - Creates agents for any expertise needed
✅ **Problem-specific agents** - Tailored prompts for exact requirements
✅ **Minimal core** - Only 3 system agents shipped
✅ **Continuous evolution** - Every project improves future performance

## What You Get

This template contains **only the essential kernel components**:

### Core System Agents (3 only)
- **SystemAgent** - Master orchestrator for multi-agent coordination
- **MemoryAnalysisAgent** - Analyzes and interprets memory logs
- **MemoryConsolidationAgent** - Consolidates short-term memory into long-term learnings

### System Specifications
- **SmartMemory.md** - Hierarchical memory architecture
- **Tool definitions** - How LLMunix concepts map to Claude Code tools
- **Memory managers** - Query and trace management systems

### The Kernel (CLAUDE.md)
- Auto-loads when you start Claude Code
- Implements the SystemAgent orchestration workflow
- Handles dynamic agent creation and execution

**That's it.** No example projects. No domain-specific agents. No bloat.

When you give Claude a goal, it creates exactly the agents needed—nothing more, nothing less.

## Quick Start

### For Public Repositories

1. **Use this template** → Click "Use this template" and create a public repository
2. **Connect to Claude Code** → Visit [claude.ai/code](https://claude.ai/code) and authorize GitHub
3. **Select your repository** → Choose your new repo from the list
4. **Give Claude a goal** → Start with any ambitious, multi-faceted problem
5. **Review and merge** → Claude pushes changes to a branch for PR review

### For Private Repositories

1. **Use this template** → Create a **private** repository from this template
2. **Configure GitHub App** → Grant Claude access to your private repositories
3. **Set environment variables** → Configure secrets and API keys securely
4. **Optional: Add SessionStart hooks** → Automate dependency installation
5. **Start your project** → Work on proprietary code in complete isolation

### Detailed Setup Guide

For complete step-by-step instructions for both public and private repositories, see **[GETTING_STARTED.md](GETTING_STARTED.md)**.

This guide covers:
- Setting up public vs private repositories
- Configuring network access and security
- Managing environment variables and secrets
- Installing dependencies with SessionStart hooks
- Moving sessions between web and local CLI
- Troubleshooting common issues

## How It Works: Dynamic Evolution

### The Workflow

```
User Goal
    ↓
1. SystemAgent analyzes and decomposes the goal
    ↓
2. Creates project structure in projects/[ProjectName]/
    ↓
3. WRITES new agent markdown files for required expertise
    ↓
4. Reads each agent and invokes via Task tool
    ↓
5. Logs all interactions to memory/short_term/
    ↓
6. Produces final outputs in output/
    ↓
7. Consolidates learnings to memory/long_term/
    ↓
8. Future projects query and reuse these learnings
```

### Agent Creation Example

When you request a machine learning pipeline, LLMunix might create:

```markdown
---
name: FeatureEngineerAgent
type: specialist
capabilities:
  - Feature selection and extraction
  - Dimensionality reduction
  - Feature encoding
tools: [Read, Write, Bash]
---

You are an expert feature engineer specializing in...
[Detailed, project-specific prompt]
```

This agent is:
- **Created at runtime** for this specific project
- **Tailored** to the exact ML problem domain
- **Logged** for future reuse
- **Ephemeral** - only exists if needed

## Example Use Cases

### Quantum Computing Research
```
Goal: "Develop a quantum annealing algorithm for supply chain optimization"

Dynamic Agents Created:
- LogisticsVisionaryAgent.md
- QuantumAlgorithmDesignerAgent.md
- QiskitImplementationAgent.md
- OptimizationValidatorAgent.md
```

### Full-Stack Development
```
Goal: "Build a real-time collaborative whiteboard with WebSocket sync"

Dynamic Agents Created:
- SystemArchitectAgent.md
- ReactFrontendAgent.md
- WebSocketBackendAgent.md
- DatabaseSchemaAgent.md
- IntegrationTestAgent.md
```

### Data Science
```
Goal: "Predict customer churn using behavioral analytics"

Dynamic Agents Created:
- DataExplorationAgent.md
- FeatureEngineeringAgent.md
- ModelTrainingAgent.md
- ResultsVisualizationAgent.md
```

## Architecture

### Project Structure (Created Dynamically)

```
llmunix-starter/
├── .claude/
│   └── agents/                    # Only 3 core agents
│       ├── SystemAgent.md
│       ├── MemoryAnalysisAgent.md
│       └── MemoryConsolidationAgent.md
├── system/
│   ├── agents/                    # Source definitions
│   ├── tools/                     # Tool specifications
│   └── SmartMemory.md
├── projects/                      # Empty - you create these
│   └── [Created dynamically per goal]
│       ├── components/
│       │   └── agents/            # Project-specific agents
│       ├── output/                # Final deliverables
│       └── memory/
│           ├── short_term/        # Interaction logs
│           └── long_term/         # Consolidated learnings
├── CLAUDE.md                      # The kernel (auto-loads)
└── README.md
```

### Memory System

**Short-Term Memory (Episodic)**
- Raw execution traces during project runtime
- Timestamped logs with full prompts and responses
- One file per agent interaction

**Long-Term Memory (Semantic)**
- Distilled knowledge for future reuse
- Agent templates and workflow patterns
- Domain insights and best practices

**Learning Loop**
```
Execute → Log → Consolidate → Store → Query → Apply → Execute...
```

Each project makes the system smarter for the next one.

## Why Pure Markdown?

LLMunix operates entirely through markdown:

- **Agent definitions** = Markdown with YAML frontmatter
- **Memory traces** = Markdown interaction logs
- **Knowledge base** = Markdown learnings documents

Benefits:
- **Human-readable** - All system state is inspectable
- **Version-controllable** - Track evolution in Git
- **Portable** - No binary dependencies or databases
- **LLM-native** - Claude reads and writes naturally

## Best Practices

### Formulate Clear, Ambitious Goals

❌ Vague: "Help me with some code"
✅ Clear: "Build a distributed task queue with Redis backend, exponential backoff, and monitoring dashboard"

❌ Too narrow: "Write a function to sort an array"
✅ Ambitious: "Create a data processing pipeline with streaming ingestion, transformation, and real-time analytics"

### Trust the System

- Don't micromanage which agents to create
- Let LLMunix decide the optimal decomposition
- Review the generated agents afterward to learn patterns

### Leverage Learning

After a few projects:
- Check `memory/long_term/` to see what the system learned
- Similar future projects will bootstrap faster
- Agent templates become more refined over time

## Advanced: The Pure Markdown Paradigm

Every component is markdown:

**1. Agent Definition = Executable Prompt**
```markdown
---
name: MyAgent
capabilities: [...]
---
System prompt goes here...
```

**2. Orchestration = Reading & Delegating**
```
1. Write agent definition
2. Read agent definition
3. Pass content to Task tool
```

**3. Learning = Markdown Analysis**
```
MemoryConsolidationAgent reads short_term/*.md
Extracts patterns → Writes long_term/learnings.md
```

## Comparison: Plugin vs Starter

This starter template is the **repository equivalent** of the [LLMunix CLI plugin](https://github.com/evolving-agents-labs/llmunix-marketplace).

| Feature | CLI Plugin | Starter Template |
|---------|------------|------------------|
| Installation | `/plugin install` | "Use this template" |
| Activation | `/llmunix "goal"` | Just chat with Claude |
| Core Agents | 3 only | 3 only |
| Projects | Dynamic | Dynamic |
| Learning | Yes | Yes |
| Best For | Local CLI users | Web users |

Both share the same philosophy: **minimal core, infinite extensibility**.

## Troubleshooting

### Issue: Agent creation quality varies
**Solution:** After a project, review `components/agents/*.md` and manually refine prompts. The next consolidation will store improved templates.

### Issue: Not reusing past learnings
**Solution:** Use domain-specific keywords in your goal. Check `long_term/` to see what knowledge exists.

### Issue: Too many agents created
**Solution:** This is actually fine—the system is thorough. Over time, it learns to consolidate similar roles.

## What This Template Does NOT Include

In line with the minimal philosophy:
- ❌ No example projects
- ❌ No domain-specific agents
- ❌ No pre-built solutions
- ❌ No sample outputs or memory logs

If you want these, you'll create them. That's the point.

## Contributing

Contributions to enhance the **core kernel** are welcome:

- Core agent prompt improvements
- System specification enhancements
- Memory management optimizations
- Bug reports and fixes

**Do not submit:** Pre-built domain agents or example projects. That defeats the purpose.

## License

Apache License 2.0 - see `LICENSE` file for details.

## Credits

**Evolving Agents Labs** - Creators of LLMunix

Built on [Claude Code](https://claude.ai/code) by Anthropic.

## Links

- **Getting Started Guide**: [GETTING_STARTED.md](GETTING_STARTED.md) - Complete setup for public and private repos
- **Documentation**: `CLAUDE_CODE_ARCHITECTURE.md`, `system/SmartMemory.md`
- **Plugin Version**: [LLMunix CLI Plugin](https://github.com/evolving-agents-labs/llmunix-marketplace)
- **Claude Code Web**: [Official Documentation](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web)
- **Claude Code**: [Official Website](https://claude.ai/code)

---

**Start building:**

Give Claude an ambitious, complex goal. Watch it create the perfect team of agents to solve it. Watch those agents become templates for the next goal.

**This is dynamic evolution in action.**
