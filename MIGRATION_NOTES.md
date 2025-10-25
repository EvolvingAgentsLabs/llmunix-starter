# LLMunix Starter Template - Clean Version Notes

## Overview

This repository is the **minimal, clean version** of LLMunix inspired by the llmunix-marketplace plugin philosophy. It provides only the essential kernel components, with zero bloat.

## Philosophy: The Factory, Not the Products

Traditional approach: Ship many pre-built agents for different domains.

**LLMunix approach**: Ship a minimal kernel that creates agents on demand.

## What's Included (Minimal Core)

### Core System Agents (3 only)
- **SystemAgent.md** - Master orchestrator
- **MemoryAnalysisAgent.md** - Log analyzer
- **MemoryConsolidationAgent.md** - Learning consolidator

### System Files
- **SmartMemory.md** - Memory architecture specification
- **Tools/** - Query, trace, and tool mapping definitions

### Kernel
- **CLAUDE.md** - Auto-loading kernel that embodies SystemAgent

### Structure
- **projects/** - Empty directory (populated dynamically)
- **.claude/agents/** - Only the 3 core agents
- **system/** - Source definitions for reference

## What's NOT Included (Intentionally)

### ❌ No Domain-Specific Agents
Removed from both `system/agents/` and `.claude/agents/`:
- VisionaryAgent.md
- MathematicianAgent.md
- QuantumEngineerAgent.md

**Why?** These should be created dynamically based on the specific project requirements. Pre-shipping them:
1. Biases users toward certain problem-solving patterns
2. Creates bloat as more domains are added
3. Prevents true customization to project needs
4. Defeats the "factory" philosophy

### ❌ No Example Projects
Removed entire `projects/` directory contents:
- Project_aorta
- Project_chaos_bifurcation_tutorial_v2
- Project_seismic_surveying

**Why?**
1. Users should create their own projects dynamically
2. Examples create false expectations
3. Takes up repository space
4. Users learn better by doing

### ❌ No Alternative Runtimes
Removed:
- qwen_runtime.py
- QWEN.md
- GEMINI.md

**Why?** This is the web-optimized version. Alternative runtimes can be documented separately or in a different branch. The focus is on Claude Code web.

### ❌ No Scenarios Directory
Removed `scenarios/` entirely.

**Why?** Usage patterns should emerge from actual use, not prescribed scenarios.

## Comparison: Before vs After Cleanup

### Before (Bloated)
```
.claude/agents/
├── SystemAgent.md                    ✓ Core
├── MemoryAnalysisAgent.md            ✓ Core
├── MemoryConsolidationAgent.md       ✓ Core
├── VisionaryAgent.md                 ✗ Domain-specific
├── MathematicianAgent.md             ✗ Domain-specific
├── QuantumEngineerAgent.md           ✗ Domain-specific
└── Project_*_*.md                    ✗ Project-specific

system/agents/
├── SystemAgent.md                    ✓ Core
├── MemoryAnalysisAgent.md            ✓ Core
├── MemoryConsolidationAgent.md       ✓ Core
├── VisionaryAgent.md                 ✗ Domain-specific
├── MathematicianAgent.md             ✗ Domain-specific
└── QuantumEngineerAgent.md           ✗ Domain-specific

projects/
├── Project_aorta/                    ✗ Example
├── Project_chaos_bifurcation.../     ✗ Example
└── Project_seismic_surveying/        ✗ Example

Root:
├── qwen_runtime.py                   ✗ Alternative runtime
├── QWEN.md                           ✗ Alternative runtime
├── GEMINI.md                         ✗ Alternative runtime
└── scenarios/                        ✗ Prescriptive examples
```

### After (Clean)
```
.claude/agents/
├── SystemAgent.md                    ✓ Core only
├── MemoryAnalysisAgent.md            ✓ Core only
└── MemoryConsolidationAgent.md       ✓ Core only

system/agents/
├── SystemAgent.md                    ✓ Core only
├── MemoryAnalysisAgent.md            ✓ Core only
└── MemoryConsolidationAgent.md       ✓ Core only

projects/
└── .gitkeep                          ✓ Empty, ready for dynamic creation

Root: Clean essentials only
```

## Key Changes to Documentation

### CLAUDE.md
- **Updated** to clarify only 3 core agents exist
- **Removed** misleading list of domain agents
- **Emphasized** dynamic agent creation as the primary workflow

### README.md
- **Rewritten** to emphasize "factory not products" philosophy
- **Added** clear comparison with plugin
- **Removed** references to example projects and alternative runtimes
- **Highlighted** what is NOT included (as a feature)

## Philosophy Alignment

This cleaned template now perfectly mirrors the llmunix-marketplace plugin:

| Component | Plugin | Starter (After Cleanup) |
|-----------|--------|------------------------|
| Core Agents | 3 | 3 |
| Domain Agents | 0 (created dynamically) | 0 (created dynamically) |
| Example Projects | 0 | 0 |
| System Files | 4 | 4 |
| Philosophy | Minimal kernel | Minimal kernel |

## Benefits of the Clean Approach

### 1. **True Dynamic Creation**
Users are forced to think about what agents they need, not pick from a limited menu.

### 2. **Reduced Repository Size**
From 51 files (22,665 lines) to ~15 core files.

### 3. **Clearer Value Proposition**
"This is a factory" is immediately obvious from the structure.

### 4. **Better Learning**
Users learn by creating their own projects, not copying examples.

### 5. **Easier Maintenance**
No need to maintain example projects or domain agents that may become outdated.

### 6. **Scalability**
The system can handle any domain without pre-shipping agents for every possibility.

## Migration Summary

**Removed:**
- 3 domain-specific agents from system/agents/
- 3 domain-specific agents from .claude/agents/
- 3 project-specific agents from .claude/agents/
- 3 example projects with all contents
- 3 alternative runtime files
- 1 scenarios directory

**Kept:**
- 3 core system agents
- 4 system specification files
- 1 kernel (CLAUDE.md)
- Essential documentation
- License and configuration files

**Result:** A minimal, clean template that embodies the "factory not products" philosophy.

## Usage After Cleanup

1. **Fork/Clone** this template
2. **Connect** to Claude Code web
3. **Give a goal** - any complex, multi-faceted problem
4. **Watch** as Claude dynamically creates the exact agents needed
5. **Learn** from the generated agent definitions
6. **Reuse** learnings in future projects through the memory system

No examples needed. No pre-built solutions. Just pure, dynamic agent creation.

---

**This is LLMunix at its essence: A self-evolving operating system that builds itself to solve your problem.**
