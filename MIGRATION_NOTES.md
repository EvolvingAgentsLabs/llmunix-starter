# LLMunix Starter Template - Migration Notes

## Overview

This repository is the **web-optimized version** of the original LLMunix project, designed specifically for Claude Code on the web with a "fork-and-run" workflow.

## What Changed?

### 1. Pre-populated `.claude/agents` Directory

**Before (Local CLI Version):**
- Setup scripts (`setup_agents.sh`, `setup_agents.ps1`) copied agents at runtime
- `.claude/agents/` was created dynamically on first run
- Not compatible with Claude Code web (no pre-execution setup)

**After (Web Template Version):**
- All system agents pre-copied to `.claude/agents/`
- Project-specific example agents included with naming prefixes
- Ready to use immediately when cloned by Claude Code web
- No setup scripts needed for web usage

**Pre-populated Agents:**
- System core agents: `SystemAgent.md`, `MemoryAnalysisAgent.md`, `MemoryConsolidationAgent.md`, etc.
- Project-specific agents: `Project_chaos_bifurcation_tutorial_v2_*.md`

### 2. New CLAUDE.md Kernel/Bootloader

**Before:**
- `CLAUDE.md` assumed user would run `boot llmunix` command
- Required manual initialization

**After:**
- `CLAUDE.md` is automatically read by Claude Code web on session start
- Acts as the "kernel" that loads the SystemAgent persona
- Contains complete workflow instructions for dynamic agent creation
- No user interaction needed to "boot" the system

### 3. Updated README.md

**Before:**
- Focused on local CLI setup with `setup_agents.sh`
- Git clone → setup script → manual command workflow

**After:**
- Web-first approach: "Use this template" → Connect to Claude Code → Start
- Clear separation between web usage (primary) and local/Qwen usage (advanced)
- Emphasis on the "fork-and-run" model

### 4. Removed Files for Web Version

The following files are **not needed** for Claude Code web but are kept in the original project for local/Qwen runtime users:
- `setup_agents.sh` - Setup script for local CLI
- `setup_agents.ps1` - Setup script for PowerShell
- `llmunix-boot` - Boot script for Gemini runtime

These files have been **excluded** from the starter template as they're incompatible with the web model.

## Architecture Differences

### Original (Local CLI Model)
```
1. User clones repository
2. User runs setup_agents.sh
3. Script copies agents to .claude/agents/
4. User runs: claude "llmunix execute: 'goal'"
5. System processes goal
```

### New (Web Template Model)
```
1. User clicks "Use this template"
2. User connects repo to Claude Code web
3. Claude Code reads CLAUDE.md automatically
4. User provides goal in chat
5. System processes goal immediately
```

## Key Improvements

### 1. **Zero Setup Required**
- No scripts to run
- No manual agent copying
- Works out-of-the-box on Claude Code web

### 2. **Version-Controlled Agents**
- All agents are committed to Git
- Full transparency and reproducibility
- Easy to customize before forking

### 3. **Cleaner Separation**
- Template repo for web users
- Original repo for local CLI/Qwen users
- Each optimized for its environment

### 4. **Self-Contained**
- Everything needed is in the repository
- No external dependencies for core functionality
- Example projects demonstrate capabilities

## How to Use This Template

### For Web Users (Recommended)
1. Click "Use this template" on GitHub
2. Create your own repository from the template
3. Connect to Claude Code web
4. Start working - no setup needed!

### For Local CLI Users (Advanced)
The agents are already in `.claude/agents/`, so you can:
```bash
git clone your-repo
cd your-repo
claude "Your goal here"
```

### For Qwen Runtime Users (Experimental)
```bash
git clone your-repo
cd your-repo
export OPENAI_API_KEY=your_key
python qwen_runtime.py "Your goal here"
```

## What Stays the Same

Despite the migration, the **core philosophy** remains identical:

- **Pure Markdown OS**: Agents are markdown files
- **Dynamic Agent Creation**: System creates specialized agents on-the-fly
- **Memory & Learning**: Short-term and long-term memory system
- **Multi-Agent Orchestration**: SystemAgent coordinates specialist agents
- **Self-Evolution**: Learns from every project

## Directory Structure

```
llmunix-starter/
├── .claude/
│   └── agents/                 # PRE-POPULATED (key difference!)
│       ├── SystemAgent.md
│       ├── MemoryAnalysisAgent.md
│       └── ... (all system agents)
├── CLAUDE.md                   # NEW: Auto-loading kernel
├── README.md                   # UPDATED: Web-first guide
├── system/                     # Source definitions
│   ├── agents/
│   ├── tools/
│   └── SmartMemory.md
├── projects/                   # Example projects
│   ├── Project_aorta/
│   ├── Project_chaos_bifurcation_tutorial_v2/
│   └── Project_seismic_surveying/
├── doc/                        # Documentation
└── scenarios/                  # Usage scenarios
```

## Migration Checklist

If you're adapting your own fork of the original LLMunix:

- [x] Create `.claude/agents/` directory
- [x] Copy all `system/agents/*.md` to `.claude/agents/`
- [x] Copy project agents with naming prefixes to avoid conflicts
- [x] Rewrite `CLAUDE.md` as auto-loading kernel
- [x] Update `README.md` for web-first workflow
- [x] Remove or move setup scripts
- [x] Create `.gitignore` for appropriate exclusions
- [x] Commit everything to Git
- [x] Enable "Template repository" in GitHub settings

## Next Steps

### To Publish Your Template

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/llmunix-starter.git
   git branch -M main
   git push -u origin main
   ```

2. **Enable Template Setting:**
   - Go to repository Settings
   - Check "Template repository"

3. **Test the Template:**
   - Use the template to create a new repo
   - Connect it to Claude Code web
   - Verify it works without setup

### To Customize

- Add your own system agents to `system/agents/` and `.claude/agents/`
- Create additional example projects
- Customize `CLAUDE.md` instructions
- Add domain-specific tools

## Support

For issues or questions:
- Original project: `evolvingagentslabs-llmunix`
- This template: `llmunix-starter`
- Claude Code docs: https://docs.claude.com/claude-code

---

**This template represents the evolution of LLMunix from a local-first to a web-first architecture while preserving its powerful core philosophy.**
