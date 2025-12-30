# Agent Templates for Legal and Consulting Professionals

This directory contains pre-built agent templates optimized for legal and consulting work. These templates can be used as-is or customized for your specific practice needs.

## Available Templates

### Legal Templates

| Template | Description | Best For |
|----------|-------------|----------|
| [LegalDocumentAgent.md](LegalDocumentAgent.md) | Legal document drafting and formatting | Contracts, memos, briefs, corporate documents |
| [LegalResearchAgent.md](LegalResearchAgent.md) | Legal research and case analysis | Case law research, statutory analysis, legal memos |

### Consulting Templates

| Template | Description | Best For |
|----------|-------------|----------|
| [ConsultingAnalystAgent.md](ConsultingAnalystAgent.md) | Business analysis and strategy | Market analysis, due diligence, client presentations |

## How to Use These Templates

### Option 1: Direct Reference (Recommended)

Simply reference the template when making your request:

```
Using the LegalDocumentAgent template, draft a non-disclosure agreement
for a software development partnership between a startup and an enterprise
technology company. Use Delaware law.
```

Claude will read the template and apply its guidelines to your task.

### Option 2: Copy to Your Project

For project-specific customization:

1. Copy the template to `projects/[YourProject]/components/agents/`
2. Modify the template as needed
3. Claude will use your customized version

### Option 3: Create New Templates

Use these as starting points to create templates for your specific practice:

1. Copy a similar template
2. Modify capabilities, tools, and instructions
3. Save with a descriptive name
4. Add to your repository for future use

## Template Structure

Each template includes:

```yaml
---
name: AgentName              # Identifier for the agent
type: specialist             # Agent classification
project: ProjectName         # Default project context
capabilities:                # What the agent can do
  - capability 1
  - capability 2
tools:                       # Claude Code tools the agent uses
  - Read
  - Write
  - etc.
---

# Agent Instructions

[Detailed instructions for how the agent should behave]
```

## Customization Tips

### For Law Firms

1. **Add your firm's style guide**: Include formatting preferences, citation styles
2. **Include template clauses**: Add boilerplate provisions your firm prefers
3. **Specify jurisdiction defaults**: Set your primary jurisdiction
4. **Add confidentiality notices**: Include firm-standard privilege markings

### For Consulting Firms

1. **Add branding guidelines**: Colors, fonts, slide layouts
2. **Include methodology**: Your firm's proprietary frameworks
3. **Specify deliverable formats**: Client-facing document standards
4. **Add quality checklists**: Firm-specific review criteria

## Creating Project-Specific Agents

For specialized matters, create project-specific agents:

```markdown
---
name: AcquisitionDueDiligenceAgent
type: specialist
project: Project_ABC_Acquisition
capabilities:
  - Healthcare industry due diligence
  - HIPAA compliance analysis
  - Provider contract review
tools:
  - Read
  - Write
  - Glob
  - Grep
---

# AcquisitionDueDiligenceAgent

You are a due diligence specialist focused on healthcare company acquisitions...

[Project-specific instructions]
```

## Best Practices

1. **Version control your templates**: Track changes over time
2. **Share across projects**: Place refined templates in this templates directory
3. **Include examples**: Add sample outputs to help calibrate quality
4. **Update regularly**: Incorporate lessons learned from each project

---

*For the complete guide on using LLMunix for legal and consulting work, see [LEGAL_PROFESSIONAL_GUIDE.md](../LEGAL_PROFESSIONAL_GUIDE.md)*
