# LLMunix for Legal and Consulting Professionals

## A Plain-Language Guide to Using Claude Code Web

This guide is written specifically for attorneys, paralegals, consultants, and other professionals who need to generate, review, and manage documents using Claude Code Web with LLMunix.

---

## Table of Contents

1. [What is Claude Code Web?](#what-is-claude-code-web)
2. [Understanding the Branch and Pull Request Workflow](#understanding-the-branch-and-pull-request-workflow)
3. [Step-by-Step: Starting a Legal Project](#step-by-step-starting-a-legal-project)
4. [Retrieving Your Generated Documents](#retrieving-your-generated-documents)
5. [Common Legal Use Cases](#common-legal-use-cases)
6. [Best Practices for Legal Document Generation](#best-practices-for-legal-document-generation)
7. [Document Retention and Version Control](#document-retention-and-version-control)
8. [Security and Confidentiality Considerations](#security-and-confidentiality-considerations)
9. [Glossary of Technical Terms](#glossary-of-technical-terms)

---

## What is Claude Code Web?

**Claude Code Web** is an AI-powered assistant that connects to your GitHub repository and helps you create, analyze, and manage documents. Think of it as having a highly capable associate who:

- Works in a secure, isolated environment
- Saves all work to a version-controlled system (GitHub)
- Creates a clear audit trail of all changes
- Delivers completed work for your review before it becomes final

### Key Benefits for Legal Professionals

| Benefit | Description |
|---------|-------------|
| **Audit Trail** | Every change is tracked with timestamps and descriptions |
| **Version Control** | You can see exactly what changed between document versions |
| **Review Before Acceptance** | All work is delivered as a "proposal" you must approve |
| **Secure Isolation** | Your documents are processed in a private, temporary environment |
| **No Permanent Storage** | The working environment is destroyed after each session |

---

## Understanding the Branch and Pull Request Workflow

### The Concept Explained in Legal Terms

When Claude Code Web completes a task, it does **not** directly modify your main repository. Instead, it follows a process similar to legal document review:

1. **Draft Creation**: Claude creates all documents in a separate "branch" (think: a draft folder)
2. **Proposal Submission**: Claude submits a "Pull Request" (think: a memo requesting approval to incorporate the draft)
3. **Your Review**: You review the proposed changes in GitHub's interface
4. **Acceptance or Rejection**: You decide whether to merge (accept) or close (reject) the proposal

### Visual Workflow

```
Your Repository (Main)          Claude's Branch (Draft)
       │                               │
       │   1. Claude creates branch ──►│
       │                               │
       │                        2. Claude works on documents
       │                               │
       │   3. Pull Request (Proposal) ─┤
       │                               │
  4. You Review ◄──────────────────────┤
       │                               │
  5. Accept/Merge ─────────────────────┤
       │                               │
       ▼                               │
  Changes Now in Main                  │
```

### Why This Matters for Legal Work

- **No Accidental Changes**: Your main documents are never modified without explicit approval
- **Complete Review Opportunity**: You can see every line that will change before accepting
- **Easy Rejection**: If the work is unsatisfactory, simply close the Pull Request with no impact to your main files
- **Historical Record**: Even rejected proposals are preserved in the system for reference

---

## Step-by-Step: Starting a Legal Project

### Step 1: Connect Your Repository

1. Navigate to [claude.ai/code](https://claude.ai/code)
2. Sign in with your organization's GitHub account
3. Select your repository from the list
4. Wait for Claude to initialize the workspace

### Step 2: Describe Your Legal Task

When giving Claude instructions, be specific and include:

- **Document Type**: Contract, memo, brief, due diligence report, etc.
- **Jurisdiction**: Specify applicable law (e.g., "Delaware corporate law")
- **Parties**: Who are the parties involved?
- **Key Terms**: What specific provisions or clauses are needed?
- **Output Format**: Markdown, Word-compatible, plain text?

**Example Request:**

```
Create a due diligence checklist for acquiring a Delaware C-Corporation
with 15 employees. Include sections for:
- Corporate governance documents
- Employment agreements and benefits
- Intellectual property portfolio
- Material contracts
- Litigation history
- Regulatory compliance

Output as a markdown document that can be shared with the client.
```

### Step 3: Review Claude's Work in Progress

Claude will:
1. Create a project folder under `projects/[ProjectName]/`
2. Generate specialized agents for the task (e.g., CorporateComplianceAgent, ContractReviewAgent)
3. Produce documents in `projects/[ProjectName]/output/`
4. Log all work in `projects/[ProjectName]/memory/`

### Step 4: Receive the Pull Request

When Claude finishes:
1. A new branch is created (e.g., `claude/legal-due-diligence-abc123`)
2. A Pull Request is automatically created
3. You'll see a summary of all files created or modified

### Step 5: Review and Approve

In GitHub:
1. Go to the "Pull Requests" tab
2. Click on Claude's Pull Request
3. Review the "Files changed" tab to see all documents
4. If satisfied, click "Merge pull request"
5. If not satisfied, click "Close pull request" and provide feedback

---

## Retrieving Your Generated Documents

### Method 1: Download from GitHub (Recommended for Most Users)

1. **Navigate to the Pull Request** in your GitHub repository
2. **Click "Files changed"** to view all generated documents
3. **For individual files**:
   - Click on the file name
   - Click the "Raw" button
   - Right-click and "Save As" to download
4. **For all files at once**:
   - Merge the Pull Request first
   - Go to your repository's main page
   - Click "Code" → "Download ZIP"

### Method 2: Clone the Branch Locally

For attorneys comfortable with command-line tools:

```bash
# Clone the entire repository
git clone https://github.com/your-org/your-repo.git

# Switch to Claude's branch
cd your-repo
git checkout claude/your-project-branch

# Your documents are now in projects/[ProjectName]/output/
```

### Method 3: Use GitHub's Web Interface

1. In the Pull Request, click on any file name
2. Click the download icon (↓) to download individual files
3. Markdown files can be viewed directly in the browser
4. Copy/paste content as needed into your document management system

### Document Locations

After a project completes, documents are organized as follows:

```
projects/[ProjectName]/
├── output/                    ← FINAL DELIVERABLES HERE
│   ├── [ProjectName]_documentation.md
│   ├── contracts/             ← Contract drafts
│   ├── memos/                 ← Legal memoranda
│   └── reports/               ← Analysis and reports
├── components/
│   └── agents/                ← Specialized agents created (for reference)
└── memory/
    ├── short_term/            ← Session logs (audit trail)
    └── long_term/             ← Consolidated learnings
```

---

## Common Legal Use Cases

### Contract Drafting and Review

```
Request: "Draft a master services agreement for a technology consulting
firm providing services to healthcare clients. Include HIPAA-compliant
data handling provisions, limitation of liability appropriate for
professional services, and standard termination clauses. Governing
law should be California."
```

### Due Diligence

```
Request: "Create a comprehensive due diligence request list for the
acquisition of a SaaS company with 50 employees. Organize by category
and include priority levels for each item. Focus on intellectual
property, customer contracts, and employee matters."
```

### Legal Research Summary

```
Request: "Research and summarize recent Delaware Chancery Court
decisions regarding fiduciary duties in the context of SPAC mergers.
Create a memo suitable for partner review with case citations and
key holdings."
```

### Compliance Checklists

```
Request: "Create a GDPR compliance checklist for a US-based company
expanding to serve EU customers. Include required documentation,
process changes, and technical requirements. Organize by compliance
category with implementation priority."
```

### Document Analysis

```
Request: "Analyze the attached vendor agreements [provide content]
and create a summary table comparing key terms: liability caps,
indemnification provisions, termination rights, and renewal terms."
```

---

## Best Practices for Legal Document Generation

### 1. Be Specific About Jurisdiction and Governing Law

**Good**: "Draft using New York law, with venue in Manhattan"
**Avoid**: "Draft a contract" (too vague)

### 2. Specify Document Purpose and Audience

**Good**: "Create a client-facing summary of the merger agreement key terms, suitable for board presentation"
**Avoid**: "Summarize the deal" (unclear audience and format)

### 3. Request Appropriate Formatting

**Good**: "Use numbered paragraphs, include signature blocks, format for Word export"
**Avoid**: Assuming Claude knows your firm's preferred format

### 4. Include Relevant Precedent

When possible, reference:
- Existing templates or forms
- Prior similar transactions
- Specific clauses or language you prefer

### 5. Request Explanatory Notes

**Good**: "For each contract provision, include a brief comment explaining its purpose and any negotiation considerations"

### 6. Specify Confidentiality Requirements

For sensitive matters:
- Use private repositories only
- Do not include client names or identifying details in requests
- Anonymize or redact sensitive information before providing context

---

## Document Retention and Version Control

### How Version Control Benefits Legal Practice

| Feature | Legal Benefit |
|---------|---------------|
| **Commit History** | Shows every change made, by whom, and when |
| **Branch Comparison** | Compare any two versions side-by-side |
| **Blame View** | See who wrote each line and when |
| **Rollback** | Restore any previous version instantly |
| **Audit Trail** | Complete history for regulatory compliance |

### Recommended Retention Practices

1. **Keep All Branches**: Do not delete Claude's branches after merging—they serve as audit trail
2. **Tag Important Versions**: Use Git tags for final client deliverables
3. **Document PR Decisions**: Add comments explaining why changes were accepted or rejected
4. **Export Periodically**: Download copies for your document management system

### Creating Permanent Records

To create a permanent record of a project:

1. After merging, navigate to the project folder in GitHub
2. Download the entire `projects/[ProjectName]/` folder
3. Store in your firm's document management system
4. The `memory/short_term/` folder contains the complete session log

---

## Security and Confidentiality Considerations

### Private Repository Best Practices

For client matters and confidential work:

| Setting | Recommendation |
|---------|---------------|
| **Repository Visibility** | Private |
| **Network Access** | Limited or None |
| **Team Access** | Minimum necessary personnel |
| **Branch Protection** | Require review before merge |

### What Claude Can and Cannot Access

**Claude CAN access**:
- Files in your connected repository
- Web resources (if network access enabled)
- Public information within its training data

**Claude CANNOT access**:
- Other repositories you haven't connected
- Your local files outside the repository
- Your email, calendar, or other systems
- Previous sessions (each session is isolated)

### Confidentiality Recommendations

1. **Use Code Names**: Refer to parties by placeholder names (e.g., "Buyer," "Target")
2. **Anonymize Data**: Remove or redact identifying information
3. **Private Repositories Only**: Never use public repositories for client work
4. **Review Before Committing**: Check all files before merging
5. **Session Isolation**: Each Claude session is isolated and temporary

### Compliance Considerations

- **Attorney-Client Privilege**: Review your jurisdiction's rules on AI tool usage
- **Data Residency**: Understand where data is processed and stored
- **Client Consent**: Consider disclosure in engagement letters
- **Supervision**: All AI-generated work requires attorney review

---

## Glossary of Technical Terms

| Term | Plain-Language Definition |
|------|--------------------------|
| **Repository** | A folder containing all your project files, hosted on GitHub |
| **Branch** | A separate copy of your files where changes can be made without affecting the original |
| **Main Branch** | Your official, approved version of all files |
| **Commit** | A saved checkpoint of changes, with a description and timestamp |
| **Pull Request (PR)** | A formal proposal to incorporate changes from one branch into another |
| **Merge** | The act of accepting a Pull Request, incorporating the changes |
| **Clone** | Creating a local copy of a repository on your computer |
| **Push** | Uploading changes from your computer to GitHub |
| **Diff** | A comparison showing what changed between two versions |
| **Markdown** | A simple text format that can be converted to HTML or Word documents |

---

## Quick Reference: Common Tasks

### "I need to download my documents"

1. Go to the Pull Request
2. Click "Files changed"
3. Click file name → "Raw" → Save

### "I need to reject Claude's work"

1. Go to the Pull Request
2. Click "Close pull request"
3. Optionally add a comment explaining why

### "I need to see what changed"

1. Go to the Pull Request
2. Click "Files changed"
3. Additions shown in green, deletions in red

### "I need to go back to a previous version"

1. Go to repository → Commits
2. Find the commit you want
3. Click the commit hash
4. Click "Browse files" to see files at that point

### "I need to continue a previous project"

1. Reference the prior project in your new request
2. Claude can read files from previous projects in the same repository
3. Example: "Building on the due diligence checklist in projects/Project_ABC_Acquisition/, add a section for environmental compliance"

---

## Support and Resources

- **Getting Started Guide**: [GETTING_STARTED.md](../GETTING_STARTED.md)
- **System Architecture**: [CLAUDE_CODE_ARCHITECTURE.md](../CLAUDE_CODE_ARCHITECTURE.md)
- **Claude Code Documentation**: [docs.anthropic.com](https://docs.anthropic.com)

---

*This guide was created to help legal professionals effectively use LLMunix with Claude Code Web. For firm-specific customization or training, please consult with your technology team.*
