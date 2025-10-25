# Getting Started with LLMunix on Claude Code Web

This guide walks you through setting up LLMunix with Claude Code on the web for both **public** and **private** repositories.

## Prerequisites

- A GitHub account
- Claude Pro or Max subscription (required for Claude Code on the web)
- Basic understanding of Git and GitHub

## Option 1: Public Repository (Recommended for Learning)

Public repositories are perfect for:
- Learning and experimenting with LLMunix
- Open-source projects
- Sharing your LLMunix projects with the community
- Collaborating with others openly

### Step-by-Step Setup

#### 1. Create Your Repository from the Template

1. Navigate to the [llmunix-starter template](https://github.com/YOUR_USERNAME/llmunix-starter)
2. Click the green **"Use this template"** button
3. Select **"Create a new repository"**
4. Configure your repository:
   - **Owner**: Select your GitHub account or organization
   - **Repository name**: Choose a name (e.g., `my-llmunix-workspace`)
   - **Visibility**: Select **"Public"**
   - **Description** (optional): "My LLMunix workspace for dynamic agent-based development"
5. Click **"Create repository"**

#### 2. Connect to Claude Code Web

1. Visit [claude.ai/code](https://claude.ai/code)
2. Click **"Connect GitHub account"**
3. Authorize the Claude GitHub App:
   - Review the permissions requested
   - Select **"All repositories"** or choose specific repositories
   - Click **"Install & Authorize"**

#### 3. Select Your Repository

1. In the Claude Code interface, click the repository selector
2. Find and select your newly created repository (e.g., `my-llmunix-workspace`)
3. Claude will clone your repository to a secure cloud environment

#### 4. Configure Environment (Optional)

The default environment should work for most cases, but you can customize:

1. Click the current environment name
2. Select **"Add environment"** or edit the default
3. Configure:
   - **Name**: "LLMunix Default"
   - **Network Access**: "Limited" (recommended) - allows access to package managers
   - **Environment Variables**: None needed by default

#### 5. Start Your First Project

Give Claude an ambitious, multi-faceted goal:

```
Create a machine learning pipeline to predict customer churn using behavioral analytics.
Include data exploration, feature engineering, model training with scikit-learn,
hyperparameter tuning, and a comprehensive evaluation report with visualizations.
```

Claude will:
1. Read `CLAUDE.md` (the LLMunix kernel)
2. Create a new project structure in `projects/`
3. Dynamically create specialized agents (DataExplorationAgent, FeatureEngineerAgent, etc.)
4. Execute the workflow
5. Push results to a new branch

#### 6. Review and Create Pull Request

1. When Claude completes the task, you'll be notified
2. Review the changes in the GitHub interface
3. Create a pull request to merge into your main branch
4. Inspect the dynamically created agents in `projects/[ProjectName]/components/agents/`
5. Review outputs in `projects/[ProjectName]/output/`
6. Check learnings in `projects/[ProjectName]/memory/long_term/`

## Option 2: Private Repository (Recommended for Production)

Private repositories are ideal for:
- Proprietary code and business logic
- Sensitive projects
- Client work
- Production systems

### Step-by-Step Setup

#### 1. Create Your Private Repository from the Template

1. Navigate to the [llmunix-starter template](https://github.com/YOUR_USERNAME/llmunix-starter)
2. Click the green **"Use this template"** button
3. Select **"Create a new repository"**
4. Configure your repository:
   - **Owner**: Select your GitHub account or organization
   - **Repository name**: Choose a name (e.g., `company-llmunix-private`)
   - **Visibility**: Select **"Private"** ⚠️
   - **Description** (optional): "Private LLMunix workspace"
5. Click **"Create repository"**

#### 2. Install Claude GitHub App for Private Repository

1. Visit [claude.ai/code](https://claude.ai/code)
2. If not already connected, click **"Connect GitHub account"**
3. **Important**: You must grant the Claude GitHub App access to your private repositories:
   - Go to GitHub Settings → Applications → Claude (under Installed GitHub Apps)
   - Click **"Configure"**
   - Under **"Repository access"**, select:
     - **"All repositories"** (grants access to all current and future repos), OR
     - **"Only select repositories"** and add your private LLMunix repository
   - Click **"Save"**

#### 3. Select Your Private Repository in Claude Code

1. Return to [claude.ai/code](https://claude.ai/code)
2. Click the repository selector
3. You should now see your private repository listed
4. Select it - Claude will clone it to an isolated, secure cloud VM

#### 4. Configure Environment with Security Considerations

For private repositories, you may want stricter security:

**Option A: Limited Network Access (Default)**
- Allows package managers and common development tools
- Blocks most external domains
- Good balance between security and functionality

**Option B: No Network Access (Maximum Security)**
1. Click environment name → Settings
2. Set **Network Access** to **"None"**
3. Note: You'll need to pre-install dependencies or use SessionStart hooks

**Option C: Custom Allowed Domains**
1. Click environment name → Settings
2. Keep **"Limited"** access
3. Claude will automatically allow common package managers
4. For additional domains, consider using environment-specific configuration

#### 5. Add Environment Variables (If Needed)

For private projects, you may need API keys or secrets:

1. Click environment name → Settings
2. Under **"Environment Variables"**, add key-value pairs:
   ```
   DATABASE_URL=postgresql://localhost/mydb
   API_KEY=your_api_key_here
   ENVIRONMENT=development
   ```
3. **Security Note**: These are stored securely and only accessible within the isolated VM

#### 6. Configure SessionStart Hooks (Optional)

For private projects with dependencies, automate setup:

Create `.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/setup.sh"
          }
        ]
      }
    ]
  }
}
```

Create `scripts/setup.sh`:
```bash
#!/bin/bash

# Only run in remote (Claude Code web) environments
if [ "$CLAUDE_CODE_REMOTE" = "true" ]; then
  echo "Installing dependencies for remote session..."

  # Install Python dependencies
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt
  fi

  # Install Node dependencies
  if [ -f package.json ]; then
    npm install
  fi

  echo "Setup complete!"
fi

exit 0
```

Make it executable:
```bash
chmod +x scripts/setup.sh
```

#### 7. Start Your Private Project

Give Claude your business goal:

```
Analyze our customer database to build a churn prediction model.
Use the data in data/customers.csv. Create a REST API endpoint
for real-time predictions using FastAPI. Include comprehensive
tests and API documentation.
```

Claude will work in complete isolation with your private code.

#### 8. Review Changes Securely

1. All work happens in an isolated, Anthropic-managed VM
2. Changes are pushed to a new branch in your private repository
3. Review the PR privately within your organization
4. Merge when satisfied

## Understanding the Environment

### What Happens in the Cloud

When you start a Claude Code task:

1. **Repository Cloning**: Your repo (public or private) is cloned to an Anthropic-managed VM
2. **Environment Setup**: Claude reads `CLAUDE.md` and runs any SessionStart hooks
3. **Network Configuration**: Internet access is configured per your settings
4. **Isolated Execution**: Claude works in complete isolation
5. **Secure Push**: Changes are pushed to a branch via secure GitHub proxy

### Security Guarantees

- **Isolated VMs**: Each session runs in a fresh, isolated virtual machine
- **Credential Protection**: Your git credentials never enter the VM - authentication uses scoped credentials through a secure proxy
- **Network Controls**: You control what external services Claude can access
- **Automatic Cleanup**: VMs are destroyed after the session ends

### Network Access Levels Explained

**None**:
- No external network access
- Maximum security
- Pre-install all dependencies

**Limited** (Default):
- Access to package managers (npm, pip, cargo, etc.)
- Access to GitHub and common dev tools
- Blocks most other external domains
- Recommended for most projects

**Full**:
- Access to all internet
- Use only when necessary
- Review security implications

## Common Workflows

### Public Repository Workflow

```
1. Use template → Create public repo
2. Connect to Claude Code
3. Give goal → Claude creates project
4. Review PR → Merge
5. Share your learnings with community
6. Clone to local machine for further development
```

### Private Repository Workflow

```
1. Use template → Create private repo
2. Configure Claude GitHub App for private access
3. Set environment variables for secrets
4. Configure SessionStart hooks for dependencies
5. Give goal → Claude creates project in isolation
6. Review PR privately → Merge securely
7. Learnings stay in your private repo
```

### Hybrid Workflow

```
1. Develop in private repo
2. Extract learnings and agent templates
3. Create public repo for sharing
4. Publish sanitized agents and learnings
5. Community benefits from your patterns
```

## Moving Between Web and Local

### From Web to Local (Any Repository Type)

After starting a task on the web:

1. Click **"Open in CLI"** button in the Claude Code interface
2. Copy the command provided
3. In your local terminal (with the repo checked out):
   ```bash
   # Paste the command from Claude Code
   claude-code connect <session-id>
   ```
4. Your local changes will be stashed
5. Remote session state is loaded
6. Continue working locally

### From Local to Web

Commit and push your changes, then start a new web session.

## Best Practices

### For Public Repositories

1. **Document Generously**: Your `CLAUDE.md` is public - make it educational
2. **Share Learnings**: Commit agent templates to help others
3. **Sanitize Examples**: Remove any personal info from example projects
4. **Enable Issues**: Let community report bugs and suggest improvements
5. **Add License**: Clarify how others can use your work

### For Private Repositories

1. **Use Environment Variables**: Never hardcode secrets in code
2. **Configure Network Access**: Use minimum necessary access level
3. **Review Dependencies**: Audit what packages Claude installs
4. **Audit Logs**: Review what Claude did in each session
5. **Access Control**: Use GitHub's team permissions for sensitive repos
6. **SessionStart Hooks**: Automate secure environment setup

### For Both

1. **Clear Goals**: Provide specific, well-defined objectives
2. **Iterative Approach**: Start simple, then expand
3. **Review Agents**: Learn from dynamically created agents
4. **Consolidate Memory**: Let the system learn and improve
5. **Stay Engaged**: Monitor and steer Claude during execution

## Troubleshooting

### Issue: Private repository not appearing in Claude Code

**Solution**:
1. Verify Claude GitHub App has access to private repos
2. Go to GitHub Settings → Applications → Claude
3. Ensure your private repo is selected under "Repository access"

### Issue: Dependencies not installing

**Solution**:
1. Add a SessionStart hook (see above)
2. Use `CLAUDE_CODE_REMOTE` check to run only in web environment
3. Ensure script is executable (`chmod +x`)

### Issue: Network access blocking required domain

**Solution**:
1. Check if domain is in [default allowed list](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web#default-allowed-domains)
2. If not, consider using Full network access (with caution)
3. Or install dependencies via SessionStart hooks before network is restricted

### Issue: Environment variables not accessible

**Solution**:
1. Verify they're set in environment settings
2. Use proper `.env` format: `KEY=value`
3. Access in code via `os.getenv('KEY')` or equivalent
4. Persist in SessionStart hook using `$CLAUDE_ENV_FILE` if needed

### Issue: Can't move session from web to local

**Solution**:
- Ensure you're authenticated to the same GitHub account in both places
- Verify you have local checkout of the repository
- Check that git credentials are configured locally

## What's Next?

After setup:

1. **Run Your First Project**: Start with a well-defined, ambitious goal
2. **Review Generated Agents**: Learn from how Claude structures agent prompts
3. **Examine Memory Logs**: See what the system learned
4. **Iterate**: Run similar projects and watch the system improve
5. **Customize**: Modify `CLAUDE.md` based on your needs
6. **Share** (public repos): Contribute your learnings back to community

## Additional Resources

- **Main README**: Overview and philosophy
- **CLAUDE.md**: The kernel specification
- **MIGRATION_NOTES.md**: Understanding the minimal design
- **system/SmartMemory.md**: Memory system architecture
- **Claude Code Web Docs**: [Official Documentation](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web)

---

**You're ready to build!** Start with a challenging goal and watch LLMunix dynamically create the perfect team of agents to solve it.
