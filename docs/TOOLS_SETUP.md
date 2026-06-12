# Tools Setup — Stock Scan (Build)

## ✅ Already Installed (Done)

### Claude Skills — auto-loaded by this IDE
Located in `.claude/skills/`:

| Skill | Triggers When |
|---|---|
| `superpowers` | Starting any feature, bug fix, or refactor; systematic debugging |
| `context-engineering` | Large file reads (scoring_engine.py etc.), long sessions |
| `xlsx` | Working with .xlsx/.csv source data sheets |
| `docx` | Reading/editing `columns to add.docx` or creating reports |
| `skill-creator` | Creating or improving new skills |

### MCP Servers — registered in `.claude/settings.json`
| Server | Status |
|---|---|
| `filesystem` | ✅ Active (CSV Data folder) |
| `fetch` | ✅ Active |
| `taskmaster-ai` | ⚠️ Needs API key (see below) |
| `codebase-memory` | ⚠️ Needs binary install (see below) |

---

## ⚠️ 2 Steps Still Required

### Step 1: Task Master AI — Add Your Anthropic API Key

Edit `.claude/settings.json` and replace `YOUR_ANTHROPIC_API_KEY_HERE` with your actual key:

```json
"taskmaster-ai": {
  "command": "npx",
  "args": ["-y", "task-master-ai"],
  "env": {
    "ANTHROPIC_API_KEY": "sk-ant-..."
  }
}
```

Your key is at: https://console.anthropic.com/settings/keys

### Step 2: Codebase Memory MCP — Download and Install Binary

Run this in PowerShell (one time):

```powershell
# Download installer
Invoke-WebRequest -Uri https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/install.ps1 -OutFile $env:TEMP\cbm_install.ps1

# (Optional) Review the script before running
notepad $env:TEMP\cbm_install.ps1

# Run installer
& $env:TEMP\cbm_install.ps1
```

After install, restart this IDE. Then say: **"Index this project"** to build the knowledge graph.

---

## ❌ Not Installed (Intentional)

**Vanna AI** — The README was reviewed. Vanna 2.0 requires:
- A running FastAPI server with user auth
- LLM API keys for the NL→SQL model
- A SQL database (your source of truth is CSV files, not SQL)

This is enterprise-grade infrastructure for a solo CSV-based project. **Skip it.** If you ever migrate your 6 CSV sheets to a SQLite DB, revisit this.

---

## How Skills Work

Skills are markdown files in `.claude/skills/<name>/SKILL.md`. They are auto-discovered by the IDE. The skill `description` field tells the AI when to activate it. No plugin commands needed — just having the file present is enough.

To create a new skill for this project, ask: *"Help me create a skill for [task]"* — the `skill-creator` skill will guide you.
