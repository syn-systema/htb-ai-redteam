# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This is an Obsidian vault + git repository for tracking progress through HackTheBox's **AI Red Team path** (11 modules). The runtime is Google Colab Pro (executed via Antigravity, Google's VS Code fork with Colab integration). This is a documentation/learning artifact, not a code project — the goal is a durable, searchable, reproducible record of the path.

Remote: `https://github.com/syn-systema/htb-ai-redteam`.

## Directory Structure

```
00-Meta/                        # Master progress index
01-Fundamentals-of-AI/          # Module 1 — theory
02-Applications-of-AI-in-InfoSec/
03-Introduction-to-Red-Teaming-AI/
04-Prompt-Injection-Attacks/
05-LLM-Output-Attacks/
06-AI-Data-Attacks/
07-Attacking-AI-Application-and-System/
08-AI-Evasion-Foundations/
09-AI-Evasion-First-Order-Attacks/
10-AI-Evasion-Sparsity-Attacks/
11-AI-Privacy/
Templates/                      # Module, Skills Assessment, Notebook scaffolds
shared/                         # Python helpers imported into Colab via git clone
Assets/                         # Screenshots, diagrams
docs/superpowers/               # Design specs + implementation plans
```

Per-module folder:

```
NN-Module-Name/
  NN-Module-Name.md            # Main note, sections as ## headers
  Skills-Assessment.md          # Per-module assessment
  notebooks/                    # .ipynb snapshots exported from Colab
  datasets.md                   # Dataset → Drive path / source URL
```

## Key Files

- **Main Index:** `00-Meta/HTB AI Red Team Path.md` — progress across all 11 modules.
- **Templates:** `Templates/Module Template.md`, `Templates/Skills Assessment Template.md`, `Templates/Notebook Scaffold Template.ipynb`.
- **Shared helpers:** `shared/data.py`, `shared/viz.py`, `shared/attacks.py` — importable into Colab.

## File Conventions

- **One markdown note per module.** Sections are `##` headers within the file. Do not fragment a module across multiple files.
- **Module frontmatter:**
  ```yaml
  ---
  module_number: NN
  module_name: ""
  status: not-started         # not-started | in-progress | completed
  difficulty: ""              # Easy | Medium | Hard
  tier: ""
  estimated_time: ""
  sections_total: 0
  sections_done: 0
  started: ""
  completed: ""
  ---
  ```
- `[[wikilinks]]` for cross-referencing notes.
- Screenshots → `Assets/`, referenced from notes.

## Colab Integration Pattern

Every new Colab notebook starts with this boilerplate cell:

```python
!git clone https://github.com/syn-systema/htb-ai-redteam.git /content/repo
import sys; sys.path.append('/content/repo')
from shared.data import load_mnist   # example — swap in what the exercise needs
```

`shared/` is the single source of truth for reusable dataset loaders, visualization helpers, and attack utilities.

## Notebook Snapshot Flow

After completing an exercise in Colab:

1. In Colab: `File → Download .ipynb`.
2. Drop the file into `NN-Module-Name/notebooks/`.
3. Claude updates the module markdown note to reference the notebook and inlines the key cells as ```` ```python ```` blocks so the note is self-contained.
4. Commit with a message like `feat(04): complete direct-prompt-injection exercise`.

## Live-Session Workflow — TEACHING MODE B (IMPORTANT)

**The goal is learning. Most execution belongs to the user; Claude's role is to scaffold and interpret.**

### What Claude does

- **Scaffold notebooks:** write starter Colab cells into the module note as ```` ```python ```` blocks that the user will paste into Colab.
- **Author helpers:** add functions to `shared/data.py`, `shared/viz.py`, `shared/attacks.py` as patterns repeat across modules.
- **Draft markdown:** module notes, skills assessment questions, interpretation of output.
- **Interpret output:** when the user pastes Colab cell output, explain it, update the note in real time, recommend the next step.
- **Propose commits:** draft commit messages; the user runs `git commit` and `git push`.

### What the user does

- Runs every Colab cell, every `!pip install`, every training/attack execution.
- Submits every Skills Assessment answer to HTB.
- Runs all git commands.

### Remote LLM work is FULLY hands-off

For **any** interaction with HTB-hosted LLMs (prompt injection challenges, LLM output attacks, Skills Assessment endpoints): Claude does not send prompts, does not craft live payloads against a running target, does not submit answers. Claude may *design* payloads and *explain* techniques in the note — but sending them to the target is the user's alone. This preserves the learning reps on prompt crafting, which is the central skill of modules 04–05.

### Real-time vault updates

Log every command, output, interpretation, and pivot into the relevant module note in-line — not at session end. Record dead ends too. The vault is the study artifact and the evidence trail.

## Progress Tracking

When a section completes:
1. `[ ]` → `[x]` in the module note.
2. Bump `sections_done` in the frontmatter.
3. Update progress % in `00-Meta/HTB AI Red Team Path.md`.

When a module completes:
1. Frontmatter: `status: completed`, `completed: YYYY-MM-DD`.
2. Update master index row.
3. Commit with `feat(NN): complete <module name>`.

## Common Tasks

**Start a new module:**
```
Copy Templates/Module Template.md into NN-Module-Name/NN-Module-Name.md,
fill frontmatter (status: in-progress, started: today),
add ## section headers for each HTB section,
update 00-Meta/HTB AI Red Team Path.md (status → in-progress).
```

**Snapshot a notebook:**
```
User downloads .ipynb from Colab → drops into NN-Module-Name/notebooks/ →
Claude updates the module note to link it + inline key cells →
User commits.
```

**Add a shared helper:**
```
When the same loader/viz/attack-util appears in two notebooks, extract it
into shared/. Add a one-line docstring. Commit under "feat(shared): ...".
```

**Mark a section complete:**
```
In NN-Module-Name.md: [ ] → [x], bump sections_done in frontmatter,
recompute progress in 00-Meta/HTB AI Red Team Path.md.
```

## Note Format Style

Module notes should be:
- **Actionable:** `- [ ]` checklists for multi-step exercises.
- **Scannable:** tables for reference data (hyperparameters, metrics).
- **Runnable-adjacent:** key Colab cells inlined as ```` ```python ```` blocks with short explanations.
- **Cross-referenced:** `[[wikilinks]]` to related modules and to the master index.
- **Concise prose:** compress theory into bullet lists and diagrams where possible.
