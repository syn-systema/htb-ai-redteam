# HTB AI Red Team Vault — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a new Obsidian vault + git repository at `~/Documents/htb-ai-redteam/` for tracking HTB's AI Red Team path across 11 modules, with Colab Pro as the runtime, following the design spec at `docs/superpowers/specs/2026-04-14-htb-ai-redteam-vault-design.md`.

**Architecture:** A single git repository that IS the Obsidian vault. Flat module-numbered folder layout (01–11). Per-module folders hold one markdown note, a Skills Assessment file, a `notebooks/` subdirectory for Colab `.ipynb` snapshots, and a `datasets.md` tracking file. A `shared/` Python package provides reusable helpers that Colab notebooks `git clone` + import. `CLAUDE.md` specifies Teaching Mode B (I scaffold, user runs) with remote LLM work fully hands-off.

**Tech Stack:** Git + GitHub (`syn-systema/htb-ai-redteam`), Obsidian (markdown + wikilinks + frontmatter), Google Colab Pro (runtime), Antigravity (VS Code fork, editor), Python 3 (`shared/` package). No CI, no build step, no external dependencies beyond what Colab supplies.

---

## File Structure

Files created during this plan:

| Path | Responsibility |
|---|---|
| `.gitignore` | Python/Jupyter/Obsidian/model-weight/dataset exclusions. |
| `README.md` | Repo-root overview, setup, workflow, static progress table. |
| `CLAUDE.md` | Operating rules for future Claude sessions — Teaching Mode B, Colab flow, git cadence. |
| `00-Meta/HTB AI Red Team Path.md` | Master progress index across all 11 modules. |
| `Templates/Module Template.md` | Frontmatter + section skeleton for a new module note. |
| `Templates/Skills Assessment Template.md` | Question blocks + methodology skeleton. |
| `Templates/Notebook Scaffold Template.ipynb` | Three starter Colab cells: repo clone, imports, GPU check. |
| `shared/__init__.py` | Package marker. |
| `shared/data.py` | Dataset loaders (start with `load_mnist`). |
| `shared/viz.py` | Plotting helpers (start with `plot_loss_curves`). |
| `shared/attacks.py` | Perturbation-norm helpers (start with `clip_linf`, `clip_l2`). |
| `Assets/.gitkeep` | Placeholder so the empty `Assets/` directory commits. |
| `NN-Module-Name/NN-Module-Name.md` (×11) | Module note seeded from `Module Template.md`. |
| `NN-Module-Name/Skills-Assessment.md` (×11) | Seeded from `Skills Assessment Template.md`. |
| `NN-Module-Name/datasets.md` (×11) | Empty dataset-tracking table. |
| `NN-Module-Name/notebooks/.gitkeep` (×11) | Placeholder so the empty subdir commits. |

Module list (exact folder names):
1. `01-Fundamentals-of-AI`
2. `02-Applications-of-AI-in-InfoSec`
3. `03-Introduction-to-Red-Teaming-AI`
4. `04-Prompt-Injection-Attacks`
5. `05-LLM-Output-Attacks`
6. `06-AI-Data-Attacks`
7. `07-Attacking-AI-Application-and-System`
8. `08-AI-Evasion-Foundations`
9. `09-AI-Evasion-First-Order-Attacks`
10. `10-AI-Evasion-Sparsity-Attacks`
11. `11-AI-Privacy`

---

## Task 1: Pre-flight — verify prerequisites and GitHub repo state

**Files:** None created — this is verification only.

- [ ] **Step 1: Verify the target directory does not already have a populated git repo**

Run:
```bash
ls -la /home/farseer/Documents/htb-ai-redteam/
```

Expected: only the `docs/superpowers/specs/` + `docs/superpowers/plans/` trees created during brainstorming. No `.git/`, no module folders, no `README.md`. If anything else is present, **stop and ask the user** before proceeding.

- [ ] **Step 2: Verify git is installed and user identity is configured**

Run:
```bash
git --version
git config --global user.name
git config --global user.email
```

Expected: `git version 2.x`, a non-empty name, a non-empty email. If user/email empty, stop and ask the user to configure them.

- [ ] **Step 3: Check the GitHub remote state**

Run:
```bash
git ls-remote https://github.com/syn-systema/htb-ai-redteam.git 2>&1 | head -5
```

Expected outcomes:
- Empty output (or `warning: You appear to have cloned an empty repository.`): repo exists but is empty. Proceed normally.
- A list of refs (e.g. `HEAD refs/heads/main`): repo has content already. **Stop and ask the user** whether to clone + merge, or reset.
- `remote: Repository not found` or auth failure: repo does not exist or credentials are missing. Stop and ask the user to create the repo or fix auth.

- [ ] **Step 4: Verify Python 3 is available (for `shared/` package validation later)**

Run:
```bash
python3 --version
```

Expected: `Python 3.10+` (any 3.x works for the empty stubs; 3.10+ matches Colab).

- [ ] **Step 5: No commit — this task is verification only.**

---

## Task 2: Initialize local git repo and configure remote

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/.git/` (by `git init`)

- [ ] **Step 1: Initialize the repo on the `main` branch**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git init -b main
```

Expected: `Initialized empty Git repository in /home/farseer/Documents/htb-ai-redteam/.git/`.

- [ ] **Step 2: Add the GitHub remote**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git remote add origin https://github.com/syn-systema/htb-ai-redteam.git
```

Expected: no output. If the remote already exists (unlikely after `git init`), expect `error: remote origin already exists` — in that case, skip this step.

- [ ] **Step 3: Verify the remote is configured**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git remote -v
```

Expected:
```
origin  https://github.com/syn-systema/htb-ai-redteam.git (fetch)
origin  https://github.com/syn-systema/htb-ai-redteam.git (push)
```

- [ ] **Step 4: Stage and commit the design/plan docs that already exist**

The design spec and this plan were created during brainstorming and already live under `docs/superpowers/`. Commit them as the first commit so history starts with "why we built it this way."

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add docs/superpowers/specs/2026-04-14-htb-ai-redteam-vault-design.md \
        docs/superpowers/plans/2026-04-14-htb-ai-redteam-vault-implementation.md
git status
```

Expected: `git status` shows both files staged under `Changes to be committed`.

Then commit:
```bash
git commit -m "docs: initial design spec + implementation plan"
```

Expected: `[main (root-commit) <sha>] docs: initial design spec + implementation plan` with `2 files changed`.

- [ ] **Step 5: Verify the first commit landed**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git log --oneline
```

Expected: one line — the root commit with message `docs: initial design spec + implementation plan`.

---

## Task 3: Create `.gitignore`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/.gitignore`

- [ ] **Step 1: Write `.gitignore` with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/.gitignore`

```
# Python
__pycache__/
*.pyc
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Obsidian workspace state (keep themes/hotkeys/plugins)
.obsidian/workspace*
.obsidian/cache
.obsidian/app.json.bak

# Model weights (retrain in Colab)
*.pt
*.pth
*.h5
*.onnx
*.safetensors

# Datasets (live in Drive, tracked in datasets.md)
data/
*.npz
*.parquet
# CSVs under 1 MB are fine to commit; larger ones belong in Drive.

# Secrets
.env

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Verify the file exists and is non-empty**

Run:
```bash
ls -la /home/farseer/Documents/htb-ai-redteam/.gitignore
wc -l /home/farseer/Documents/htb-ai-redteam/.gitignore
```

Expected: file present, >20 lines.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add .gitignore
git commit -m "chore: add gitignore for python/jupyter/obsidian/model-weights"
```

Expected: `[main <sha>] chore: add gitignore ...` with `1 file changed`.

---

## Task 4: Create `CLAUDE.md`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/CLAUDE.md`

- [ ] **Step 1: Write `CLAUDE.md` with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/CLAUDE.md`

````markdown
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
````

- [ ] **Step 2: Verify frontmatter-like conventions are syntactically consistent**

Run:
```bash
grep -n "frontmatter" /home/farseer/Documents/htb-ai-redteam/CLAUDE.md
grep -n "yaml" /home/farseer/Documents/htb-ai-redteam/CLAUDE.md
```

Expected: references to frontmatter + the YAML block above are both present.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with teaching mode B + colab workflow"
```

Expected: `[main <sha>] docs: add CLAUDE.md ...` with `1 file changed`.

---

## Task 5: Create `README.md`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/README.md`

- [ ] **Step 1: Write `README.md` with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/README.md`

````markdown
# HTB AI Red Team

Personal Obsidian vault + study artifact for HackTheBox's **AI Red Team path** (11 modules). Runtime is Google Colab Pro; editor is Antigravity (VS Code fork with Colab integration); vault viewer is Obsidian.

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/syn-systema/htb-ai-redteam.git ~/Documents/htb-ai-redteam
   ```
2. Open the folder in **Obsidian** (`Open folder as vault`).
3. Open the folder in **Antigravity** for editing + Colab integration.
4. Ensure you have **Google Colab Pro** active — several modules (AI Data Attacks, Evasion First-Order/Sparsity, AI Privacy) train models that benefit from GPU + background execution.

## Workflow

A typical study session:

1. **Start a module** — copy `Templates/Module Template.md` into `NN-Module-Name/NN-Module-Name.md`, fill in frontmatter (`status: in-progress`, `started: YYYY-MM-DD`).
2. **Work through sections** — theory sections get summarized into the module note. Interactive sections get a notebook.
3. **Scaffold a notebook** — copy `Templates/Notebook Scaffold Template.ipynb` → open in Colab → start the exercise. First cell clones this repo into `/content/repo` so `from shared.data import ...` works.
4. **Run cells in Colab** — Pro tier gives A100/L4 access and background execution.
5. **Snapshot on completion** — in Colab: `File → Download .ipynb`. Drop the file into `NN-Module-Name/notebooks/`.
6. **Update the module note** — link the notebook, inline key cells, record interpretation of results.
7. **Commit** — `git add <changed>; git commit -m "feat(NN): <what>"; git push`.

## Directory layout

```
00-Meta/                        # Master progress index
01-Fundamentals-of-AI/          # Module 1
…
11-AI-Privacy/                  # Module 11
Templates/                      # Module / Skills Assessment / Notebook scaffolds
shared/                         # Python helpers (data, viz, attacks) importable into Colab
Assets/                         # Screenshots, diagrams
docs/superpowers/               # Design specs + implementation plans
CLAUDE.md                       # Operating rules for Claude Code sessions
```

## Progress

See `00-Meta/HTB AI Red Team Path.md` for the canonical progress index. Snapshot:

| # | Module | Difficulty | Status |
|---|---|---|---|
| 01 | Fundamentals of AI | Medium | not-started |
| 02 | Applications of AI in InfoSec | — | not-started |
| 03 | Introduction to Red Teaming AI | — | not-started |
| 04 | Prompt Injection Attacks | Medium | not-started |
| 05 | LLM Output Attacks | Medium | not-started |
| 06 | AI Data Attacks | Hard | not-started |
| 07 | Attacking AI — Application and System | Medium | not-started |
| 08 | AI Evasion — Foundations | Medium | not-started |
| 09 | AI Evasion — First-Order Attacks | Hard | not-started |
| 10 | AI Evasion — Sparsity Attacks | Hard | not-started |
| 11 | AI Privacy | Medium | not-started |

## Notes

- Notebook outputs **are** committed (loss curves, confusion matrices, adv-image grids — the visual outputs are the point).
- Model weights (`*.pt`, `*.pth`) and large datasets are **not** committed — they're retrained in Colab or mounted from Drive. See `.gitignore` + per-module `datasets.md`.
- Teaching mode: see `CLAUDE.md` for how Claude Code should behave during study sessions.
````

- [ ] **Step 2: Verify file integrity**

Run:
```bash
ls -la /home/farseer/Documents/htb-ai-redteam/README.md
head -10 /home/farseer/Documents/htb-ai-redteam/README.md
```

Expected: file exists, first line is `# HTB AI Red Team`.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add README.md
git commit -m "docs: add README with setup + workflow + progress snapshot"
```

Expected: `[main <sha>] docs: add README ...`.

---

## Task 6: Create master progress index `00-Meta/HTB AI Red Team Path.md`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/00-Meta/HTB AI Red Team Path.md`

- [ ] **Step 1: Create the `00-Meta/` directory**

Run:
```bash
mkdir -p /home/farseer/Documents/htb-ai-redteam/00-Meta
```

Expected: directory exists (verify with `ls -la /home/farseer/Documents/htb-ai-redteam/`).

- [ ] **Step 2: Write the index file with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/00-Meta/HTB AI Red Team Path.md`

````markdown
# HTB AI Red Team Path

Master progress index across the 11 modules in the HackTheBox AI Red Team path.

## Progress

**Overall: 0 / 11 modules completed (0%)**

| # | Module | Difficulty | Tier | Sections | Interactive | Est. Time | Progress | Status |
|---|---|---|---|---|---|---|---|---|
| 01 | [[01-Fundamentals-of-AI/01-Fundamentals-of-AI\|Fundamentals of AI]] | Medium | — | 24 | 1 | — | 0% | not-started |
| 02 | [[02-Applications-of-AI-in-InfoSec/02-Applications-of-AI-in-InfoSec\|Applications of AI in InfoSec]] | — | — | 25 | 17 | — | 0% | not-started |
| 03 | [[03-Introduction-to-Red-Teaming-AI/03-Introduction-to-Red-Teaming-AI\|Introduction to Red Teaming AI]] | — | — | 11 | 3 | — | 0% | not-started |
| 04 | [[04-Prompt-Injection-Attacks/04-Prompt-Injection-Attacks\|Prompt Injection Attacks]] | Medium | 2 | 12 | 6 | 8h | 0% | not-started |
| 05 | [[05-LLM-Output-Attacks/05-LLM-Output-Attacks\|LLM Output Attacks]] | Medium | 2 | 14 | 6 | 8h | 0% | not-started |
| 06 | [[06-AI-Data-Attacks/06-AI-Data-Attacks\|AI Data Attacks]] | Hard | 2 | 25 | 22 | 3d | 0% | not-started |
| 07 | [[07-Attacking-AI-Application-and-System/07-Attacking-AI-Application-and-System\|Attacking AI — Application and System]] | Medium | 2 | 14 | 8 | 8h | 0% | not-started |
| 08 | [[08-AI-Evasion-Foundations/08-AI-Evasion-Foundations\|AI Evasion — Foundations]] | Medium | 2 | 12 | 2 | 8h | 0% | not-started |
| 09 | [[09-AI-Evasion-First-Order-Attacks/09-AI-Evasion-First-Order-Attacks\|AI Evasion — First-Order Attacks]] | Hard | 2 | 23 | 4 | 2d | 0% | not-started |
| 10 | [[10-AI-Evasion-Sparsity-Attacks/10-AI-Evasion-Sparsity-Attacks\|AI Evasion — Sparsity Attacks]] | Hard | 2 | 28 | 3 | 3d | 0% | not-started |
| 11 | [[11-AI-Privacy/11-AI-Privacy\|AI Privacy]] | Medium | 2 | 21 | 3 | 2d | 0% | not-started |

## Learning path

1. **Foundations (01, 02):** theory + hands-on infosec applications of ML (spam classifier, anomaly detection, malware classification).
2. **Red-team intro (03):** ML OWASP Top 10 + LLM OWASP Top 10.
3. **Prompt + output attacks (04, 05):** direct/indirect prompt injection, jailbreaks, insecure output handling (XSS/SQLi/command-injection via LLM).
4. **Data attacks (06):** label flipping, clean-label, trojans/backdoors, tensor steganography.
5. **App + system attacks (07):** model reverse engineering, insecure components, MCP server vulnerabilities.
6. **Evasion (08, 09, 10):** GoodWords → FGSM/DeepFool → ElasticNet/JSMA.
7. **Privacy (11):** membership inference attacks, DP-SGD, PATE.

Each module ends with a Skills Assessment (`Skills-Assessment.md` inside the module folder).

## References

- Spec: [[docs/superpowers/specs/2026-04-14-htb-ai-redteam-vault-design|Vault design]]
- Plan: [[docs/superpowers/plans/2026-04-14-htb-ai-redteam-vault-implementation|Implementation plan]]
- Repo: https://github.com/syn-systema/htb-ai-redteam
````

- [ ] **Step 3: Verify rendering-safe markdown**

Run:
```bash
wc -l "/home/farseer/Documents/htb-ai-redteam/00-Meta/HTB AI Red Team Path.md"
grep -c "^| " "/home/farseer/Documents/htb-ai-redteam/00-Meta/HTB AI Red Team Path.md"
```

Expected: >30 lines; at least 13 table rows (header + separator + 11 modules).

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add 00-Meta/
git commit -m "docs: add master progress index for 11-module path"
```

---

## Task 7: Create `Templates/Module Template.md`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/Templates/Module Template.md`

- [ ] **Step 1: Create `Templates/` directory**

Run:
```bash
mkdir -p /home/farseer/Documents/htb-ai-redteam/Templates
```

- [ ] **Step 2: Write the template file with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/Templates/Module Template.md`

````markdown
---
module_number: NN
module_name: ""
status: not-started
difficulty: ""
tier: ""
estimated_time: ""
sections_total: 0
sections_done: 0
started: ""
completed: ""
---

# Module NN — <Module Name>

> Copy this template into `NN-Module-Name/NN-Module-Name.md`, populate the frontmatter, and replace this blockquote with the module's real overview.

## Overview

*One-paragraph summary drawn from the HTB module description.*

## Prerequisites

- Prereq 1
- Prereq 2

## Recon plan

*(Optional — only for modules with a skills-assessment-style target.)*

```
Section 1  →  …
Section 2  →  …
```

## Sections

### 1. <Section name>

**Status:** `- [ ]`  |  **Type:** Theory / Interactive

*Summary of the section. For Interactive sections, link the notebook:*

> Notebook: [[notebooks/<name>.ipynb]]

**Key cells (inlined for vault searchability):**

```python
# paste the cell that matters most for future-you
```

**Takeaways:**
- …

---

### 2. <Section name>

**Status:** `- [ ]`  |  **Type:** Theory / Interactive

…

---

## Skills Assessment

See [[Skills-Assessment]].

## References

- HTB module URL: <paste>
- Related vault notes: [[…]]
- Papers / external refs: <paste>
````

- [ ] **Step 3: Verify frontmatter parses as valid YAML**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('/home/farseer/Documents/htb-ai-redteam/Templates/Module Template.md').read().split('---')[1])"
```

Expected: no output, exit code 0. If `yaml` import fails, try:
```bash
pip install pyyaml && python3 -c "import yaml; yaml.safe_load(open('/home/farseer/Documents/htb-ai-redteam/Templates/Module Template.md').read().split('---')[1])"
```

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add "Templates/Module Template.md"
git commit -m "docs: add module note template"
```

---

## Task 8: Create `Templates/Skills Assessment Template.md`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/Templates/Skills Assessment Template.md`

- [ ] **Step 1: Write the template with the exact contents below**

File: `/home/farseer/Documents/htb-ai-redteam/Templates/Skills Assessment Template.md`

````markdown
# Skills Assessment — Module NN

> Copy into `NN-Module-Name/Skills-Assessment.md` when starting the assessment. Replace this blockquote.

**Module:** [[NN-Module-Name]]
**Started:** YYYY-MM-DD
**Completed:** —
**Status:** not-started

## Target / Setup

*For remote (HTB-hosted LLM) assessments:*
- **Endpoint:** `https://…` (update when spawned)
- **Auth:** n/a | API key | session cookie
- **Tools used:** curl, Python `requests`, browser

*For local (Colab-based) assessments:*
- **Notebook:** [[notebooks/assessment.ipynb]]
- **Dataset:** see [[datasets]]
- **Model:** see notebook

## Questions

### Q1 — <question text>

**Approach:**

*Brief description of the technique / algorithm / payload type.*

**Commands / payload:**

```bash
# or: prompt text, or: python snippet
```

**Output:**

```
…paste the relevant response / metric / confidence score…
```

**Answer:** `__________________`

---

### Q2 — <question text>

**Approach:**

…

**Answer:** `__________________`

---

*(Repeat for remaining questions.)*

## Cleanup

*(If remote:)*
```bash
# revoke any API keys / session cookies, remove any local /etc/hosts entries
```

## Lessons Learned

- What worked the first time
- What didn't — and why
- What I'll do differently next time
````

- [ ] **Step 2: Verify file integrity**

Run:
```bash
ls -la "/home/farseer/Documents/htb-ai-redteam/Templates/Skills Assessment Template.md"
head -5 "/home/farseer/Documents/htb-ai-redteam/Templates/Skills Assessment Template.md"
```

Expected: file present, first non-blank line begins `# Skills Assessment`.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add "Templates/Skills Assessment Template.md"
git commit -m "docs: add skills assessment template"
```

---

## Task 9: Create `Templates/Notebook Scaffold Template.ipynb`

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/Templates/Notebook Scaffold Template.ipynb`

- [ ] **Step 1: Write the notebook file with the exact JSON contents below**

File: `/home/farseer/Documents/htb-ai-redteam/Templates/Notebook Scaffold Template.ipynb`

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Module NN — <Exercise Name>\n",
        "\n",
        "Colab notebook scaffold. Duplicate into the module's `notebooks/` folder and rename before editing."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Clone the vault repo so shared helpers are importable\n",
        "!git clone https://github.com/syn-systema/htb-ai-redteam.git /content/repo 2>/dev/null || (cd /content/repo && git pull)\n",
        "import sys\n",
        "if '/content/repo' not in sys.path:\n",
        "    sys.path.append('/content/repo')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Standard imports + GPU check\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Torch: {torch.__version__}')\n",
        "print(f'Device: {device}')\n",
        "if device.type == 'cuda':\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Load data\n",
        "\n",
        "Use `shared.data` loaders where possible. Add a new loader there if this dataset isn't handled yet."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# from shared.data import load_mnist\n",
        "# train_loader, test_loader, meta = load_mnist(batch_size=128)\n",
        "pass"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Model / attack implementation"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Evaluation + visualization"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10"
    },
    "accelerator": "GPU",
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the JSON is valid and loads as a notebook**

Run:
```bash
python3 -c "import json; nb = json.load(open('/home/farseer/Documents/htb-ai-redteam/Templates/Notebook Scaffold Template.ipynb')); print(f'cells: {len(nb[\"cells\"])}'); print(f'nbformat: {nb[\"nbformat\"]}.{nb[\"nbformat_minor\"]}')"
```

Expected output:
```
cells: 7
nbformat: 4.5
```

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add "Templates/Notebook Scaffold Template.ipynb"
git commit -m "docs: add colab notebook scaffold template"
```

---

## Task 10: Create `shared/` Python package

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/shared/__init__.py`
- Create: `/home/farseer/Documents/htb-ai-redteam/shared/data.py`
- Create: `/home/farseer/Documents/htb-ai-redteam/shared/viz.py`
- Create: `/home/farseer/Documents/htb-ai-redteam/shared/attacks.py`

- [ ] **Step 1: Create the directory**

Run:
```bash
mkdir -p /home/farseer/Documents/htb-ai-redteam/shared
```

- [ ] **Step 2: Write `shared/__init__.py`**

File: `/home/farseer/Documents/htb-ai-redteam/shared/__init__.py`

```python
"""Shared helpers for HTB AI Red Team notebooks.

Import into Colab via:
    !git clone https://github.com/syn-systema/htb-ai-redteam.git /content/repo
    import sys; sys.path.append('/content/repo')
    from shared.data import load_mnist
"""
```

- [ ] **Step 3: Write `shared/data.py`**

File: `/home/farseer/Documents/htb-ai-redteam/shared/data.py`

```python
"""Dataset loaders for HTB AI Red Team exercises.

Each loader returns (train_loader, test_loader, meta_dict).
"""

from __future__ import annotations


def load_mnist(batch_size: int = 128, data_root: str = "./data"):
    """Return MNIST train/test DataLoaders + metadata.

    Downloads via torchvision on first call. Subsequent calls reuse the cache.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    meta = {
        "name": "MNIST",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }
    return train_loader, test_loader, meta
```

- [ ] **Step 4: Write `shared/viz.py`**

File: `/home/farseer/Documents/htb-ai-redteam/shared/viz.py`

```python
"""Plotting helpers for HTB AI Red Team exercises."""

from __future__ import annotations


def plot_loss_curves(train_losses, test_losses=None, title: str = "Loss"):
    """Plot training (and optional test) loss per epoch."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    if test_losses is not None:
        ax.plot(test_losses, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, ax
```

- [ ] **Step 5: Write `shared/attacks.py`**

File: `/home/farseer/Documents/htb-ai-redteam/shared/attacks.py`

```python
"""Perturbation / norm helpers for adversarial attacks."""

from __future__ import annotations


def clip_linf(x, eps: float):
    """Clip tensor to the L_inf ball of radius eps around 0, element-wise."""
    import torch

    return torch.clamp(x, -eps, eps)


def clip_l2(x, eps: float):
    """Project tensor onto the L2 ball of radius eps (per-sample)."""
    import torch

    flat = x.view(x.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = (eps / norms).clamp(max=1.0)
    return (flat * factor).view_as(x)
```

- [ ] **Step 6: Verify every file parses as valid Python**

Run:
```bash
python3 -m py_compile /home/farseer/Documents/htb-ai-redteam/shared/__init__.py \
                      /home/farseer/Documents/htb-ai-redteam/shared/data.py \
                      /home/farseer/Documents/htb-ai-redteam/shared/viz.py \
                      /home/farseer/Documents/htb-ai-redteam/shared/attacks.py
```

Expected: no output, exit code 0. (A syntax error would print a traceback.)

- [ ] **Step 7: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add shared/
git commit -m "feat(shared): add data/viz/attacks helpers importable from colab"
```

---

## Task 11: Scaffold all 11 module directories

**Files:** Per module NN-Module-Name (×11):
- Create: `NN-Module-Name/NN-Module-Name.md`
- Create: `NN-Module-Name/Skills-Assessment.md`
- Create: `NN-Module-Name/datasets.md`
- Create: `NN-Module-Name/notebooks/.gitkeep`

- [ ] **Step 1: Create the module scaffold script as a throwaway shell command**

Run this whole block in one go (paste into terminal):

```bash
cd /home/farseer/Documents/htb-ai-redteam

MODULES=(
  "01-Fundamentals-of-AI"
  "02-Applications-of-AI-in-InfoSec"
  "03-Introduction-to-Red-Teaming-AI"
  "04-Prompt-Injection-Attacks"
  "05-LLM-Output-Attacks"
  "06-AI-Data-Attacks"
  "07-Attacking-AI-Application-and-System"
  "08-AI-Evasion-Foundations"
  "09-AI-Evasion-First-Order-Attacks"
  "10-AI-Evasion-Sparsity-Attacks"
  "11-AI-Privacy"
)

for M in "${MODULES[@]}"; do
  mkdir -p "$M/notebooks"
  touch "$M/notebooks/.gitkeep"

  # Module note seeded from template, with module_number + module_name pre-filled
  NUM="${M%%-*}"
  NAME="${M#*-}"
  NAME="${NAME//-/ }"

  cat > "$M/$M.md" <<EOF
---
module_number: $NUM
module_name: "$NAME"
status: not-started
difficulty: ""
tier: ""
estimated_time: ""
sections_total: 0
sections_done: 0
started: ""
completed: ""
---

# Module $NUM — $NAME

> Populate from the HTB module description. See [[Templates/Module Template]] for the full skeleton.

## Overview

*Replace with one-paragraph summary from HTB.*

## Sections

### 1. <Section name>

**Status:** - [ ]  |  **Type:** Theory / Interactive

---

## Skills Assessment

See [[Skills-Assessment]].

## References

- HTB module URL: <paste>
EOF

  # Skills Assessment stub
  cat > "$M/Skills-Assessment.md" <<EOF
# Skills Assessment — Module $NUM

**Module:** [[$M]]
**Started:** —
**Completed:** —
**Status:** not-started

> Copy in the full skeleton from [[Templates/Skills Assessment Template]] when you begin the assessment.
EOF

  # datasets.md stub
  cat > "$M/datasets.md" <<EOF
# Datasets — Module $NUM ($NAME)

| Dataset | Source | Drive path (when mounted) | Used by |
|---------|--------|---------------------------|---------|
|         |        |                           |         |
EOF
done

echo "Done. Created scaffolds for ${#MODULES[@]} modules."
```

Expected: `Done. Created scaffolds for 11 modules.` and no errors.

- [ ] **Step 2: Verify the scaffolding landed correctly**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
ls -d [0-9][0-9]-*/ | wc -l
find [0-9][0-9]-* -maxdepth 2 -type f | sort | head -20
```

Expected: `ls` count = `11`. `find` output includes `.md` + `.gitkeep` files across all modules.

- [ ] **Step 3: Spot-check one module for frontmatter correctness**

Run:
```bash
head -15 /home/farseer/Documents/htb-ai-redteam/06-AI-Data-Attacks/06-AI-Data-Attacks.md
```

Expected: frontmatter with `module_number: 06` and `module_name: "AI Data Attacks"`.

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add [0-9][0-9]-*/
git status --short | head
git commit -m "feat: scaffold 11 module folders with note/assessment/datasets stubs"
```

Expected: 44 files changed (11 modules × 4 files each).

---

## Task 12: Create `Assets/` directory placeholder

**Files:**
- Create: `/home/farseer/Documents/htb-ai-redteam/Assets/.gitkeep`

- [ ] **Step 1: Create the directory + placeholder**

Run:
```bash
mkdir -p /home/farseer/Documents/htb-ai-redteam/Assets
touch /home/farseer/Documents/htb-ai-redteam/Assets/.gitkeep
```

- [ ] **Step 2: Commit**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git add Assets/
git commit -m "chore: add Assets/ dir for screenshots and diagrams"
```

---

## Task 13: Push to GitHub

This step requires user authentication. Run when you (the user) are at the terminal.

- [ ] **Step 1: Confirm the commit history is clean**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git log --oneline
```

Expected: 11 commits in order:
1. `docs: initial design spec + implementation plan`
2. `chore: add gitignore ...`
3. `docs: add CLAUDE.md ...`
4. `docs: add README ...`
5. `docs: add master progress index ...`
6. `docs: add module note template`
7. `docs: add skills assessment template`
8. `docs: add colab notebook scaffold template`
9. `feat(shared): add data/viz/attacks helpers ...`
10. `feat: scaffold 11 module folders ...`
11. `chore: add Assets/ dir ...`

- [ ] **Step 2: Push (user runs)**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git push -u origin main
```

Expected: successful push. If the remote was non-empty (per Task 1 Step 3), this will fail — stop and reconcile.

- [ ] **Step 3: Verify remote state**

Run:
```bash
cd /home/farseer/Documents/htb-ai-redteam
git log --oneline origin/main | head -3
```

Expected: the top 3 commits on `origin/main` match the last 3 local commits.

---

## Task 14: Open in Obsidian and verify vault renders

This task is manual verification by the user.

- [ ] **Step 1: Open the vault in Obsidian**

Action: Obsidian → `Open folder as vault` → select `/home/farseer/Documents/htb-ai-redteam/`.

- [ ] **Step 2: Verify file tree**

Expected in the file-tree pane:
- `00-Meta/` containing `HTB AI Red Team Path.md`
- `01-Fundamentals-of-AI/` through `11-AI-Privacy/`, each containing:
  - a module note `NN-Module-Name.md`
  - `Skills-Assessment.md`
  - `datasets.md`
  - `notebooks/` (empty, only `.gitkeep`)
- `Templates/` with three files
- `shared/` with four Python files
- `Assets/` (empty)
- `README.md` + `CLAUDE.md` at root

- [ ] **Step 3: Open the master index**

Action: click `00-Meta/HTB AI Red Team Path.md`.

Expected: the progress table renders with all 11 modules. Click one of the wikilinks (e.g. `Fundamentals of AI`) — it should open the module note. If wikilinks show as plain text, enable "Reading mode" or check that Obsidian's default-link format is set to `Shortest path when possible`.

- [ ] **Step 4: Open `CLAUDE.md`**

Expected: renders with the Teaching Mode B section visible.

- [ ] **Step 5: Open the Templates**

Action: click `Templates/Module Template.md` and `Templates/Skills Assessment Template.md`. Expected: both render with frontmatter visible (in Source mode) or hidden (in Reading mode).

- [ ] **Step 6: No commit — this task is verification only.**

---

## Self-Review

**Spec coverage check** (mapped against the design doc sections):

| Spec section | Covered by |
|---|---|
| Repo layout | Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| Per-module folder | Task 11 |
| `CLAUDE.md` (10 sub-sections) | Task 4 |
| `README.md` (5 sections) | Task 5 |
| `.gitignore` policy | Task 3 |
| `shared/` package | Task 10 |
| `Templates/` (3 files) | Tasks 7, 8, 9 |
| `00-Meta/HTB AI Red Team Path.md` | Task 6 |
| Colab integration pattern | Task 9 (notebook) + Task 4 (CLAUDE.md) |
| Notebook output policy (keep outputs) | Task 3 (`.gitignore` does NOT exclude `.ipynb`) + Task 4 (CLAUDE.md) |
| Teaching Mode B operationalized | Task 4 |
| Success criteria — `git clone` → Obsidian opens | Task 13 (push) + Task 14 (open) |
| Success criteria — templates discoverable | Tasks 7, 8, 9 under `Templates/` |
| Success criteria — Colab imports from `shared.data` | Task 10 (`load_mnist`) + Task 9 (scaffold with import example) |

No uncovered requirements.

**Placeholder scan:** none of the steps use "TBD", "TODO", "fill in later", "similar to task N". Every file's full contents are inline. Every command has expected output.

**Type consistency:** the function signatures in `shared/data.py` (`load_mnist → (train_loader, test_loader, meta)`) match the commented-out example in Task 9's notebook scaffold. The frontmatter fields in Task 7's Module Template match those referenced in Task 4's CLAUDE.md and Task 11's scaffold script.

**Open-questions note:** Task 1 Step 3 handles the spec's open question "does the GitHub repo already have content?" by explicitly branching on what `git ls-remote` returns and stopping for user reconciliation when the repo is non-empty. Colab secrets strategy is appropriately deferred (no module needs them yet).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-14-htb-ai-redteam-vault-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
