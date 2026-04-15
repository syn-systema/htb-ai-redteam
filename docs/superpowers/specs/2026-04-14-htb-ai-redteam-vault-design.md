# HTB AI Red Team Vault — Design

**Date:** 2026-04-14
**Status:** Approved — ready for implementation plan
**Target repo:** `https://github.com/syn-systema/htb-ai-redteam`
**Local path:** `~/Documents/htb-ai-redteam/`

## Purpose

An Obsidian vault + git repo for tracking progress through HTB's AI Red Team path (11 modules). Modeled on the existing `htb-notes` (CPTS) vault, adapted for the ML-heavy content, Colab-based execution, and per-module skills assessments.

## Scope

In scope:
- Repo and folder structure for all 11 modules.
- Templates for module notes, skills assessments, and notebook scaffolds.
- Shared Python helpers importable into Colab notebooks via `git clone`.
- `CLAUDE.md` operating rules specialized for AI Red Team work.
- Progress tracking index (`00-Meta/HTB AI Red Team Path.md`).
- Git/notebook hygiene (`.gitignore`, output retention policy).

Out of scope (future work):
- Populating section content for any specific module — that happens during the live study sessions.
- CI / GitHub Actions.
- Cross-references to the CPTS vault (intentionally kept as separate vaults).

## Context & constraints

- **Execution environment:** Google Colab Pro (A100/L4 GPU, background execution, 24h sessions). No local Jupyter runtime needed. User edits via Antigravity (VS Code fork with Colab integration).
- **Git-backed:** the vault IS the repo. Remote at `syn-systema/htb-ai-redteam`. Push-based workflow; user runs all git commands.
- **Prior art:** existing `~/Documents/htb-notes` CPTS vault — flat module-numbered in spirit (though it uses phase grouping); one markdown file per module with sections as `##` headers; Templates/ folder at root.
- **Course shape:** 11 modules, some theory-heavy (Fundamentals, Intro RT), some interactive-heavy (AI Data Attacks has 22 interactives). Each module has its own Skills Assessment — no single capstone.

## Decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Organization | Flat module-numbered (01–11) | Simplest; matches HTB's own ordering; no grouping decisions to relitigate. |
| Vault location | `~/Documents/htb-ai-redteam/` (sibling to htb-notes) | Clean separation of mental context and CLAUDE.md rules. |
| Teaching mode | **Stance B** — I scaffold, you run | Removes typing friction for boilerplate while preserving learning reps on execution + interpretation. Remote LLM work stays fully hands-off. |
| Notebook runtime | Colab Pro via Antigravity | No local env setup tax; free GPU; background execution for long training. |
| Notebook storage | Colab for runtime, `.ipynb` snapshot exported to `notebooks/` on completion | Vault stays self-contained; git diffable; offline readable. |
| Notebook outputs | Keep outputs in committed `.ipynb` | Learning artifact — loss curves, confusion matrices, adv images are the value. No `nbstripout`. |
| File granularity | One markdown note per module, sections as `##` | Matches existing CPTS convention and the user's mental model. |

## Repo layout

```
~/Documents/htb-ai-redteam/
├── README.md
├── CLAUDE.md
├── .gitignore
├── 00-Meta/
│   └── HTB AI Red Team Path.md
├── Templates/
│   ├── Module Template.md
│   ├── Skills Assessment Template.md
│   └── Notebook Scaffold Template.ipynb
├── shared/
│   ├── __init__.py
│   ├── data.py
│   ├── viz.py
│   └── attacks.py
├── Assets/
├── docs/superpowers/specs/
│   └── 2026-04-14-htb-ai-redteam-vault-design.md  (this file)
├── 01-Fundamentals-of-AI/
├── 02-Applications-of-AI-in-InfoSec/
├── 03-Introduction-to-Red-Teaming-AI/
├── 04-Prompt-Injection-Attacks/
├── 05-LLM-Output-Attacks/
├── 06-AI-Data-Attacks/
├── 07-Attacking-AI-Application-and-System/
├── 08-AI-Evasion-Foundations/
├── 09-AI-Evasion-First-Order-Attacks/
├── 10-AI-Evasion-Sparsity-Attacks/
└── 11-AI-Privacy/
```

### Per-module folder

```
NN-Module-Name/
├── NN-Module-Name.md         # Main note, sections as ## headers
├── Skills-Assessment.md       # Module's own assessment
├── notebooks/                 # .ipynb snapshots exported from Colab on completion
└── datasets.md                # Dataset → Drive mount path / source URL
```

#### `datasets.md` format

Minimal markdown table — one row per dataset used anywhere in the module:

```markdown
| Dataset | Source | Drive path (when mounted) | Used by |
|---------|--------|---------------------------|---------|
| MNIST   | torchvision built-in | n/a (downloads at runtime) | `fgsm_attack.ipynb` |
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | `/content/drive/MyDrive/htb-ai-redteam/data/nsl-kdd/` | `network-anomaly.ipynb` |
```

## Component specs

### `CLAUDE.md` (repo root)

Mirrors the shape of the htb-notes CLAUDE.md but with the following specialized content:

1. **Project Overview** — one paragraph: Obsidian vault for HTB AI Red Team path, 11 modules, Colab Pro + Antigravity runtime.
2. **Directory Structure** — the repo layout above, briefly annotated.
3. **Key Files** — links to `00-Meta/HTB AI Red Team Path.md`, Templates, `shared/`.
4. **File Conventions** — one markdown per module, sections as `##`, frontmatter fields (`status`, `difficulty`, `tier`, `estimated_time`, `started`, `completed`, `sections_total`, `sections_done`).
5. **Live-Session Workflow — Teaching Mode B (IMPORTANT):**
   - What I do: scaffold notebook cells into module notes as ```` ```python ```` blocks, author helpers in `shared/`, draft prose, interpret Colab output the user pastes, update the note in real time.
   - What the user does: runs every Colab cell, runs every `!pip install`, runs every training/attack, submits every Skills Assessment answer.
   - **Remote LLM work is fully hands-off**: prompt injection against HTB-hosted targets, LLM output attacks against hosted endpoints, and Skills Assessment submissions are the user's alone.
   - Git: I propose commit messages; user runs `git commit` / `git push`.
6. **Colab Integration Pattern** — every Colab notebook starts with:
   ```python
   !git clone https://github.com/syn-systema/htb-ai-redteam.git /content/repo
   import sys; sys.path.append('/content/repo')
   from shared.data import load_mnist    # example
   ```
7. **Notebook Snapshot Flow** — after an exercise: `File → Download .ipynb` in Colab → drop into `NN-Module-Name/notebooks/` → I update the markdown note to link to it and inline key cells.
8. **Real-time vault updates** — carried over from htb-notes: log every command, output, interpretation, and pivot into the relevant note in-line, not at session end. (Already a saved memory.)
9. **Progress Tracking** — how to update module note + master index when a section completes.
10. **Common Tasks** — "add a new module note", "snapshot a notebook", "mark a section complete".

### `README.md` (repo root)

Audience: future-you + anyone who stumbles on the repo. Sections:
1. One-paragraph what-this-is.
2. Setup: clone, open in Antigravity, open the vault in Obsidian, configure Colab Pro.
3. Workflow: how a study session flows (module note → Colab → snapshot → commit).
4. Progress table — static markdown (modules/progress/status columns) with a link to `00-Meta/HTB AI Red Team Path.md` for the canonical version. No shields.io / dynamic badges (avoid external dependencies).
5. Directory layout.

### `.gitignore`

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
# NB: CSVs under 1 MB are fine to commit (small reference tables); anything larger
# lives in Drive and is tracked via datasets.md.

# Secrets
.env

# OS
.DS_Store
Thumbs.db
```

### `shared/` (Python package)

Minimal to start. Grown as modules demand shared logic.

- `__init__.py` — empty marker.
- `data.py` — dataset loaders (MNIST, CIFAR-10, NSL-KDD, UCI SMS). One function per dataset, returning standard PyTorch `DataLoader` + a small metadata dict.
- `viz.py` — `plot_loss_curves`, `plot_confusion_matrix`, `show_adversarial_grid(original, adversarial, labels, preds)`.
- `attacks.py` — perturbation norm helpers: `clip_linf(x, eps)`, `clip_l2(x, eps)`, `clip_l0(x, k)`.

Initial module stubs contain `pass` or a single minimal function; content grows organically as modules introduce new patterns.

### `Templates/`

- **Module Template.md** — frontmatter:
  ```yaml
  ---
  module_number: NN
  module_name: ""
  status: not-started   # not-started | in-progress | completed
  difficulty: ""        # Easy | Medium | Hard
  tier: ""
  estimated_time: ""
  sections_total: 0
  sections_done: 0
  started: ""
  completed: ""
  ---
  ```
  Body: Overview, Prerequisites, Sections (`## Section 1 — ...`, etc.), Skills Assessment cross-link (`[[Skills-Assessment]]`), References.

- **Skills Assessment Template.md** — Target info box (if remote), Questions (Q1–QN) with "Methodology" + "Answer" blocks, Cleanup, Lessons Learned.

- **Notebook Scaffold Template.ipynb** — three starter cells:
  1. Repo clone + `sys.path.append` + imports.
  2. GPU availability check (`torch.cuda.is_available()`, device select).
  3. Markdown header matching the exercise name.

### `00-Meta/HTB AI Red Team Path.md`

Master index. Table:

| # | Module | Difficulty | Tier | Sections | Interactive | Est. Time | Progress | Status |
|---|---|---|---|---|---|---|---|---|
| 01 | Fundamentals of AI | Medium | — | 24 | 1 | — | 0% | not-started |
| 02 | Applications of AI in InfoSec | — | — | 25 | 17 | — | 0% | not-started |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 11 | AI Privacy | Medium | 2 | 21 | 3 | 2d | 0% | not-started |

Below: per-module anchor links + a simple progress bar (N/11 completed).

## Data & state flow

1. **Start a module:** user updates frontmatter of the module note (`status: in-progress`, `started: YYYY-MM-DD`).
2. **Read a theory section:** I summarize into the module note under the appropriate `##`.
3. **Hit an interactive:** I scaffold a notebook into `notebooks/` (template + starter cells) and update the module note with a link.
4. **User runs the notebook in Colab.** Pastes key outputs back.
5. **I interpret + update the note** with findings, commentary, gotchas.
6. **User exports `.ipynb`** on completion, drops into `notebooks/`, I update the note's "Outputs" section.
7. **Section complete:** `[ ]` → `[x]`, bump `sections_done`, update master index.
8. **Module complete:** `status: completed`, `completed: YYYY-MM-DD`, master index updated.
9. **Commit:** I propose a message, user runs `git commit` + `git push`.

## Trade-offs acknowledged

- **Flat numbering** (vs. phase grouping) loses thematic clustering — the three Evasion modules are spread across 08/09/10. Accepted because user preferred simplicity and HTB's own ordering.
- **Keeping notebook outputs in git** creates noisy diffs. Accepted because the outputs are the learning artifact; disk is cheap; no collaborators to annoy.
- **Colab dependency** means the repo isn't self-executable. Accepted because local env setup for the harder modules (Trojan CNN, DP-SGD) would be painful and Colab Pro gives better hardware.
- **Per-module Skills Assessment files** (vs. one combined) creates 11 separate files instead of 1. Accepted because each assessment is independent; treating them together would conflate unrelated skills.

## Open questions for implementation

- Does the GitHub repo at `syn-systema/htb-ai-redteam` already have content? If yes, implementation plan needs a "reconcile existing content" step. (Verify before scaffolding.)
- Colab API keys / secrets strategy if any module needs them — defer until a module actually requires it.

## Success criteria

- `git clone` the repo → open in Obsidian → master index renders, module folders are there, templates are discoverable.
- Starting a new module: copy `Module Template.md` → populate frontmatter → fill in sections. No manual folder creation required beyond that template-copy step (or a future helper script).
- First Colab notebook runs the repo-clone boilerplate without error and successfully imports from `shared.data`.
- `CLAUDE.md` is concrete enough that a fresh Claude session picks up the Teaching Mode B rules immediately.
