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
CLAUDE.md                        # Operating rules for Claude Code sessions
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
