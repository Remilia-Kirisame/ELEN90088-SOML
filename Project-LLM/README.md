# Project-LLM — Parts 1–5 (starter notebook scaffold)

This folder holds the **starter notebook** for Project 2 (Hands-on LLMs) of ELEN90088 — the work for **Parts 1–5** of the project brief. It is *scaffolding*, not the graded deliverable.

The graded **mini-project (Part 6)** — a DoRA paper reproduction extended into a regime-contrast generalization study — lives in [`../mini-project-DoRA/`](../mini-project-DoRA/). That folder has its own README, dependencies (`uv`-managed), tests, scripts, configs, results, and figures.

## What's here

- `SOML_LLM_project.ipynb` — the working notebook, edited locally and executed on UniMelb HPC (Spartan, via Open OnDemand). Uses Mistral / `Phi-3.5-mini-instruct` (model choice does not affect marks per the brief).
- `SOML_LLM_project.py` — jupytext-paired Python script (the source of truth; edit this and `jupytext --sync` to refresh the `.ipynb`).
- `Results-ipynb/` — outputs and write-ups for selected parts (Part 3a, Part 4.3.3) + a small sentiment-results CSV + an object-detection screenshot.

## How to run

The notebook is intended for HPC execution; the local repo is the editing surface. The shared exercise environment at the repo root (`.venv_soml`, from `../requirements.txt`) is fine for cell editing and lightweight inspection but does not include the LLM stack (transformers, peft, etc.) — those are loaded on Spartan via the conda env specs in `../Project-Description/`.

For the canonical project run instructions and the brief itself, see [`../Project-Description/Project_2_Hands-on_LLMs.md`](../Project-Description/Project_2_Hands-on_LLMs.md).

## See also

- [`../mini-project-DoRA/README.md`](../mini-project-DoRA/README.md) — Part 6 mini-project (the graded deliverable).
- [`../Project-Description/Project_2_Hands-on_LLMs.md`](../Project-Description/Project_2_Hands-on_LLMs.md) — project brief.
- [`../Project-Description/oral-info.md`](../Project-Description/oral-info.md) — oral assessment schedule and instructions.
