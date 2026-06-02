# ELEN90088 — System Optimisation and Machine Learning

Coursework repository for [ELEN90088 System Optimisation and Machine Learning](https://handbook.unimelb.edu.au/subjects/elen90088) at the University of Melbourne (2026).

This repository collects my solutions, supporting notes, and the final project for the subject. Weekly exercises live under `Exercises/` (with my own notes and supporting materials under `Exercises/docs/`); the project brief and starter materials live under `Project-Description/`. The Project 2 (Hands-on LLMs) work is split across two folders: `Project-LLM/` holds the Parts 1–5 starter notebook together with the written report and oral slide deck, and `mini-project-DoRA/` holds the graded mini-project (Part 6) — a DoRA paper reproduction. See [Project-LLM/README.md](./Project-LLM/README.md) and [mini-project-DoRA/README.md](./mini-project-DoRA/README.md) for the respective write-ups and reproduce instructions.

---

## Repository Structure

```
.
├── Exercises/                          # Weekly exercises — my solutions, notes, paired .py scripts
│   ├── SOML2026_Exercise_{1,2,3}.ipynb # My solutions (Ex 2 & 3 also Jupytext-paired as .py)
│   ├── Reference Solution/             # Official solutions (Ex 1 & 2)
│   ├── Zhiqi's Solution/               # Peer reference solution (Ex 2)
│   └── docs/                           # My notes and supporting materials
│
├── Project-Description/                # Teaching-team project brief, starter notebook, oral schedule
├── Project-LLM/                        # Project 2: Parts 1–5 starter notebook, written report, oral deck
├── mini-project-DoRA/                  # Project 2 Part 6: graded DoRA reproduction (self-contained uv project)
│
├── requirements.txt                    # pip deps for the Exercises (mini-project uses uv)
├── LICENSE.md                          # Usage notes (educational reference + upstream licenses)
└── README.md
```

The two project folders each have their own README with a full file-by-file map: [Project-LLM/README.md](./Project-LLM/README.md) and [mini-project-DoRA/README.md](./mini-project-DoRA/README.md).

Progress:

- [x] Exercise 1 — Convexity, linear models, logistic regression
- [x] Exercise 2 — SVM, clustering (K-Means / GMM), DNN, VAE
- [x] Exercise 3 — Duality and the duality gap, KKT conditions, SVM (binary + importance-weighted), backpropagation
- [x] Subject Project (Option 2 — Hands-on LLMs): starter notebook; mini-project (DoRA reproduction) results — Tier 2 (the canonical study, 36 runs + zero-shot) plus a separate Tier-3 enrichment (24 runs at 10k cs170k steps × 4 seeds).

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- JupyterLab / Jupyter Notebook (or VS Code with the Jupyter extension)

### Setup (Exercises)

Clone the repository and install the exercise dependencies into a virtual environment:

```bash
git clone https://github.com/<your-username>/ELEN90088-SOML.git
cd ELEN90088-SOML

python3 -m venv .venv_soml
source .venv_soml/bin/activate         # Windows: .venv_soml\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`.venv_soml/` is gitignored.

### Running the Notebooks

```bash
jupyter lab
```

Then open any notebook under `Exercises/` and run the cells top-to-bottom. In VS Code, point the interpreter/kernel at `.venv_soml/bin/python` when prompted.

### Project 2 (Hands-on LLMs)

I developed Project 2 locally and ran it on the **HPC (Spartan, via Open OnDemand)**. The work is split across two folders:

- [`Project-LLM/`](./Project-LLM/) — the Parts 1–5 starter notebook (`SOML_LLM_project.ipynb`), plus the written **report** (`Report/`, including the compiled `main.pdf`) and the **oral slide deck** (`Oral/`). The notebook was edited locally and executed on Spartan, reading `SPARTAN_PROJECT_DIR` to route the Hugging Face / transformers / datasets caches under that path so model weights persisted across sessions. See [`Project-LLM/README.md`](./Project-LLM/README.md).
- [`mini-project-DoRA/`](./mini-project-DoRA/) — Part 6 mini-project: a DoRA paper reproduction extended into a regime-contrast generalization study (Tier 2: 36 training runs + zero-shot baseline), with a separate Tier-3 enrichment at 10k cs170k steps × 4 seeds (24 additional runs). Self-contained `uv` project with its own dependencies, scripts, tests, configs, results, and figures. See [`mini-project-DoRA/README.md`](./mini-project-DoRA/README.md) for the full story and reproduce instructions.

See [`Project-Description/Project_2_Hands-on_LLMs.md`](./Project-Description/Project_2_Hands-on_LLMs.md) for the project brief and [`Project-Description/oral-info.md`](./Project-Description/oral-info.md) for the oral assessment schedule. The two `.yaml` files in `Project-Description/` are conda env specs provided by the teaching team for students who prefer to run Parts 1–5 locally; I didn't use them.

---

## Dependencies

Listed in [`requirements.txt`](./requirements.txt):

| Package        | Purpose                                      |
| -------------- | -------------------------------------------- |
| `numpy`        | Numerical computing, linear algebra          |
| `scipy`        | Scientific computing (stats, distributions)  |
| `sympy`        | Symbolic mathematics (duality / KKT work)    |
| `pandas`       | Data manipulation                            |
| `matplotlib`   | Plotting                                     |
| `seaborn`      | Statistical visualisation                    |
| `scikit-learn` | Classical ML algorithms, datasets, utilities |
| `cvxpy`        | Disciplined convex programming               |
| `torch`        | Deep learning                                |
| `torchvision`  | Vision datasets (CIFAR-10) and transforms    |

## Notes on Use

- The notebooks are my own working solutions for study purposes. They are shared as a personal record and for discussion with peers.
- If you are currently enrolled in ELEN90088, please consult your subject's academic integrity policy before referring to any material here. Copying submissions is a breach of the University's Academic Integrity rules.
- Official solutions in this repository were provided by the teaching team; redistribution beyond personal reference is not intended.

**Import to html (then to pdf)**

If export to html fails: (In my condition with VSCode, `pyzmq<25` is required while my pip list `pyzmq=27.1`).

```bash
pip install nbconvert
# Now export button should work, or:
jupyter nbconvert --to html SOML2026_Exercise_x.ipynb
```

## License

This repository is provided for personal and educational reference. See [LICENSE.md](./LICENSE.md) for usage details and notes on the upstream DoRA code license.
