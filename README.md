# ELEN90088 — System Optimisation and Machine Learning

Coursework repository for [ELEN90088 System Optimisation and Machine Learning](https://handbook.unimelb.edu.au/subjects/elen90088) at the University of Melbourne (2026).

This repository collects my solutions, supporting notes, and the final project for the subject. Weekly exercises live under `Exercises/` (with my own notes and supporting materials under `Exercises/docs/`), and the project brief together with its starter materials under `Project-Description/`. `Project-LLM/` is the work-in-process folder.

---

## Repository Structure

```
.
├── Exercises/
│   ├── SOML2026_Exercise_1.ipynb       # Exercise 1 — my solution
│   ├── SOML2026_Exercise_1_Sol.ipynb   # Official solution (reference)
│   ├── SOML2026_Exercise_2.ipynb       # Exercise 2 — my solution
│   ├── SOML2026_Exercise_2.py          # Jupytext-paired script for Exercise 2
│   ├── Zhiqi's Solution/               # Peer reference solution for Ex 2
│   └── docs/                           # My notes and supporting materials
│
├── Project-Description/
│   ├── SOML_Project_26S1.md            # Overall project brief
│   ├── Project_1_Learn_to_Optimise.md  # Option 1 — learn-to-optimise
│   ├── Project_2_Hands-on_LLMs.md      # Option 2 — hands-on LLMs
│   ├── SOML_LLM_project.ipynb          # Starter notebook (Option 2)
│   ├── SOML_LLM_project_MacOS.yaml     # Conda env spec (macOS, optional)
│   └── SOML_LLM_project_Windows.yaml   # Conda env spec (Windows / CUDA, optional)
├── Project-LLM/
│   ├── SOML_LLM_project.ipynb          # Working project notebook (edited locally, run on HPC)
│   └── Results-ipynb/                  # Outputs and write-ups for selected parts
├── requirements.txt                    # pip requirements for the exercises
└── README.md
```

Progress:

- [x] Exercise 1 — Convexity, linear models, logistic regression
- [x] Exercise 2 — SVM, clustering (K-Means / GMM), DNN, VAE
- [ ] Exercise 3
- [ ] Subject Project (Option 2 — Hands-on LLMs) — starter notebook complete; mini project in progress

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

I run the subject project on the **University of Melbourne HPC (Spartan, via Open OnDemand)**. `Project-LLM/SOML_LLM_project.ipynb` is edited locally and executed remotely; the notebook reads `SPARTAN_PROJECT_DIR` and routes the Hugging Face / transformers / datasets caches and working directories under that path so model weights persist across sessions. See `Project-Description/Project_2_Hands-on_LLMs.md` for the brief. The two `.yaml` files in `Project-Description/` are conda env specs provided by the teaching team for students who prefer to run the project locally; I don't use them.

---

## Dependencies

Listed in [`requirements.txt`](./requirements.txt):

| Package        | Purpose                                      |
| -------------- | -------------------------------------------- |
| `numpy`        | Numerical computing, linear algebra          |
| `scipy`        | Scientific computing (stats, distributions)  |
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

This repository is provided for personal and educational reference. No explicit license is granted.
