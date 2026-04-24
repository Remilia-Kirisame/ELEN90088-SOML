# ELEN90088 — System Optimisation and Machine Learning

Coursework repository for [ELEN90088 System Optimisation and Machine Learning](https://handbook.unimelb.edu.au/subjects/elen90088) at the University of Melbourne (2026).

This repository collects my solutions, supporting notes, and the final project for the subject. Weekly exercises live under `Exercises/`, and the project brief together with its starter materials live under `Project-Description/`.

## Subject Overview

ELEN90088 covers the mathematical foundations and practical tools of modern optimisation and machine learning, including:

- Convex analysis and convex optimisation
- Unconstrained and constrained optimisation, duality, and KKT conditions
- Gradient-based methods (GD, SGD, accelerated and adaptive variants)
- Linear models, classification, and regularisation
- Neural networks and deep learning
- Disciplined convex programming with CVXPY and modern ML frameworks

## Repository Structure

```
.
├── Exercises/
│   ├── SOML2026_Exercise_1.ipynb       # Exercise 1 — my solution
│   ├── SOML2026_Exercise_1_Sol.ipynb   # Official solution (reference)
│   ├── SOML2026_Exercise_2.ipynb       # Exercise 2 — my solution
│   ├── Additional_Notes_Ex_1.md        # Exercise 1 notes
│   ├── Additional_Notes_Ex_2.md        # Exercise 2 notes
│   └── How It Works - DNN/             # Supporting materials for the DNN question
├── Project-Description/
│   ├── SOML_Project_26S1.md            # Overall project brief
│   ├── Project_1_Learn_to_Optimise.md  # Option 1 — learn-to-optimise
│   ├── Project_2_Hands-on_LLMs.md      # Option 2 — hands-on LLMs
│   ├── SOML_LLM_project.ipynb          # Starter notebook (Option 2)
│   ├── SOML_LLM_project_MacOS.yaml     # Conda env spec (macOS, optional)
│   └── SOML_LLM_project_Windows.yaml   # Conda env spec (Windows / CUDA, optional)
├── requirements.txt                    # pip requirements for the exercises
└── README.md
```

Progress:

- [x] Exercise 1 — Convexity, linear models, logistic regression
- [ ] Exercise 2
- [ ] Exercise 3
- [ ] Subject Project

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

I run the subject project on **Google Colab**: `SOML_LLM_project.ipynb` is uploaded there and executed on Colab's GPU/TPU, so no local environment is required for the project itself. See `Project-Description/Project_2_Hands-on_LLMs.md` for details. The two `.yaml` files in `Project-Description/` are conda env specs provided by the teaching team for students who prefer to run the project locally.

## Dependencies

Listed in [`requirements.txt`](./requirements.txt):

| Package        | Purpose                                      |
| -------------- | -------------------------------------------- |
| `numpy`        | Numerical computing, linear algebra          |
| `pandas`       | Data manipulation                            |
| `matplotlib`   | Plotting                                     |
| `seaborn`      | Statistical visualisation                    |
| `scikit-learn` | Classical ML algorithms, datasets, utilities |
| `cvxpy`        | Disciplined convex programming               |
| `torch`        | Deep learning                                |

## Notes on Use

- The notebooks are my own working solutions for study purposes. They are shared as a personal record and for discussion with peers.
- If you are currently enrolled in ELEN90088, please consult your subject's academic integrity policy before referring to any material here. Copying submissions is a breach of the University's Academic Integrity rules.
- Official solutions in this repository were provided by the teaching team; redistribution beyond personal reference is not intended.

## License

This repository is provided for personal and educational reference. No explicit license is granted; please contact me before reusing substantial portions of the material.
