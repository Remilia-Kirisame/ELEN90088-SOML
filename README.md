# ELEN90088 — System Optimisation and Machine Learning

Coursework repository for [ELEN90088 System Optimisation and Machine Learning](https://handbook.unimelb.edu.au/subjects/elen90088) at the University of Melbourne (2026).

This repository collects my solutions, supporting notes, and the final project for the subject. Content is organised by assessment: weekly exercises live under `Exercises/`, and the capstone project will be added in its own directory once released.

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
│   └── Additional_Notes_Ex_1.md        # Extra derivations and Python notes
├── requirements.txt
└── README.md
```

Planned additions:

- [x] Exercise 1 — Convexity, linear models, logistic regression
- [ ] Exercise 2
- [ ] Exercise 3
- [ ] Subject Project

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` or `conda`
- JupyterLab / Jupyter Notebook (or VS Code with the Jupyter extension)

### Setup

Clone the repository and install dependencies into a virtual environment:

```bash
git clone https://github.com/<your-username>/ELEN90088-SOML.git
cd ELEN90088-SOML

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter lab
```

Then open any notebook under `Exercises/` and run the cells top-to-bottom.

## Dependencies

Listed in [`requirements.txt`](./requirements.txt):

| Package        | Purpose                                             |
| -------------- | --------------------------------------------------- |
| `numpy`        | Numerical computing, linear algebra                 |
| `pandas`       | Data manipulation                                   |
| `matplotlib`   | Plotting                                            |
| `seaborn`      | Statistical visualisation                           |
| `scikit-learn` | Classical ML algorithms, datasets, utilities        |
| `cvxpy`        | Disciplined convex programming                      |
| `torch`        | Deep learning                                       |

## Notes on Use

- The notebooks are my own working solutions for study purposes. They are shared as a personal record and for discussion with peers.
- If you are currently enrolled in ELEN90088, please consult your subject's academic integrity policy before referring to any material here. Copying submissions is a breach of the University's Academic Integrity rules.
- Official solutions in this repository were provided by the teaching team; redistribution beyond personal reference is not intended.

## License

This repository is provided for personal and educational reference. No explicit license is granted; please contact me before reusing substantial portions of the material.
