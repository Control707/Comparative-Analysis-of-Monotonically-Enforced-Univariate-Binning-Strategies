# Comparative Analysis of Monotonically Enforced Univariate Binning Strategies in Classification Using Logistic Regression

This repository contains the source code, data pipeline, and LaTeX manuscript for the research paper: **"Comparative Analysis of Monotonically Enforced Univariate Binning Strategies in Classification Using Logistic Regression."**

## Project Overview

Discretization (binning) is a critical preprocessing step for logistic regression in credit scoring. This study rigorously compares ten monotonically enforced binning strategies across 1,000 bootstrap replicates to evaluate their discriminatory power (AUROC/KS) and statistical stability.

### Key Finding
**EqualFreqChi** (a hybrid of Equal Frequency initial partitioning and ChiMerge refinement) emerged as the superior strategy, achieving a mean AUROC of **0.8850** and the highest stability (narrowest confidence intervals).

## Repository Structure

- `bootstrap.py`: The core experimental pipeline. Implements 10 binning algorithms, monotonic enforcement, and a 1,000-iteration bootstrap loop with 150,000 independent samples per replicate.
- `generate_paper_figures.py`: Generates the publication-quality figures used in the manuscript.
- `paper.tex`: LaTeX source for the research paper.
- `references.bib`: BibTeX bibliography.
- `paper_figures/`: Directory containing the resulting plots (Pipeline flowchart, Forest plots, Stability charts, Scatter plots).
- `README.md`: This file.

## Getting Started

### Prerequisites
- Python 3.8+
- Matplotlib, NumPy, SciPy, Pandas, Scikit-Learn
- TeX Live (for PDF compilation)

### Running the Analysis
To execute the bootstrap simulation (Note: this is computationally intensive):
```bash
python bootstrap.py
```

### Generating Figures
To re-generate the figures from the results:
```bash
python generate_paper_figures.py
```

### Compiling the Paper
To build the PDF manuscript:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Authors
- **Zakaria Sherif** (Kennesaw State University)
- **Dr. Herman Ray** (Kennesaw State University)
- **Dr. Jonathan Boardman** (Equifax)
- **Dr. Joseph White** (Equifax)

## License
[Insert License Here - e.g., MIT]
