# Comparative Analysis of Monotonically Enforced Univariate Binning Strategies in Classification Using Logistic Regression

This repository contains the **source code and data pipeline** for the comparative study: *"Comparative Analysis of Monotonically Enforced Univariate Binning Strategies in Classification Using Logistic Regression."*

## Project Overview

Discretization (binning) is a critical preprocessing step for logistic regression in credit scoring. This study rigorously compares ten monotonically enforced binning strategies across 1,000 bootstrap replicates (150,000 observations each) to evaluate their discriminatory power (AUROC/KS) and statistical stability.

### Key Finding
**EqualFreqChi** (a hybrid of Equal Frequency initial partitioning and ChiMerge refinement) emerged as the superior strategy, achieving a mean AUROC of **0.8850** and the highest stability (narrowest confidence intervals).

## Repository Structure

- `bootstrap.py`: The core experimental pipeline. Implements 10 binning algorithms, monotonic enforcement, and a 1,000-iteration bootstrap loop with 150,000 independent samples per replicate.
- `generate_paper_figures.py`: Generates the publication-quality figures from analysis results.
- `fig0_pipeline.png` to `fig6_Violin_auroc.png`: High-resolution figures used in the study, including pipeline architecture, forest plots, stability charts, and multi-metric scatter plots.
- `README.md`: This document.
- `.gitignore`: Configured to exclude LaTeX auxiliary files, manuscripts, and private meeting slides.

## Getting Started

### Prerequisites
- Python 3.8+
- Required Libraries: `matplotlib`, `numpy`, `scipy`, `pandas`, `scikit-learn`

### Running the Analysis
To execute the bootstrap simulation (Note: this is computationally intensive):
```bash
python bootstrap.py
```

### Generating Figures
To re-generate the figures from analysis results:
```bash
python generate_paper_figures.py
```

## Credits
- **Zakaria Sherif** (Kennesaw State University)
- **Dr. Herman Ray** (Kennesaw State University)
- **Dr. Jonathan Boardman** (Equifax)
- **Dr. Joseph White** (Equifax)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
