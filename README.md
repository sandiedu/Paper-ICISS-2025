# A Comparative Study of Regression Models for Assessing Clean Architecture in Python-based Projects

This repository contains the dataset, source code, and experimental results for the research paper titled:

**"A Comparative Study of Regression Models for Assessing Clean Architecture in Python-based Projects"**  
Submitted to **ICISS 2025 ITB Bandung** and publicly available for transparency, reproducibility, and community collaboration.

## 📌 Overview

The goal of this study is to evaluate and compare the performance of five regression models in predicting the architectural quality of Python-based software projects. The models include:

- Random Forest Regressor
- Linear Regression
- Gradient Boosting Regressor
- Support Vector Regressor
- XGBoost Regressor

Architectural quality is quantified based on static analysis features extracted from project structure (AST) and code complexity metrics (Radon).

## 🧪 Methodology Summary

1. **Data Collection**  
   - Scraped Python repositories from GitHub using the following architecture-related queries:

     ```text
     'clean architecture', 'hexagonal architecture', 'domain driven design', 'microservices architecture', 'layered architecture', 'event sourcing', 'CQRS', 'modular monolith'
     ```

   - Ranked by GitHub stars, language set to `python`, and results paginated via the GitHub API.

2. **Repository Processing**
   - Cloned repositories and extracted metadata: `repo_id`, `owner`, `name`, `description`, `stars`, `forks`, etc.
   - Extracted architecture-related features using:
     - Python's Abstract Syntax Tree (AST)
     - `radon` for code metrics (Cyclomatic Complexity, Halstead, etc.)
   - Computed an overall architecture score from combined features.

3. **Model Training & Evaluation**
   - Trained 5 regression models using scikit-learn and XGBoost.
   - Applied cross-validation and evaluated with MAE, MSE, and R² metrics.
   - Visualized model performance and architecture score distributions.

## 📊 Results

- The **XGBoost Regressor** demonstrated the highest predictive accuracy across all bins in cross-validation.
- Results suggest that ensemble methods are better suited for this type of quality prediction task.

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/sandiedu/paper-iciss-2025.git
cd paper-iciss-2025
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🏃‍♂️ Research Step

- The dataset is provided in `dataset.csv`.
- Open `scoring.ipynb` and run all cells to generate the scored dataset `dataset_scored.csv`.
- Run `training.py` to train the models. The trained models will be stored in the `models` directory.
- Open `prediction.ipynb` and run all cells to validate the models.
- Open `result.ipynb` to view the performance report.

## 🔍 Citation

If you use this work in your own research, please cite it as:

`Sandi Mulyadi. (2025). A Comparative Study of Regression Models for Assessing Clean Architecture in Python-based Projects. GitHub. https://github.com/sandiedu/paper-iciss-2025`

## 📬 Contact

For questions or collaboration, please contact:

<sandimvlyadi@gmail.com>
