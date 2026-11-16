
# Reproducible Data Science Project

## 1. Project Overview

This project demonstrates a reproducible machine learning workflow using a synthetically generated binary classification dataset.

The project follows important data science practices, including problem definition, data generation, data exploration, baseline development, model training, cross-validation, testing, evaluation, and error analysis.

## 2. Research Question

**Can a Logistic Regression classification model outperform a simple majority-class baseline when predicting binary outcomes from a reproducibly generated synthetic dataset?**

## 3. Problem Description

The objective of this project is to investigate whether a machine learning classification model can perform better than a simple baseline model.

A binary classification problem is used, where the target variable contains two classes:

- Class 0
- Class 1

A Logistic Regression model is trained to predict the target class using numerical features.

A Dummy Classifier is used as the baseline to establish a simple performance benchmark.

## 4. Dataset Description

Since this project focuses on reproducibility and does not require real-world personal information, a synthetic dataset is generated using the `make_classification()` function from Scikit-learn.

### Dataset Details

| Property | Value |
|---|---|
| Number of records | 2,000 |
| Number of features | 20 |
| Target classes | 2 |
| Class 0 records | 1,267 |
| Class 1 records | 733 |
| Missing values | 0 |
| Duplicate records | 0 |
| Random state | 42 |

### Dataset Generation Parameters

- `n_samples = 2000`
- `n_features = 20`
- `n_informative = 10`
- `n_redundant = 5`
- `n_classes = 2`
- `weights = [0.65, 0.35]`
- `flip_y = 0.12`
- `class_sep = 0.6`
- `random_state = 42`

The generated dataset is saved as:

`data/raw/dataset.csv`

## 5. Technologies and Libraries

The project uses the following technologies:

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Visual Studio Code is used as the development environment.

## 6. Project Structure

```text
reproducible-ds-project/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_data.py
│   ├── explore_data.py
│   └── train.py
│
├── outputs/
│   ├── figures/
│   │   └── confusion_matrix.png
│   │
│   └── reports/
│       ├── metrics.json
│       ├── all_errors.csv
│       └── error_analysis_10.csv
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 7. Methodology

The project follows these steps:

### Step 1: Problem Definition

A research question was defined before generating or analyzing the dataset.

### Step 2: Data Generation

A synthetic binary classification dataset was generated using Scikit-learn's `make_classification()` function.

A fixed random state of 42 was used to support reproducibility.

### Step 3: Data Exploration

The generated dataset was explored using Pandas.

The following checks were performed:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Duplicate records
- Statistical summary
- Target class distribution

### Step 4: Train-Test Split

The dataset was divided into:

- Training data: 80% (1,600 records)
- Testing data: 20% (400 records)

Stratified sampling was used to maintain the class distribution in both sets.

### Step 5: Baseline Model

A Dummy Classifier with the `most_frequent` strategy was implemented.

This model always predicts the majority class.

It provides a simple benchmark to evaluate whether Logistic Regression learns useful patterns from the dataset.

### Step 6: Machine Learning Model

Logistic Regression was selected as the classification algorithm.

A Scikit-learn Pipeline was created using:

1. StandardScaler – standardizes the numerical features.
2. Logistic Regression – performs binary classification.

The maximum number of iterations was set to 1,000.

### Step 7: Cross-Validation

Five-fold Stratified Cross-Validation was performed using the training dataset.

Macro F1-score was selected as the evaluation metric.

This helps evaluate model performance across different training and validation splits.

### Step 8: Model Evaluation

The models were evaluated using the held-out test dataset.

The following evaluation metrics were calculated:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification Report
- Confusion Matrix

### Step 9: Error Analysis

Incorrect predictions were extracted and saved.

The errors were divided into:

- False Positives
- False Negatives

The first 10 incorrect predictions were selected for manual inspection.

## 8. Experimental Results

### Baseline vs Logistic Regression

The following results were obtained using the held-out test dataset.

| Evaluation Metric | Baseline | Logistic Regression |
|---|---:|---:|
| Accuracy | 63.25% | 72.75% |
| Macro Precision | 31.63% | 70.74% |
| Macro Recall | 50.00% | 68.91% |
| Macro F1-score | 38.74% | 69.48% |

### Result Interpretation

Logistic Regression outperformed the Dummy Classifier across all four evaluation metrics.

The accuracy improved from 63.25% to 72.75%.

The macro F1-score improved from 38.74% to 69.48%.

This represents an improvement of approximately 30.73 percentage points in macro F1-score.

These results suggest that Logistic Regression learned useful patterns from the generated dataset and performed better than the majority-class baseline.

## 9. Cross-Validation Results

Five-fold Stratified Cross-Validation was performed on the training dataset.

### Individual Fold Scores

- Fold 1: 0.6718
- Fold 2: 0.6680
- Fold 3: 0.6627
- Fold 4: 0.6911
- Fold 5: 0.6726

**Mean Cross-Validation Macro F1-score:** 0.6732 (67.32%)

**Standard Deviation:** 0.0096

The relatively small standard deviation indicates that the model achieved similar macro F1-scores across the five validation folds.

## 10. Classification Report

The Logistic Regression model achieved the following classification results on the test dataset:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Class 0 | 0.76 | 0.83 | 0.79 | 253 |
| Class 1 | 0.66 | 0.54 | 0.59 | 147 |
| Macro Average | 0.71 | 0.69 | 0.69 | 400 |
| Weighted Average | 0.72 | 0.73 | 0.72 | 400 |

The model performed better at identifying Class 0 than Class 1.

In particular, Class 1 had a lower recall of approximately 54%.

This indicates that the model missed some records belonging to Class 1.

## 11. Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions.

| Actual / Predicted | Predicted Class 0 | Predicted Class 1 |
|---|---:|---:|
| Actual Class 0 | 211 | 42 |
| Actual Class 1 | 67 | 80 |

The confusion matrix is saved at:

`outputs/figures/confusion_matrix.png`

### Interpretation

- True Negatives (TN): 211
- False Positives (FP): 42
- False Negatives (FN): 67
- True Positives (TP): 80

The model correctly classified 211 Class 0 records and 80 Class 1 records.

## 12. Error Analysis and Interpretation

The Logistic Regression model incorrectly classified 109 out of 400 test records.

The errors were divided into two categories:

- False Positives: 42
- False Negatives: 67

A False Positive occurs when the actual class is 0 but the model predicts class 1.

A False Negative occurs when the actual class is 1 but the model predicts class 0.

The model produced more False Negatives than False Positives. This indicates that it had more difficulty identifying records belonging to Class 1.

The first 10 incorrect predictions were selected for manual inspection. These records contained different combinations of numerical feature values. However, no single feature could be identified as the direct cause of the incorrect predictions from this inspection alone.

Since the dataset contains synthetically generated features, these values do not represent real-world variables. Therefore, the analysis focuses on prediction errors rather than assigning real-world meanings to individual features.

Overall, the error analysis highlights the limitations of the Logistic Regression model and the importance of examining incorrect predictions in addition to overall evaluation metrics.

## 13. Limitations

This project has several limitations:

1. The dataset is synthetically generated and does not represent a real-world dataset.
2. The model uses Logistic Regression only, so other machine learning algorithms may achieve different results.
3. The model does not classify all test records correctly.
4. Class 1 has a lower recall than Class 0.
5. The numerical features do not have specific real-world meanings.
6. The results may differ when using another dataset or different random seed.

## 14. Reproducibility

Reproducibility is an important objective of this project.

The following practices were implemented:

- A fixed random state of 42 was used.
- Project settings and file paths were centralized in `src/config.py`.
- Dataset generation was automated.
- Model preprocessing and training were implemented using a Scikit-learn Pipeline.
- Stratified train-test splitting was used.
- Five-fold Stratified Cross-Validation was implemented.
- Evaluation results were saved as JSON.
- Incorrect predictions were saved as CSV files.
- The confusion matrix was saved as an image.
- Project dependencies were recorded in `requirements.txt`.

These practices help reproduce the same experimental workflow.

## 15. How to Run the Project

### Prerequisites

Install Python 3 and Visual Studio Code.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd reproducible-ds-project
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Generate the Dataset

```bash
python -m src.generate_data
```

### Step 6: Explore the Dataset

```bash
python -m src.explore_data
```

### Step 7: Train and Evaluate the Model

```bash
python -m src.train
```

After execution, the evaluation results will be available inside the `outputs/` directory.

## 16. Conclusion

This project successfully demonstrates a reproducible machine learning workflow for a binary classification problem.

The Logistic Regression model achieved an accuracy of 72.75% and a macro F1-score of 69.48%, outperforming the majority-class baseline.

Five-fold cross-validation produced a mean macro F1-score of 67.32%.

Error analysis identified 42 False Positives and 67 False Negatives, highlighting the model's difficulty in identifying some Class 1 records.

Overall, the experiment demonstrates the importance of establishing a baseline, applying appropriate evaluation techniques, analyzing incorrect predictions, and maintaining reproducibility in data science projects.
