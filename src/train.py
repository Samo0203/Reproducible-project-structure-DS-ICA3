
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS
)


def train_and_evaluate():

    # 1. Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Create the baseline model
    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    # 4. Create the Logistic Regression pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    # 5. Perform 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    print("\nCross-validation F1 scores:")
    print(cv_scores)

    print("Mean CV F1:", cv_scores.mean())
    print("CV F1 standard deviation:", cv_scores.std())

    # 6. Train the Logistic Regression model
    model.fit(X_train, y_train)

    # 7. Predict test data
    predictions = model.predict(X_test)

    # 8. Calculate evaluation metrics
    def calculate_metrics(actual, predicted):

        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_macro": float(
                precision_score(actual, predicted, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(actual, predicted, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(actual, predicted, average="macro", zero_division=0)
            )
        }

    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    model_metrics = calculate_metrics(
        y_test, predictions
    )

    # 9. Print model comparison
    print("\n========== MODEL COMPARISON ==========")

    print("\nBaseline metrics:")
    print(baseline_metrics)

    print("\nLogistic Regression metrics:")
    print(model_metrics)

    # 10. Print classification report
    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    # 11. Save confusion matrix
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Class 0", "Class 1"]
    )

    display.plot(cmap="Blues")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # 12. Save all incorrect predictions
    errors = X_test.copy()

    errors["actual_target"] = y_test
    errors["predicted_target"] = predictions

    errors = errors[
        errors["actual_target"] != errors["predicted_target"]
    ]

    errors.to_csv(
        REPORTS_DIR / "all_errors.csv",
        index=True
    )

    # Save first 10 incorrect predictions
    errors.head(10).to_csv(
        REPORTS_DIR / "error_analysis_10.csv",
        index=True
    )

    print("\nIncorrect predictions:", len(errors))

    # 13. Count False Positives and False Negatives
    false_positives = (
        (errors["actual_target"] == 0) &
        (errors["predicted_target"] == 1)
    ).sum()

    false_negatives = (
        (errors["actual_target"] == 1) &
        (errors["predicted_target"] == 0)
    ).sum()

    print("\n========== ERROR ANALYSIS ==========")

    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)

    print("\nFirst 10 incorrect predictions:")
    print(errors.head(10))

    # 14. Save evaluation results as JSON
    results = {
        "baseline": baseline_metrics,
        "logistic_regression": model_metrics,
        "cross_validation_f1_scores": cv_scores.tolist(),
        "mean_cv_f1": float(cv_scores.mean()),
        "std_cv_f1": float(cv_scores.std()),
        "test_samples": int(len(X_test)),
        "incorrect_predictions": int(len(errors)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives)
    }

    with open(REPORTS_DIR / "metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nTraining and evaluation completed!")
    print("Results saved in the outputs folder.")


if __name__ == "__main__":
    train_and_evaluate()
