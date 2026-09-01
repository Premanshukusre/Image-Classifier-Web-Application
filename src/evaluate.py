from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_datasets, CLASS_NAMES


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/apple_disease_model.keras")
RESULTS_DIR = Path("results")

CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model():
    print("=" * 70)
    print("APPLE DISEASE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print(f"\nLoading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_ds = load_datasets()

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("APPLE DISEASE CLASSIFICATION REPORT\n")
        file.write("=" * 70 + "\n\n")
        file.write(f"Test Accuracy: {accuracy:.4f}\n")
        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title("Apple Disease Classification - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    # Display values inside matrix
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nResults saved:")
    print(f"  Classification report: {REPORT_PATH}")
    print(f"  Confusion matrix      : {CONFUSION_MATRIX_PATH}")

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate_model()