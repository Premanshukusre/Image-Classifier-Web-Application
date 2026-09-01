from pathlib import Path
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 20


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():
    print("=" * 70)
    print("APPLE DISEASE CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_ds, val_ds, test_ds = load_datasets()

    print("\nDatasets loaded successfully.")
    print(f"Training batches   : {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation batches : {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches       : {tf.data.experimental.cardinality(test_ds).numpy()}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # --------------------------------------------------------
    # Evaluate on test dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST DATA")
    print("=" * 70)

    test_loss, test_accuracy = model.evaluate(test_ds)

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------
    
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = "models/apple_disease_model.keras"

    model.save(model_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nModel saved to: {model_path}")

    return model, history


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()