import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# APPLE DISEASE CLASSIFICATION MODEL
# ============================================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4


def build_model():
    """
    Build and compile the CNN model for Apple disease classification.
    """

    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),

        # Convolutional block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolutional block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convolutional block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classification layers
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("BUILDING APPLE DISEASE CLASSIFICATION MODEL")
    print("=" * 70)

    model = build_model()

    model.summary()

    print("=" * 70)
    print("MODEL CREATED SUCCESSFULLY")
    print("=" * 70)