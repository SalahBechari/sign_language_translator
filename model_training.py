import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from config import (
    MODEL_DIR, MODEL_PATH, LOG_DIR,
    NUM_CLASSES, SEQUENCE_LENGTH, FEATURES_PER_FRAME,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, LABELS,
)
from preprocessing import get_splits
from utils import get_logger, save_label_map, plot_training_history

log = get_logger("model_training")


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME),
                         name="landmarks")

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name="bilstm_1"
    )(inputs)
    x = layers.Dropout(0.4)(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False), name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs, outputs, name="SignLanguageClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(augment_train=True)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
        ),
    ]

    unique, counts = np.unique(y_train, return_counts=True)
    total_samples  = len(y_train)
    class_weight   = {
        int(cls): total_samples / (NUM_CLASSES * cnt)
        for cls, cnt in zip(unique, counts)
    }

    log.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    log.info("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log.info(f"Test accuracy : {test_acc:.4f}")
    log.info(f"Test loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    _plot_confusion_matrix(y_test, y_pred)

    save_label_map()
    log.info(f"Model saved → {MODEL_PATH}")

    plot_path = os.path.join(LOG_DIR, "training_history.png")
    plot_training_history(history, save_path=plot_path)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im   = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    log.info(f"Confusion matrix saved → {save_path}")


if __name__ == "__main__":
    train()
