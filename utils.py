import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from config import LOG_DIR, LABELS, LABEL_MAP_PATH


def get_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def label_to_index(label: str) -> int:
    return LABELS.index(label)

def index_to_label(idx: int) -> str:
    return LABELS[int(idx)]

def save_label_map() -> None:
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    mapping = {str(i): lbl for i, lbl in enumerate(LABELS)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

def load_label_map() -> dict:
    with open(LABEL_MAP_PATH, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = landmarks.reshape(21, 3).copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def print_class_distribution(labels_array: np.ndarray) -> None:
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\nClass distribution:")
    for idx, count in zip(unique, counts):
        bar = "█" * (count // 5)
        print(f"  {index_to_label(idx):>14}  {count:4d}  {bar}")
    print()


def plot_training_history(history, save_path: str = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training plot saved → {save_path}")
    else:
        plt.show()
