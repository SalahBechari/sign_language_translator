import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

from config import VALIDATION_SPLIT, SEQUENCE_LENGTH
from feature_extraction import load_features
from utils import get_logger, print_class_distribution

log = get_logger("preprocessing")


def _add_noise(X: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
    return X + noise


def _time_shift(X: np.ndarray, max_shift: int = 3) -> np.ndarray:
    out = np.zeros_like(X)
    for i in range(len(X)):
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            out[i, shift:] = X[i, :-shift]
        elif shift < 0:
            out[i, :shift] = X[i, -shift:]
        else:
            out[i] = X[i]
    return out


def augment(X: np.ndarray, y: np.ndarray,
            copies: int = 1) -> tuple[np.ndarray, np.ndarray]:
    X_aug_parts = [X]
    y_aug_parts = [y]

    for _ in range(copies):
        Xa = _add_noise(_time_shift(X.copy()))
        X_aug_parts.append(Xa)
        y_aug_parts.append(y)

    X_out = np.concatenate(X_aug_parts, axis=0)
    y_out = np.concatenate(y_aug_parts, axis=0)
    return sk_shuffle(X_out, y_out, random_state=42)


def get_splits(augment_train: bool = True):
    X, y = load_features()
    log.info(f"Loaded features: X={X.shape}, y={y.shape}")
    print_class_distribution(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        stratify=y,
        random_state=42,
    )

    val_ratio = VALIDATION_SPLIT / (1 - VALIDATION_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=42,
    )

    if augment_train:
        log.info("Augmenting training set (1 extra copy)...")
        X_train, y_train = augment(X_train, y_train, copies=1)

    log.info(
        f"Split sizes — "
        f"train: {len(X_train)}, "
        f"val: {len(X_val)}, "
        f"test: {len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")
