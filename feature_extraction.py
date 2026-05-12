import os
import cv2
import urllib.request
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import (
    DATASET_DIR, LABELS, FEATURES_PATH,
    SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_COORDS, MAX_HANDS,
    FEATURES_PER_FRAME,
)
from utils import get_logger, normalize_landmarks, label_to_index

log = get_logger("feature_extraction")

MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model() -> None:
    if not os.path.isfile(MODEL_FILE):
        log.info("Downloading MediaPipe hand landmarker model (~25 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        log.info("Model downloaded.")


def _make_detector() -> HandLandmarker:
    _ensure_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def sample_frames(video_path: str, n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def extract_landmarks_from_frame(frame: np.ndarray, detector: HandLandmarker) -> np.ndarray:
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    combined = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    for i, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32,
        ).flatten()
        normalised = normalize_landmarks(raw)
        combined[i * NUM_LANDMARKS * NUM_COORDS : (i + 1) * NUM_LANDMARKS * NUM_COORDS] = normalised

    return combined


def extract_sequence(video_path: str, detector: HandLandmarker):
    frames = sample_frames(video_path, SEQUENCE_LENGTH)
    if not frames:
        return None

    seq      = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
    detected = 0

    for i, frame in enumerate(frames):
        landmarks = extract_landmarks_from_frame(frame, detector)
        if np.any(landmarks != 0):
            seq[i]   = landmarks
            detected += 1

    if detected < SEQUENCE_LENGTH // 2:
        return None

    return seq


def build_features() -> None:
    detector = _make_detector()
    X_list, y_list = [], []
    skipped = 0

    for label in LABELS:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            log.warning(f"No folder for label '{label}' — skipping")
            continue
        clips = [f for f in os.listdir(label_dir) if f.endswith(".mp4")]
        if not clips:
            log.warning(f"No clips found for '{label}'")
            continue
        log.info(f"Processing '{label}' ({len(clips)} clips)...")
        class_idx = label_to_index(label)
        for clip_name in tqdm(clips, desc=label, leave=False):
            seq = extract_sequence(os.path.join(label_dir, clip_name), detector)
            if seq is None:
                skipped += 1
                continue
            X_list.append(seq)
            y_list.append(class_idx)

    detector.close()

    if not X_list:
        log.error("No sequences extracted.")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    np.savez_compressed(FEATURES_PATH, X=X, y=y)
    log.info(f"Features saved: X={X.shape}, y={y.shape}, Skipped={skipped}")


def load_features():
    if not os.path.isfile(FEATURES_PATH):
        raise FileNotFoundError(
            f"Features not found at {FEATURES_PATH}. Run: python main.py --extract"
        )
    data = np.load(FEATURES_PATH)
    return data["X"], data["y"]


if __name__ == "__main__":
    build_features()