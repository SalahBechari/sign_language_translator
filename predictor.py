import os
import collections
import urllib.request
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

from tensorflow import keras

from config import (
    MODEL_PATH, LABEL_MAP_PATH,
    SEQUENCE_LENGTH, FEATURES_PER_FRAME,
    CONFIDENCE_THRESHOLD, SMOOTHING_BUFFER,
    NUM_LANDMARKS, NUM_COORDS, MAX_HANDS,
)
from utils import get_logger, load_label_map, normalize_landmarks

log = get_logger("predictor")

MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_LANDMARK_COLOR   = (0, 220, 100)
_CONNECTION_COLOR = (255, 255, 255)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _ensure_model() -> None:
    if not os.path.isfile(MODEL_FILE):
        log.info("Downloading MediaPipe hand landmarker model (~25 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        log.info("Model downloaded.")


def _draw_hand(frame: np.ndarray, landmarks: list) -> None:
    h, w   = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, points[a], points[b], _CONNECTION_COLOR, 1)
    for pt in points:
        cv2.circle(frame, pt, 4, _LANDMARK_COLOR, -1)


class SignPredictor:
    def __init__(self) -> None:
        _ensure_model()
        log.info("Loading sign language model...")
        self.model     = keras.models.load_model(MODEL_PATH)
        self.label_map = load_label_map()
        log.info(f"Model ready. {len(self.label_map)} classes.")

        self._frame_buffer: collections.deque = collections.deque(maxlen=SEQUENCE_LENGTH)
        self._pred_buffer:  collections.deque = collections.deque(maxlen=SMOOTHING_BUFFER)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_FILE),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector     = HandLandmarker.create_from_options(options)
        self.current_label = ""
        self.current_conf  = 0.0

    def process(self, frame: np.ndarray) -> tuple[str, float, np.ndarray]:
        annotated, landmarks = self._extract_landmarks(frame)
        self._frame_buffer.append(landmarks)

        if len(self._frame_buffer) < SEQUENCE_LENGTH:
            return "", 0.0, annotated

        label, conf = self._predict_sequence()
        return label, conf, annotated

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._pred_buffer.clear()
        self.current_label = ""
        self.current_conf  = 0.0

    def close(self) -> None:
        self._detector.close()

    def _extract_landmarks(self, frame: np.ndarray) -> tuple:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect(mp_image)

        combined = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

        for i, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
            _draw_hand(frame, hand)
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand],
                dtype=np.float32,
            ).flatten()
            normalised = normalize_landmarks(raw)
            slot_start = i * NUM_LANDMARKS * NUM_COORDS
            slot_end   = slot_start + NUM_LANDMARKS * NUM_COORDS
            combined[slot_start:slot_end] = normalised

        return frame, combined

    def _predict_sequence(self) -> tuple[str, float]:
        seq   = np.array(list(self._frame_buffer), dtype=np.float32)[np.newaxis, ...]
        probs = self.model.predict(seq, verbose=0)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])

        self._pred_buffer.append(idx)
        smooth_idx = collections.Counter(self._pred_buffer).most_common(1)[0][0]
        label      = self.label_map.get(smooth_idx, "")

        if conf < CONFIDENCE_THRESHOLD:
            return "", conf

        self.current_label = label
        self.current_conf  = conf
        return label, conf