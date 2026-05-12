
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "ms_asl")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

MODEL_PATH       = os.path.join(MODEL_DIR, "sign_model.h5")
LABEL_MAP_PATH   = os.path.join(MODEL_DIR, "label_map.json")
FEATURES_PATH    = os.path.join(DATASET_DIR, "features.npz")

MSASL_TRAIN_JSON = os.path.join(DATASET_DIR, "MSASL_train.json")
MSASL_VAL_JSON   = os.path.join(DATASET_DIR, "MSASL_val.json")
MSASL_TEST_JSON  = os.path.join(DATASET_DIR, "MSASL_test.json")

LABELS = [
    "hello",      "please",     "sorry",      "yes",        "no",
    "help",       "water",      "more",       "go",         "good",
    "bad",        "want",       "like",       "love",       "know",
    "understand", "name",       "where",      "what",       "how",
    "again",
]
NUM_CLASSES = len(LABELS)

NUM_LANDMARKS      = 21
NUM_COORDS         = 3
MAX_HANDS          = 2


FEATURES_PER_FRAME = NUM_LANDMARKS * NUM_COORDS * MAX_HANDS  


SEQUENCE_LENGTH    = 30

EPOCHS           = 60
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.15

CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_BUFFER     = 15


CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480