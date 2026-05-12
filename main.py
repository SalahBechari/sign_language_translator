import sys
import os
import time
import argparse
import threading
import queue


_speech_queue: queue.Queue = queue.Queue()
_voice_available: bool     = False


def _voice_worker() -> None:
    try:
        import pyttsx3
        while True:
            text = _speech_queue.get()
            try:
                if text is None:
                    break
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"  [Voice] Error speaking: {e}")
            finally:
                _speech_queue.task_done()

    except Exception as e:
        print(f"  [Voice] Worker error: {e}")


def _init_voice() -> bool:
    global _voice_available
    try:
        import pyttsx3
        t = threading.Thread(target=_voice_worker, daemon=True)
        t.start()
        _voice_available = True
        return True
    except ImportError:
        print(
            "\n  [Voice] pyttsx3 not installed — running without audio.\n"
            "  Install it with:  pip install pyttsx3\n"
        )
        _voice_available = False
        return False


def speak(text: str) -> None:
    if _voice_available and text:
        _speech_queue.put(text)


def _stop_voice() -> None:
    if _voice_available:
        _speech_queue.put(None)


def check_environment() -> bool:
    print("\n-- Sign Language Translator -- Environment Check --\n")
    all_ok = True

    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 9
    print(f"  {'OK' if ok else 'X '}  Python {major}.{minor}  (need 3.9+)")
    all_ok = all_ok and ok

    libs = {
        "cv2":        "opencv-python",
        "mediapipe":  "mediapipe",
        "numpy":      "numpy",
        "tensorflow": "tensorflow",
        "sklearn":    "scikit-learn",
        "matplotlib": "matplotlib",
        "tqdm":       "tqdm",
        "yt_dlp":     "yt-dlp",
        "pyttsx3":    "pyttsx3",
    }
    for import_name, pip_name in libs.items():
        try:
            __import__(import_name)
            print(f"  OK  {pip_name}")
        except ImportError:
            print(f"  X   {pip_name}  <- run:  pip install {pip_name}")
            all_ok = False

    from config import DATASET_DIR, MODEL_DIR, LOG_DIR
    for folder in [DATASET_DIR, MODEL_DIR, LOG_DIR]:
        exists = os.path.isdir(folder)
        label  = os.path.relpath(folder)
        if not exists:
            os.makedirs(folder, exist_ok=True)
            print(f"  +   {label}/  (created)")
        else:
            print(f"  OK  {label}/")

    from config import MSASL_TRAIN_JSON
    if not os.path.isfile(MSASL_TRAIN_JSON):
        print(
            "\n  WARNING: MS-ASL JSON files not found in dataset/ms_asl/\n"
            "  Download from:\n"
            "  https://www.microsoft.com/en-us/research/project/ms-asl/\n"
            "  Place MSASL_train.json, MSASL_val.json, MSASL_test.json there.\n"
        )
    else:
        print("  OK  MS-ASL JSON files present")

    print(
        "\n-- " +
        ("All good! Ready for Phase 2." if all_ok else "Fix the issues above, then re-run.") +
        " --\n"
    )
    return all_ok


def run_translator() -> None:
    import cv2
    from config import (
        CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
        MODEL_PATH, SEQUENCE_LENGTH,
    )
    from predictor import SignPredictor
    from ui import draw_overlay, show_frame, destroy_windows

    if not os.path.isfile(MODEL_PATH):
        print(
            f"\nModel not found at {MODEL_PATH}\n"
            "Train the model first:  python main.py --train\n"
        )
        sys.exit(1)

    voice_on = _init_voice()
    if voice_on:
        print("  [Voice] Text-to-speech enabled.")
    else:
        print("  [Voice] Text-to-speech disabled.")

    predictor = SignPredictor()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam. Check CAMERA_INDEX in config.py")
        sys.exit(1)

    sentence:  list[str] = []
    prev_time: float     = time.time()

    print("\nSign Language Translator running.")
    print("  SPACE — confirm word (speaks it out loud)")
    print("  ENTER — speak the full sentence")
    print("  R     — reset sentence")
    print("  Q     — quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            label, confidence, frame = predictor.process(frame)
            buffering = len(predictor._frame_buffer) < SEQUENCE_LENGTH

            frame = draw_overlay(
                frame      = frame,
                label      = label,
                confidence = confidence,
                sentence   = sentence,
                fps        = fps,
                buffering  = buffering,
            )

            show_frame(frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" ") and label:
                sentence.append(label)
                print(f"  + '{label}'  ->  {' '.join(sentence)}")
                speak(label)
                predictor.reset()

            elif key == 13 and sentence:
                full = " ".join(sentence)
                print(f"  Speaking: {full}")
                speak(full)

            elif key == ord("r"):
                sentence.clear()
                predictor.reset()
                print("  Sentence reset.")

    finally:
        predictor.close()
        cap.release()
        destroy_windows()
        _stop_voice()
        print("\nTranslator closed.")
        if sentence:
            print(f"Final sentence: {' '.join(sentence)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign Language Translator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--check",    action="store_true",
                        help="Run environment check")
    parser.add_argument("--download", action="store_true",
                        help="Download MS-ASL + WLASL clips (Phase 2)")
    parser.add_argument("--extract",  action="store_true",
                        help="Extract landmarks from clips (Phase 3)")
    parser.add_argument("--train",    action="store_true",
                        help="Train the model (Phase 4)")
    parser.add_argument("--gui",      action="store_true",
                        help="Launch desktop GUI")
    args = parser.parse_args()

    if args.check:
        check_environment()
    elif args.download:
        from data_loader import load_dataset
        load_dataset(max_per_label=150)
    elif args.extract:
        from feature_extraction import build_features
        build_features()
    elif args.train:
        from model_training import train
        train()
    elif args.gui:
        from app_gui import run_gui
        run_gui()
    else:
        run_translator()


if __name__ == "__main__":
    main()