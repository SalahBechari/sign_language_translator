import os
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH, SEQUENCE_LENGTH
from predictor import SignPredictor


class TranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("1100x760")
        self.root.minsize(980, 700)
        self.root.configure(bg="#0b1220")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train first with: python main.py --train"
            )

        self.predictor = SignPredictor()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check CAMERA_INDEX in config.py")

        self.sentence: list[str] = []
        self.current_label = ""
        self.current_confidence = 0.0
        self.last_valid_label = ""
        self.last_valid_ts = 0.0
        self.prev_time = time.time()
        self.running = True

        self._apply_styles()
        self._build_layout()
        self._update_status("Ready", "#22c55e")
        self._update_frame()

    def _apply_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0b1220")
        style.configure("Card.TFrame", background="#111827")
        style.configure("Header.TLabel", background="#0b1220", foreground="#f8fafc", font=("Segoe UI", 20, "bold"))
        style.configure("SubHeader.TLabel", background="#0b1220", foreground="#94a3b8", font=("Segoe UI", 10))
        style.configure("Info.TLabel", background="#111827", foreground="#e2e8f0", font=("Segoe UI", 11))
        style.configure("Accent.TLabel", background="#111827", foreground="#22d3ee", font=("Segoe UI", 14, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=8)
        style.map(
            "Primary.TButton",
            background=[("active", "#0ea5e9"), ("!disabled", "#0284c7")],
            foreground=[("!disabled", "white")],
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#334155"), ("!disabled", "#1e293b")],
            foreground=[("!disabled", "#e2e8f0")],
        )

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root_frame, text="Sign Language Translator", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            root_frame,
            text="Real-time recognition with a clean desktop interface",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        content = ttk.Frame(root_frame, style="Root.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        video_card = ttk.Frame(content, style="Card.TFrame", padding=10)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_label = tk.Label(video_card, bg="#000000", bd=0, highlightthickness=0)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        side_card = ttk.Frame(content, style="Card.TFrame", padding=14)
        side_card.grid(row=0, column=1, sticky="nsew")

        self.word_var = tk.StringVar(value="Word: -")
        self.conf_var = tk.StringVar(value="Confidence: 0%")
        self.sentence_var = tk.StringVar(value="Sentence: -")
        self.fps_var = tk.StringVar(value="FPS: 0")
        self.status_var = tk.StringVar(value="Status: Ready")

        ttk.Label(side_card, text="Live Output", style="Accent.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(side_card, textvariable=self.word_var, style="Info.TLabel").pack(anchor="w", pady=3)
        ttk.Label(side_card, textvariable=self.conf_var, style="Info.TLabel").pack(anchor="w", pady=3)
        ttk.Label(side_card, textvariable=self.sentence_var, style="Info.TLabel", wraplength=360).pack(anchor="w", pady=3)
        ttk.Label(side_card, textvariable=self.fps_var, style="Info.TLabel").pack(anchor="w", pady=3)
        self.status_label = ttk.Label(side_card, textvariable=self.status_var, style="Info.TLabel")
        self.status_label.pack(anchor="w", pady=(8, 14))

        buttons = ttk.Frame(side_card, style="Card.TFrame")
        buttons.pack(fill=tk.X, pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)

        ttk.Button(
            buttons, text="Confirm Word", command=self._confirm_word, style="Primary.TButton"
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(
            buttons, text="Speak Sentence", command=self._speak_sentence, style="Primary.TButton"
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)
        ttk.Button(
            buttons, text="Reset", command=self._reset_sentence, style="Secondary.TButton"
        ).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(
            buttons, text="Quit", command=self._on_close, style="Secondary.TButton"
        ).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=4)

    def _update_status(self, text: str, color: str) -> None:
        self.status_var.set(f"Status: {text}")
        self.status_label.configure(foreground=color)

    def _update_frame(self) -> None:
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        label, confidence, frame = self.predictor.process(frame)
        self.current_label = label
        self.current_confidence = confidence
        if label:
            self.last_valid_label = label
            self.last_valid_ts = time.time()

        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        buffering = len(self.predictor._frame_buffer) < SEQUENCE_LENGTH
        shown_label = "Buffering..." if buffering else (label if label else "No sign detected")
        status_text = "Buffering camera sequence" if buffering else ("Prediction ready" if label else "Waiting for clear sign")
        status_color = "#f59e0b" if buffering else ("#22c55e" if label else "#94a3b8")

        self.word_var.set(f"Word: {shown_label}")
        self.conf_var.set(f"Confidence: {confidence * 100:.0f}%")
        self.sentence_var.set(f"Sentence: {' '.join(self.sentence) if self.sentence else '-'}")
        self.fps_var.set(f"FPS: {fps:.0f}")
        self._update_status(status_text, status_color)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((640, 480))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

        self.root.after(15, self._update_frame)

    def _confirm_word(self) -> None:
        now = time.time()
        label_to_confirm = self.current_label
        if not label_to_confirm and self.last_valid_label and (now - self.last_valid_ts) <= 1.5:
            label_to_confirm = self.last_valid_label

        if label_to_confirm:
            self.sentence.append(label_to_confirm)
            self.predictor.reset()
            self.current_label = ""
            self._update_status("Word confirmed", "#22c55e")
        else:
            self._update_status("No word to confirm", "#ef4444")

    def _speak_sentence(self) -> None:
        if not self.sentence:
            self._update_status("Sentence is empty", "#ef4444")
            return
        text = " ".join(self.sentence)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            self._update_status("Sentence spoken", "#22c55e")
        except Exception as e:
            print(f"[Voice] Could not speak sentence: {e}")
            self._update_status("Voice error", "#ef4444")

    def _reset_sentence(self) -> None:
        self.sentence.clear()
        self.predictor.reset()
        self.current_label = ""
        self.last_valid_label = ""
        self.last_valid_ts = 0.0
        self.sentence_var.set("Sentence: -")
        self._update_status("Sentence reset", "#94a3b8")

    def _on_close(self) -> None:
        self.running = False
        try:
            self.predictor.close()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()

