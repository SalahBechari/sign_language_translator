import cv2
import numpy as np

_GREEN  = (50,  200, 50)
_RED    = (50,  50,  220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_DARK   = (20,  20,  20)
_YELLOW = (0,   220, 220)

_PANEL_H      = 110
_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_FONT_LARGE   = 1.6
_FONT_MEDIUM  = 0.75
_FONT_SMALL   = 0.55
_THICKNESS    = 2


def draw_overlay(
    frame:      np.ndarray,
    label:      str,
    confidence: float,
    sentence:   list[str],
    fps:        float,
    buffering:  bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    panel_y = h - _PANEL_H
    cv2.rectangle(frame, (0, panel_y), (w, h), _DARK, -1)
    cv2.line(frame, (0, panel_y), (w, panel_y), _GREEN, 1)

    if buffering:
        display_text = "Buffering..."
        colour       = _YELLOW
    elif label:
        display_text = label.upper()
        colour       = _GREEN
    else:
        display_text = "No sign detected"
        colour       = _RED

    (tw, th), _ = cv2.getTextSize(display_text, _FONT, _FONT_LARGE, _THICKNESS)
    tx = (w - tw) // 2
    ty = panel_y + 52
    _draw_text_with_shadow(frame, display_text, (tx, ty),
                           _FONT, _FONT_LARGE, colour, _THICKNESS)

    if label and not buffering:
        bar_w    = 220
        bar_h    = 10
        bar_x    = (w - bar_w) // 2
        bar_y    = panel_y + 62
        fill_w   = int(bar_w * confidence)
        bar_col  = _confidence_colour(confidence)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        conf_str = f"{confidence * 100:.0f}%"
        cv2.putText(frame, conf_str,
                    (bar_x + bar_w + 8, bar_y + bar_h),
                    _FONT, _FONT_SMALL, _WHITE, 1)

    sentence_str = " ".join(sentence[-8:]) if sentence else "—"
    cv2.putText(frame, f"Sentence: {sentence_str}",
                (10, panel_y + 18),
                _FONT, _FONT_SMALL, _WHITE, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}",
                (10, 26), _FONT, _FONT_SMALL, (160, 160, 160), 1)

    help_lines = ["SPACE — confirm word", "R — reset sentence", "Q — quit"]
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line,
                    (w - 210, 22 + i * 20),
                    _FONT, _FONT_SMALL, (160, 160, 160), 1)

    return frame


def _draw_text_with_shadow(
    img, text, org, font, scale, colour, thickness
) -> None:
    sx, sy = org[0] + 2, org[1] + 2
    cv2.putText(img, text, (sx, sy), font, scale, _BLACK, thickness + 1)
    cv2.putText(img, text,  org,     font, scale, colour, thickness)


def _confidence_colour(conf: float) -> tuple:
    if conf >= 0.85:
        return _GREEN
    elif conf >= 0.70:
        return _YELLOW
    return _RED


def show_frame(frame: np.ndarray, window_name: str = "Sign Language Translator") -> None:
    cv2.imshow(window_name, frame)


def destroy_windows() -> None:
    cv2.destroyAllWindows()
