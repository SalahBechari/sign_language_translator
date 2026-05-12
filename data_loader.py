

import os
import json
import subprocess
from tqdm import tqdm

from config import (
    DATASET_DIR, BASE_DIR, LABELS,
    MSASL_TRAIN_JSON, MSASL_VAL_JSON, MSASL_TEST_JSON,
)
from utils import get_logger, ensure_dir, load_json

log = get_logger("data_loader")

WLASL_JSON = os.path.join(BASE_DIR, "dataset", "WLASL_v0.3.json")


def _label_dirs() -> None:
    for label in LABELS:
        ensure_dir(os.path.join(DATASET_DIR, label))


def _clip_path(label: str, clip_id: str) -> str:
    return os.path.join(DATASET_DIR, label, f"{clip_id}.mp4")


def _already_downloaded(label: str, clip_id: str) -> bool:
    path = _clip_path(label, clip_id)
    return os.path.isfile(path) and os.path.getsize(path) > 1024


def _download_clip(url: str, label: str, clip_id: str,
                   start: float = None, end: float = None) -> bool:
    if _already_downloaded(label, clip_id):
        return True

    out_path = _clip_path(label, clip_id)
    tmp_path = out_path.replace(".mp4", "_full.mp4")

    dl_cmd = [
        "yt-dlp",
        "--quiet",
        "--format", "worst[ext=mp4]/worst",
        "--output", tmp_path,
        url,
    ]
    result = subprocess.run(dl_cmd, capture_output=True)
    if result.returncode != 0 or not os.path.isfile(tmp_path):
        return False

    if start is not None and end is not None:
        duration = max(0.1, end - start)
        trim_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", tmp_path,
            "-t", str(duration),
            "-c", "copy",
            out_path,
        ]
        result = subprocess.run(trim_cmd, capture_output=True)
        os.remove(tmp_path)
        if result.returncode != 0:
            return False
    else:

        os.rename(tmp_path, out_path)

    return True


def _load_msasl(max_per_label: int) -> dict[str, list]:
    split_files = {
        "train": MSASL_TRAIN_JSON,
        "val":   MSASL_VAL_JSON,
        "test":  MSASL_TEST_JSON,
    }

    by_label: dict[str, list] = {lbl: [] for lbl in LABELS}

    for split_name, path in split_files.items():
        if not os.path.isfile(path):
            log.warning(f"MS-ASL: {split_name} file not found — skipping")
            continue
        data = load_json(path)
        for entry in data:
            lbl = entry.get("text", "").lower()
            if lbl in by_label and len(by_label[lbl]) < max_per_label:
                by_label[lbl].append(entry)
        log.info(f"MS-ASL {split_name}: loaded entries")

    return by_label


def _download_msasl(by_label: dict, max_per_label: int) -> tuple[int, int]:
    success = failed = 0
    for label, entries in by_label.items():
        if not entries:
            continue
        log.info(f"  MS-ASL '{label}' ({len(entries)} clips)...")
        for entry in tqdm(entries, desc=f"msasl/{label}", leave=False):
            url     = entry.get("url", "")
            start   = float(entry.get("start_time", 0))
            end     = float(entry.get("end_time", 0))
            signer  = entry.get("signer_id", 0)
            clip_id = f"msasl_{label}_{signer}_{int(start*100)}"
            if _download_clip(url, label, clip_id, start, end):
                success += 1
            else:
                failed += 1
    return success, failed


def _load_wlasl(max_per_label: int) -> dict[str, list]:
    if not os.path.isfile(WLASL_JSON):
        log.warning(
            f"WLASL JSON not found at {WLASL_JSON}\n"
            "  Download WLASL_v0.3.json from https://dxli94.github.io/WLASL/\n"
            "  and place it in dataset/"
        )
        return {lbl: [] for lbl in LABELS}

    data     = load_json(WLASL_JSON)
    by_label: dict[str, list] = {lbl: [] for lbl in LABELS}

    for entry in data:
        gloss = entry.get("gloss", "").lower()
        if gloss not in by_label:
            continue
        for instance in entry.get("instances", []):
            if len(by_label[gloss]) >= max_per_label:
                break
            url = instance.get("url", "")
            if not url:
                continue
            by_label[gloss].append({
                "url":      url,
                "video_id": instance.get("video_id", "unknown"),
                "label":    gloss,
            })

    total = sum(len(v) for v in by_label.values())
    log.info(f"WLASL: found {total} entries for {len(LABELS)} labels")
    return by_label


def _download_wlasl(by_label: dict) -> tuple[int, int]:
    success = failed = 0
    for label, entries in by_label.items():
        if not entries:
            continue
        log.info(f"  WLASL '{label}' ({len(entries)} clips)...")
        for entry in tqdm(entries, desc=f"wlasl/{label}", leave=False):
            url      = entry["url"]
            video_id = entry["video_id"]
            clip_id  = f"wlasl_{label}_{video_id}"
            if _download_clip(url, label, clip_id):
                success += 1
            else:
                failed += 1
    return success, failed


def load_dataset(max_per_label: int = 150) -> None:
    _label_dirs()

    total_success = 0
    total_failed  = 0

    log.info("Loading MS-ASL entries...")
    msasl_by_label = _load_msasl(max_per_label)
    msasl_total    = sum(len(v) for v in msasl_by_label.values())

    if msasl_total > 0:
        log.info(f"Downloading {msasl_total} MS-ASL clips...")
        s, f = _download_msasl(msasl_by_label, max_per_label)
        total_success += s
        total_failed  += f
    else:
        log.warning("No MS-ASL clips to download.")

    log.info("Loading WLASL entries...")
    wlasl_by_label = _load_wlasl(max_per_label)
    wlasl_total    = sum(len(v) for v in wlasl_by_label.values())

    if wlasl_total > 0:
        log.info(f"Downloading {wlasl_total} WLASL clips...")
        s, f = _download_wlasl(wlasl_by_label)
        total_success += s
        total_failed  += f
    else:
        log.warning(
            "No WLASL clips found. Make sure WLASL_v0.3.json is in dataset/"
        )

    log.info(f"\nDone. Success: {total_success}  |  Failed: {total_failed}")
    _print_summary()


def _print_summary() -> None:
    print("\nDataset summary (all sources combined):")
    total = 0
    for label in LABELS:
        folder = os.path.join(DATASET_DIR, label)
        count  = len([f for f in os.listdir(folder) if f.endswith(".mp4")])
        bar    = "X" * (count // 5)
        print(f"  {label:>14}  {count:4d}  {bar}")
        total += count
    print(f"\n  Total clips: {total}")
    print()


if __name__ == "__main__":
    load_dataset(max_per_label=150)