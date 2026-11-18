"""
Batch analysis for FakeWave: score every audio file in a folder with
timeline.analyze_timeline and produce a table plus CSV/JSON files.

Read-only with respect to the scanned folder: nothing is written into, renamed
within, or deleted from it. Output goes under reports/ (gitignored).

Forward passes only -- no Grad-CAM -- and one file at a time: each file's
conditioned waveform is discarded as soon as its summary row is recorded.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import report
import timeline
from preprocess import CONFIG

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".aac"}

_NUMERIC_FIELDS = ["n_windows", "analyzed_duration_sec", "mean_fake_prob", "max_fake_prob", "n_suspect"]
CSV_COLUMNS = ["filename", "n_windows", "analyzed_duration_sec", "mean_fake_prob",
               "max_fake_prob", "n_suspect", "headline", "error"]


def find_audio_files(folder, recursive=False):
    """Sorted audio files in `folder` (case-insensitive extension match)."""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    it = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _sort_key(row):
    # errors last; otherwise most suspicious first. Never compares None.
    if row["error"]:
        return (1, 0.0, row["filename"])
    return (0, -row["mean_fake_prob"], row["filename"])


def analyze_folder(folder, model, recursive=False, expected_label=None, config=CONFIG,
                   progress_cb=None, model_path="detector.pt"):
    """Score every audio file in `folder`. One bad file never aborts the run."""
    if expected_label not in (None, "real", "fake"):
        raise ValueError(f"expected_label must be None, 'real' or 'fake', got {expected_label!r}")
    files = find_audio_files(folder, recursive)
    root = Path(folder)
    started = datetime.now(timezone.utc)

    rows = []
    for i, path in enumerate(files):
        # relative name keeps recursive results distinguishable (a/x.wav vs b/x.wav)
        name = path.relative_to(root).as_posix()
        try:
            r = timeline.analyze_timeline(str(path), model, config)
            rows.append({
                "filename": name, "n_windows": r["n_windows"],
                "analyzed_duration_sec": r["analyzed_duration_sec"],
                "mean_fake_prob": r["mean_fake_prob"], "max_fake_prob": r["max_fake_prob"],
                "n_suspect": r["n_suspect"], "headline": r["headline"], "error": "",
            })
            del r  # drop the conditioned waveform before the next file
        except Exception as exc:
            row = {"filename": name, "headline": None, "error": f"{type(exc).__name__}: {exc}"}
            row.update({k: None for k in _NUMERIC_FIELDS})
            rows.append(row)
        if progress_cb is not None:
            progress_cb(i + 1, len(files), name)

    rows.sort(key=_sort_key)
    ok = [r for r in rows if not r["error"]]
    accuracy = n_correct = n_inconclusive = None
    if expected_label is not None:
        want = "Likely synthetic" if expected_label == "fake" else "Likely authentic"
        n_correct = sum(r["headline"] == want for r in ok)
        n_inconclusive = sum(r["headline"] == "Inconclusive" for r in ok)
        accuracy = (n_correct / len(ok)) if ok else None  # denominator is n_ok

    return {
        "folder": str(folder), "recursive": recursive, "expected_label": expected_label,
        "model_file": Path(model_path).name, "model_sha256": report.sha256_file(model_path),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "rows": rows,
        "accuracy": accuracy, "n_correct": n_correct, "n_inconclusive": n_inconclusive,
        "thresholds": {"suspect": timeline.SUSPECT_THRESHOLD, "clear": timeline.CLEAR_THRESHOLD},
        "pipeline": report.pipeline_params(config),
    }


def write_batch_csv(result, out_dir="reports"):
    """Write rows as CSV (None -> empty cell) plus a JSON sidecar of everything else."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(result["finished_utc"]).strftime("%Y%m%dT%H%M%SZ")
    csv_path, json_path = out / f"fakewave_batch_{stamp}.csv", out / f"fakewave_batch_{stamp}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in result["rows"]:
            w.writerow(["" if r[c] is None else r[c] for c in CSV_COLUMNS])
    meta = {k: v for k, v in result.items() if k != "rows"}
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(csv_path), str(json_path)
