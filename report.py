"""
Case report export for FakeWave: turns ONE cached analysis into a durable,
standalone evidence artifact (a self-contained HTML file plus a JSON sidecar).

Everything here is built from the analysis result the user was looking at (held
in gr.State) -- nothing re-runs analyze_timeline, so the report cannot disagree
with the screen. Images are embedded as base64 data URIs and the HTML has no
external CSS, fonts or scripts, so it renders offline.

Output goes under reports/, which is gitignored: reports carry evidence
filenames, hashes and examiner notes and must never be committed.
"""

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import timeline
from preprocess import CONFIG

TOOL_NAME = "FakeWave"
TOOL_VERSION = "0.2.0"

LIMITATIONS = (
    "This report records the output of an automated classifier. It estimates whether "
    "the analyzed audio contains synthetic (AI-generated) speech. It does not identify "
    "the speaker, does not determine which synthesis system produced any segment, and "
    "does not establish the origin, location, or transmission path of the recording. "
    "Reported values are uncalibrated model outputs, not statistical likelihoods. "
    'Segments marked "inconclusive" fall between the configured thresholds and should '
    "be read as neither finding. This output is investigative support and requires "
    "corroboration by independent evidence."
)

_PIPELINE_KEYS = [
    "target_sr", "target_duration_sec", "target_rms_dbfs", "n_fft",
    "hop_length", "n_mels", "mp3_bitrate_kbps", "silence_top_db",
]


def pipeline_params(config=CONFIG):
    """The preprocessing parameters that pin exactly how a result was produced."""
    return {k: config[k] for k in _PIPELINE_KEYS}


def sha256_file(path, chunk_size=1 << 20):
    """Hex SHA-256 of a file, streamed so large files are never fully in memory."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def png_data_uri(rgb_array):
    """(H, W, 3) uint8 array -> 'data:image/png;base64,...'."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb_array, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_report_data(result, metadata, source_path, model_path,
                      case_ref="", examiner="", notes="", config=CONFIG):
    """JSON-serializable report dict (no image data). `result` is the cached
    analyze_timeline dict, `metadata` the cached clip_metadata dict."""
    now_utc = datetime.now(timezone.utc)
    return {
        "report": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "generated_utc": now_utc.isoformat(timespec="seconds"),
            "generated_local": now_utc.astimezone().isoformat(timespec="seconds"),
        },
        "case": {
            "case_reference": case_ref or "",
            "examiner": examiner or "",
            "notes": notes or "",
        },
        "evidence": {
            "filename": Path(source_path).name,
            "sha256": metadata["sha256"],
            "size_bytes": metadata["size_bytes"],
            "format": metadata.get("format"),
            "subtype": metadata.get("subtype"),
            "samplerate": metadata.get("samplerate"),
            "channels": metadata.get("channels"),
            "duration_sec": metadata.get("duration_sec"),
        },
        "analysis": {
            "model_file": Path(model_path).name,
            "model_sha256": sha256_file(model_path),
            "analyzed_duration_sec": result["analyzed_duration_sec"],
            "window_sec": result.get("window_sec", config["target_duration_sec"]),
            "hop_sec": result.get("hop_sec"),
            "n_windows": result["n_windows"],
            "suspect_threshold": timeline.SUSPECT_THRESHOLD,
            "clear_threshold": timeline.CLEAR_THRESHOLD,
            "pipeline": pipeline_params(config),
        },
        "findings": {
            "headline": result["headline"],
            "mean_fake_prob": result["mean_fake_prob"],
            "max_fake_prob": result["max_fake_prob"],
            "n_suspect": result["n_suspect"],
            "n_inconclusive": sum(1 for w in result["windows"] if w["verdict"] == "inconclusive"),
            "n_windows": result["n_windows"],
            "segments": [
                {k: w[k] for k in ("index", "start_sec", "end_sec", "fake_prob", "verdict")}
                for w in result["windows"]
            ],
        },
        "limitations": LIMITATIONS,
    }


def plain_language_summary(data):
    """Two to three sentences describing the finding in non-technical language.

    TEMPLATED, NOT GENERATED. The same analysis must always produce the same words:
    a forensic document whose wording drifted between runs would not be defensible.
    No model, no randomness, no dependence on the current date.

    Avoids stating any reading as a probability -- the limitations section is
    explicit that these are uncalibrated outputs, so the text frames every number
    as a score against a threshold, never as a percentage likelihood.
    """
    an, fi = data["analysis"], data["findings"]
    n = an["n_windows"]
    suspect, clear = an["suspect_threshold"], an["clear_threshold"]

    out = [
        f"This recording was examined in {n} overlapping four-second "
        f"{'segment' if n == 1 else 'segments'} covering "
        f"{an['analyzed_duration_sec']:.1f} seconds of audio."
    ]

    headline = fi["headline"]
    if headline == "Likely authentic":
        out.append(
            "No segment showed the characteristics this system associates with "
            "AI-generated speech; the strongest reading on any single segment was "
            f"{fi['max_fake_prob']:.2f}, against the {suspect:.2f} threshold at "
            "which a segment is flagged."
        )
    elif headline == "Likely synthetic":
        k = fi["n_suspect"]
        out.append(
            f"{k} of those segments showed characteristics this system associates with "
            f"AI-generated speech, the strongest reading being {fi['max_fake_prob']:.2f} "
            f"against a flagging threshold of {suspect:.2f}."
        )
    else:
        out.append(
            "The readings were mixed and fall between the two thresholds used to make a "
            f"finding, averaging {fi['mean_fake_prob']:.2f} where {clear:.2f} or below "
            f"indicates authentic speech and {suspect:.2f} or above indicates synthetic speech."
        )

    if headline == "Likely synthetic":
        flagged = sorted((s for s in fi["segments"] if s["verdict"] == "synthetic"),
                         key=lambda s: s["index"])
        idx = [s["index"] for s in flagged]
        if flagged and idx == list(range(idx[0], idx[0] + len(idx))):
            out.append(
                "The flagged segments are contiguous, covering "
                f"{min(s['start_sec'] for s in flagged):.1f} to "
                f"{max(s['end_sec'] for s in flagged):.1f} seconds of the recording."
            )
        else:
            out.append(
                "The flagged segments are distributed across the recording rather "
                "than forming one contiguous passage."
            )
    elif fi["n_inconclusive"] > 0:
        k = fi["n_inconclusive"]
        out.append(
            f"{k} segment{' fell' if k == 1 else 's fell'} into the inconclusive band "
            "and could not be classified either way."
        )
    return out


_CSS = """
:root { color-scheme: light; }
body { font-family: Arial, Helvetica, sans-serif; color: #1b1b1b; background: #fff;
       max-width: 960px; margin: 24px auto; padding: 0 16px; line-height: 1.45; }
h1 { margin-bottom: 0; }
h2 { border-bottom: 2px solid #ccc; padding-bottom: 4px; margin-top: 32px; }
.sub { color: #555; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
.kv th { width: 220px; }
.mono { font-family: Consolas, "Courier New", monospace; word-break: break-all; }
.band { padding: 16px 18px; border-radius: 6px; color: #fff; }
.band .headline { font-size: 1.8em; font-weight: 700; }
.pre { white-space: pre-wrap; }
.exhibit img { max-width: 100%; border: 1px solid #ccc; }
.pair { display: flex; gap: 12px; }
.pair figure { flex: 1; margin: 0; }
.pair img { width: 100%; image-rendering: pixelated; }
figcaption, .caption { font-size: 0.9em; color: #444; margin-top: 4px; }
tr.synthetic td { background: #f8d7d7; }
tr.inconclusive td { background: #fdf0c4; }
.limits { border: 1px solid #999; padding: 10px 14px; background: #f7f7f7; }
@media print {
  body { max-width: none; margin: 0; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  table, tr, figure, .band, .limits { page-break-inside: avoid; break-inside: avoid; }
  h2 { page-break-after: avoid; break-after: avoid; }
}
"""


def _e(value):
    """Escape anything that came from the user or the filesystem; None -> 'n/a'."""
    return "n/a" if value is None else html.escape(str(value), quote=True)


def _num(v, spec):
    return "n/a" if v is None else format(v, spec)


def _kv_table(rows):
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv">{body}</table>'


def render_technical_html(data, timeline_png, spectrogram_png, gradcam_png, selected_label,
                          summary_filename):
    """Complete standalone HTML document; images are data URIs. `*_png` arguments
    are data URIs as returned by png_data_uri(). `summary_filename` is used only
    for the cross-reference line."""
    rep, case, ev = data["report"], data["case"], data["evidence"]
    an, fi = data["analysis"], data["findings"]
    color = timeline.HEADLINE_COLORS[fi["headline"]]

    case_tbl = _kv_table([
        ("Case reference", _e(case["case_reference"])),
        ("Examiner", _e(case["examiner"])),
        ("Notes", f'<div class="pre">{_e(case["notes"])}</div>'),
    ])
    ev_tbl = _kv_table([
        ("File name", _e(ev["filename"])),
        ("SHA-256", f'<span class="mono">{_e(ev["sha256"])}</span>'),
        ("Size", f'{ev["size_bytes"]:,} bytes'),
        ("Container format", _e(ev["format"])),
        ("Sample format", _e(ev["subtype"])),
        ("Sample rate", f'{_e(ev["samplerate"])} Hz'),
        ("Channels", _e(ev["channels"])),
        ("Duration", f'{_num(ev["duration_sec"], ".2f")} s'),
    ])
    findings = (
        f'<div class="band" style="background:{color}">'
        f'<div class="headline">{_e(fi["headline"])}</div>'
        f'<div>mean p(synthetic) <b>{fi["mean_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'max <b>{fi["max_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'{fi["n_suspect"]} of {fi["n_windows"]} segments flagged '
        f'(p &ge; {an["suspect_threshold"]})</div></div>'
    )
    rows = "".join(
        f'<tr class="{_e(s["verdict"])}"><td>{s["index"]}</td><td>{s["start_sec"]:.2f}</td>'
        f'<td>{s["end_sec"]:.2f}</td><td>{s["fake_prob"]:.4f}</td><td>{_e(s["verdict"])}</td></tr>'
        for s in fi["segments"]
    )
    seg_tbl = (
        "<table><thead><tr><th>#</th><th>start (s)</th><th>end (s)</th>"
        f"<th>p(synthetic)</th><th>verdict</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    pipe = an["pipeline"]
    an_tbl = _kv_table([
        ("Model file", _e(an["model_file"])),
        ("Model SHA-256", f'<span class="mono">{_e(an["model_sha256"])}</span>'),
        ("Analyzed duration (after silence trim)", f'{an["analyzed_duration_sec"]:.2f} s'),
        ("Window / hop", f'{an["window_sec"]} s / {_num(an["hop_sec"], ".3f")} s'),
        ("Segments", _e(an["n_windows"])),
        ("Flag thresholds", f'synthetic &ge; {an["suspect_threshold"]}, authentic &le; {an["clear_threshold"]}'),
        ("Pipeline", "<br>".join(f"{html.escape(k)} = {_e(v)}" for k, v in pipe.items())),
    ])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FakeWave — Technical Report</title>
<style>{_CSS}</style></head><body>
<h1>FakeWave — Technical Report</h1>
<p class="sub">Summary report: {_e(summary_filename)}</p>
<p class="sub">Generated {_e(rep["generated_utc"])} (UTC) / {_e(rep["generated_local"])} (local) &middot; {_e(rep["tool"])} {_e(rep["tool_version"])}</p>

<h2>Case details</h2>
{case_tbl}

<h2>Evidence identification</h2>
{ev_tbl}

<h2>Findings</h2>
{findings}

<h2>Timeline exhibit</h2>
<div class="exhibit"><img src="{timeline_png}" alt="Timeline of p(synthetic) per segment">
<div class="caption">p(synthetic) for each {an["window_sec"]}-second window across the analyzed clip. Dashed lines mark the two thresholds.</div></div>

<h2>Selected segment exhibit</h2>
<div class="pair">
<figure><img src="{spectrogram_png}" alt="Mel spectrogram"><figcaption>Mel spectrogram &mdash; {_e(selected_label)}</figcaption></figure>
<figure><img src="{gradcam_png}" alt="Grad-CAM overlay"><figcaption>Grad-CAM overlay &mdash; {_e(selected_label)}</figcaption></figure>
</div>

<h2>Per-segment scores</h2>
{seg_tbl}

<h2>Analysis parameters</h2>
{an_tbl}

<h2>Limitations</h2>
<div class="limits">{_e(data["limitations"])}</div>
</body></html>
"""


def render_summary_html(data, timeline_png, technical_filename):
    """Standalone summary document: the finding in plain language plus the timeline.
    Rendered from the same data dict as the technical report."""
    rep, case, ev = data["report"], data["case"], data["evidence"]
    an, fi = data["analysis"], data["findings"]
    color = timeline.HEADLINE_COLORS[fi["headline"]]

    case_html = ""
    if any(case.values()):
        case_html = "<h2>Case details</h2>" + _kv_table([
            ("Case reference", _e(case["case_reference"])),
            ("Examiner", _e(case["examiner"])),
            ("Notes", f'<div class="pre">{_e(case["notes"])}</div>'),
        ])
    ev_tbl = _kv_table([
        ("File name", _e(ev["filename"])),
        ("Duration", f'{_num(ev["duration_sec"], ".2f")} s'),
        ("SHA-256", f'<span class="mono">{_e(ev["sha256"])}</span>'),
    ])
    band = (
        f'<div class="band" style="background:{color}">'
        f'<div class="headline">{_e(fi["headline"])}</div>'
        f'<div>mean p(synthetic) <b>{fi["mean_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'max <b>{fi["max_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'{fi["n_suspect"]} of {fi["n_windows"]} segments flagged '
        f'(p &ge; {an["suspect_threshold"]})</div></div>'
    )
    para = " ".join(_e(t) for t in plain_language_summary(data))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FakeWave — Summary Report</title>
<style>{_CSS}</style></head><body>
<h1>FakeWave — Summary Report</h1>
<p class="sub">Generated {_e(rep["generated_utc"])} (UTC) / {_e(rep["generated_local"])} (local) &middot; {_e(rep["tool"])} {_e(rep["tool_version"])}</p>

{case_html}

<h2>Evidence</h2>
{ev_tbl}

<h2>Findings</h2>
{band}
<p>{para}</p>

<h2>Timeline exhibit</h2>
<div class="exhibit"><img src="{timeline_png}" alt="Timeline of p(synthetic) per segment">
<div class="caption">p(synthetic) for each {an["window_sec"]}-second window across the analyzed clip. Dashed lines mark the two thresholds.</div></div>

<h2>Limitations</h2>
<div class="limits">{_e(data["limitations"])}</div>

<p><em>Full technical detail, including per-segment scores, spectrogram and Grad-CAM exhibits, and the analysis parameters, is in the accompanying technical report: {_e(technical_filename)}.</em></p>
</body></html>
"""


def report_filenames(data):
    """(summary, technical, json) filenames sharing one timestamp and hash."""
    stamp = datetime.fromisoformat(data["report"]["generated_utc"]).strftime("%Y%m%dT%H%M%SZ")
    tail = f"{stamp}_{data['evidence']['sha256'][:8]}"
    return (f"fakewave_summary_{tail}.html", f"fakewave_technical_{tail}.html",
            f"fakewave_report_{tail}.json")


def write_report(data, summary_html, technical_html, out_dir="reports"):
    """Write summary, technical and JSON files under out_dir; returns the three paths.
    Callers needing the filenames before rendering use report_filenames()."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = report_filenames(data)
    contents = (summary_html, technical_html, json.dumps(data, indent=2))
    for name, text in zip(names, contents):
        (out / name).write_text(text, encoding="utf-8")
    return [str(out / n) for n in names]
