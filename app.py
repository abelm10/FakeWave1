"""
Simple web app: upload or record Hindi audio and get a segment-by-segment
real/fake analysis across the WHOLE clip, plus an explanation of what drove the
score for any chosen segment -- its mel spectrogram and a Grad-CAM heatmap over
it (see gradcam.py; hooks the actual trained detector.pt, not any
pretrained/reference weights).

The clip is scored with sliding 4-second windows (see timeline.py), so a long
recording is no longer judged from one window in the middle. The headline
verdict comes from the MEAN p(synthetic) across windows; the timeline shows
where in the clip the flagged regions are. Grad-CAM is computed only for the
selected window, so the pictures and the number always agree.

Usage:
    python app.py
Then open http://127.0.0.1:7860
"""

import gradio as gr

print(f"gradio {gr.__version__}")

from predict import load_model
import batch
import report
import timeline

MODEL_PATH = "detector.pt"   # keep in sync with whatever load_model actually opens
_model = load_model(MODEL_PATH)  # loaded once at startup, reused across requests

_PLACEHOLDER = '<div style="padding:18px;border-radius:8px;background:#eee;color:#555">Upload a clip and press Analyze.</div>'
_TABLE_HEADERS = ["#", "start (s)", "end (s)", "p(synthetic)", "verdict"]


def _banner(result):
    color = timeline.HEADLINE_COLORS[result["headline"]]
    return (
        f'<div style="padding:18px;border-radius:8px;background:{color};color:#fff">'
        f'<div style="font-size:1.8em;font-weight:700">{result["headline"]}</div>'
        f'<div style="margin-top:6px">mean p(synthetic) <b>{result["mean_fake_prob"]:.2f}</b>'
        f' &nbsp;·&nbsp; max <b>{result["max_fake_prob"]:.2f}</b></div>'
        f'<div>{result["n_suspect"]} of {result["n_windows"]} segments flagged</div>'
        f'</div>'
    )


def _details_md(meta, analyzed_sec):
    def fmt(v, spec=""):
        return "n/a" if v is None else format(v, spec)
    return (
        "**Clip details**\n\n"
        f"- Duration: {fmt(meta['duration_sec'], '.1f')} s (analyzed after silence trim: {analyzed_sec:.1f} s)\n"
        f"- Sample rate: {fmt(meta['samplerate'])} Hz\n"
        f"- Channels: {fmt(meta['channels'])}\n"
        f"- Format: {fmt(meta['format'])} / {fmt(meta['subtype'])}\n"
        f"- Size: {meta['size_bytes']:,} bytes\n"
        f"- SHA-256: `{meta['sha256'][:16]}…`"
    )


def _choices(windows):
    return [
        (f"{w['index']}: {w['start_sec']:.1f}-{w['end_sec']:.1f}s  (p={w['fake_prob']:.2f})", w["index"])
        for w in windows
    ]


def analyze(audio_path, progress=gr.Progress()):
    if audio_path is None:
        raise gr.Error("Upload or record a clip first.")
    try:
        meta = timeline.clip_metadata(audio_path)
        progress(0, desc="Decoding and conditioning audio")
        result = timeline.analyze_timeline(
            audio_path, _model,
            progress_cb=lambda done, total: progress(done / total, desc=f"Scoring segment {done}/{total}"),
        )
        windows = result["windows"]
        best = max(windows, key=lambda w: w["fake_prob"])
        _, spec_rgb, overlay_rgb = timeline.explain_window(
            result["conditioned_wav"], result["sr"],
            round(best["start_sec"] * result["sr"]), round(best["end_sec"] * result["sr"]), _model,
        )
        strip = timeline.render_timeline_strip(result)
        rows = [[w["index"], round(w["start_sec"], 2), round(w["end_sec"], 2),
                 round(w["fake_prob"], 4), w["verdict"]] for w in windows]
    except Exception as exc:
        raise gr.Error(f"Could not analyze this clip: {exc}")
    return (
        _details_md(meta, result["analyzed_duration_sec"]),
        _banner(result),
        strip,
        gr.update(choices=_choices(windows), value=best["index"]),
        _window_audio(result, best),
        spec_rgb,
        overlay_rgb,
        rows,
        # the report is built from this cached dict alone (never by re-running the analysis)
        {**result, "source_path": audio_path, "metadata": meta},
    )


def _window_audio(result, w):
    """Slice the cached conditioned waveform for one window (no re-decode, no re-analysis)."""
    sr = result["sr"]
    return timeline.window_audio(result["conditioned_wav"], sr, round(w["start_sec"] * sr), round(w["end_sec"] * sr))


def show_window(index, result):
    """Dropdown handler: Grad-CAM for one window of the cached analysis. Does not re-run the timeline."""
    if index is None or result is None:
        return None, None, None
    try:
        w = result["windows"][int(index)]
        sr = result["sr"]
        _, spec_rgb, overlay_rgb = timeline.explain_window(
            result["conditioned_wav"], sr, round(w["start_sec"] * sr), round(w["end_sec"] * sr), _model,
        )
    except Exception as exc:
        raise gr.Error(f"Could not analyze this clip: {exc}")
    return _window_audio(result, w), spec_rgb, overlay_rgb


def make_report(result, index, case_ref, examiner, notes):
    """Build the case report from the cached analysis only: no re-decode of the
    source file and no re-run of analyze_timeline."""
    if result is None:
        raise gr.Error("Analyze a clip first")
    try:
        sr = result["sr"]
        w = result["windows"][int(index) if index is not None else 0]
        _, spec_rgb, overlay_rgb = timeline.explain_window(
            result["conditioned_wav"], sr, round(w["start_sec"] * sr), round(w["end_sec"] * sr), _model,
        )
        strip = timeline.render_timeline_strip(result)
        data = report.build_report_data(
            result, result["metadata"], result["source_path"], MODEL_PATH,
            case_ref=case_ref, examiner=examiner, notes=notes,
        )
        label = f"segment {w['index']}: {w['start_sec']:.1f}-{w['end_sec']:.1f}s (p={w['fake_prob']:.2f})"
        html_str = report.render_report_html(
            data, report.png_data_uri(strip), report.png_data_uri(spec_rgb),
            report.png_data_uri(overlay_rgb), label,
        )
        html_path, json_path = report.write_report(data, html_str)
    except Exception as exc:
        raise gr.Error(f"Could not generate the report: {exc}")
    return [html_path, json_path]


def run_batch(folder, recursive, expected, progress=gr.Progress()):
    """Batch tab handler: forward passes only, results to reports/. The scanned folder is never written to."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        raise gr.Error("Enter a folder path.")
    expected_label = {"all real": "real", "all fake": "fake"}.get(expected)
    try:
        files = batch.find_audio_files(folder, bool(recursive))
    except FileNotFoundError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Could not read this folder: {exc}")
    if not files:
        raise gr.Error(f"No audio files ({', '.join(sorted(batch.AUDIO_EXTS))}) found in {folder}")
    try:
        result = batch.analyze_folder(
            folder, _model, recursive=bool(recursive), expected_label=expected_label, model_path=MODEL_PATH,
            progress_cb=lambda done, total, name: progress(done / total, desc=f"{done}/{total}: {name}"),
        )
        csv_path, json_path = batch.write_batch_csv(result)
    except Exception as exc:
        raise gr.Error(f"Batch failed: {exc}")
    lines = [
        f"**Files scanned:** {result['n_files']} &nbsp; **Succeeded:** {result['n_ok']} &nbsp; **Failed:** {result['n_failed']}",
    ]
    if expected_label is not None:
        acc = "n/a" if result["accuracy"] is None else f"{result['accuracy']:.1%}"
        lines.append(
            f"**Expected {expected_label}:** {result['n_correct']} / {result['n_ok']} correct ({acc}) "
            f"&nbsp; **Inconclusive:** {result['n_inconclusive']} (counted as neither right nor wrong)"
        )
    table_rows = [[("" if r[c] is None else r[c]) for c in batch.CSV_COLUMNS] for r in result["rows"]]
    return "\n\n".join(lines), table_rows, [csv_path, json_path]


with gr.Blocks(title="FakeWave") as demo:
    gr.Markdown(
        '<h1 style="text-align:center">FakeWave</h1>\n'
        '<h3 style="text-align:center">Fake voice detection with XAI</h3>'
    )
    result_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Single clip"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(type="filepath", label="Hindi audio clip (upload or record)")
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    details_md = gr.Markdown("*Clip details appear here after analysis.*")
                    with gr.Accordion("Case details", open=False):
                        case_ref_box = gr.Textbox(label="Case reference")
                        examiner_box = gr.Textbox(label="Examiner")
                        notes_box = gr.Textbox(label="Notes", lines=3)
                with gr.Column(scale=2):
                    banner = gr.HTML(_PLACEHOLDER)
                    strip_img = gr.Image(label="Timeline: p(synthetic) per 4-second window", show_label=True)
                    window_dd = gr.Dropdown(label="Segment (defaults to the most suspicious)", choices=[], value=None)
                    segment_audio = gr.Audio(label="Selected segment (as analyzed)", interactive=False)
                    gr.Markdown(
                        "*Playback is the conditioned audio the model scored — 16 kHz, codec-normalized, "
                        "silence-trimmed. It is not the original recording, and these timestamps are "
                        "relative to the trimmed audio.*"
                    )
                    with gr.Row():
                        spectrogram_img =gr.Image(label="Mel spectrogram (selected segment)", height=320)
                        overlay_img = gr.Image(label="Grad-CAM (selected segment)", height=320)
                    report_btn = gr.Button("Generate case report")
                    report_files = gr.File(label="Case report", file_count="multiple")
                    with gr.Accordion("All segment scores", open=False):
                        table = gr.Dataframe(headers=_TABLE_HEADERS, interactive=False)
                    clear_btn = gr.ClearButton(
                        [audio_input, details_md, banner, strip_img, window_dd, segment_audio, spectrogram_img, overlay_img, table]
                    )

        with gr.Tab("Batch"):
            gr.Markdown("Score every audio file in a folder. The folder is only read; results are written to `reports/`.")
            batch_folder = gr.Textbox(label="Folder path")
            batch_recursive = gr.Checkbox(label="Include subfolders", value=False)
            batch_expected = gr.Dropdown(["(unknown)", "all real", "all fake"], value="(unknown)", label="Expected label")
            batch_btn = gr.Button("Run batch", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(headers=batch.CSV_COLUMNS, interactive=False)
            batch_files = gr.File(label="Batch results (CSV + JSON)", file_count="multiple")

    outputs = [details_md, banner, strip_img, window_dd, segment_audio, spectrogram_img, overlay_img, table, result_state]
    # Button only: wiring audio_input.change as well would analyze every upload twice.
    analyze_btn.click(fn=analyze, inputs=audio_input, outputs=outputs)
    window_dd.change(fn=show_window, inputs=[window_dd, result_state], outputs=[segment_audio, spectrogram_img, overlay_img])
    clear_btn.click(fn=lambda: None, inputs=None, outputs=result_state)
    report_btn.click(
        fn=make_report,
        inputs=[result_state, window_dd, case_ref_box, examiner_box, notes_box],
        outputs=report_files,
    )

    batch_btn.click(fn=run_batch, inputs=[batch_folder, batch_recursive, batch_expected], outputs=[batch_summary, batch_table, batch_files])

if __name__ == "__main__":
    demo.launch()
