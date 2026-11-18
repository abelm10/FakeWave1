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
import timeline

_model = load_model()  # loaded once at startup, reused across requests

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
        spec_rgb,
        overlay_rgb,
        rows,
        result,
    )


def show_window(index, result):
    """Dropdown handler: Grad-CAM for one window of the cached analysis. Does not re-run the timeline."""
    if index is None or result is None:
        return None, None
    try:
        w = result["windows"][int(index)]
        sr = result["sr"]
        _, spec_rgb, overlay_rgb = timeline.explain_window(
            result["conditioned_wav"], sr, round(w["start_sec"] * sr), round(w["end_sec"] * sr), _model,
        )
    except Exception as exc:
        raise gr.Error(f"Could not analyze this clip: {exc}")
    return spec_rgb, overlay_rgb


with gr.Blocks(title="FakeWave") as demo:
    gr.Markdown(
        '<h1 style="text-align:center">FakeWave</h1>\n'
        '<h3 style="text-align:center">Fake voice detection with XAI</h3>'
    )
    result_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Hindi audio clip (upload or record)")
            analyze_btn = gr.Button("Analyze", variant="primary")
            details_md = gr.Markdown("*Clip details appear here after analysis.*")
        with gr.Column(scale=2):
            banner = gr.HTML(_PLACEHOLDER)
            strip_img = gr.Image(label="Timeline: p(synthetic) per 4-second window", show_label=True)
            window_dd = gr.Dropdown(label="Segment (defaults to the most suspicious)", choices=[], value=None)
            with gr.Row():
                spectrogram_img = gr.Image(label="Mel spectrogram (selected segment)", height=320)
                overlay_img = gr.Image(label="Grad-CAM (selected segment)", height=320)
            with gr.Accordion("All segment scores", open=False):
                table = gr.Dataframe(headers=_TABLE_HEADERS, interactive=False)
            clear_btn = gr.ClearButton(
                [audio_input, details_md, banner, strip_img, window_dd, spectrogram_img, overlay_img, table]
            )

    outputs = [details_md, banner, strip_img, window_dd, spectrogram_img, overlay_img, table, result_state]
    # Button only: wiring audio_input.change as well would analyze every upload twice.
    analyze_btn.click(fn=analyze, inputs=audio_input, outputs=outputs)
    window_dd.change(fn=show_window, inputs=[window_dd, result_state], outputs=[spectrogram_img, overlay_img])
    clear_btn.click(fn=lambda: None, inputs=None, outputs=result_state)

if __name__ == "__main__":
    demo.launch()
