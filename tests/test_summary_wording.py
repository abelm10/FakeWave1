"""Wording tests for report.plain_language_summary().

Run: python tests/test_summary_wording.py   (no pytest; exits non-zero on failure)
Builds data dicts inline; does not read data/, load the model, or use the network.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import report  # noqa: E402

SUSPECT, CLEAR = 0.65, 0.35


def make(headline, probs, step=2.0, win=4.0):
    segs = []
    for i, p in enumerate(probs):
        v = "synthetic" if p >= SUSPECT else "authentic" if p <= CLEAR else "inconclusive"
        segs.append({"index": i, "start_sec": i * step, "end_sec": i * step + win,
                     "fake_prob": p, "verdict": v})
    return {
        "analysis": {"suspect_threshold": SUSPECT, "clear_threshold": CLEAR,
                     "n_windows": len(probs),
                     "analyzed_duration_sec": step * (len(probs) - 1) + win},
        "findings": {
            "headline": headline, "segments": segs,
            "n_suspect": sum(s["verdict"] == "synthetic" for s in segs),
            "n_inconclusive": sum(s["verdict"] == "inconclusive" for s in segs),
            "max_fake_prob": max(probs), "mean_fake_prob": sum(probs) / len(probs),
            "n_windows": len(probs),
        },
    }


LOW = 0.03
# name -> (data, substrings that must appear, substrings that must not appear, n sentences)
CASES = {
    "authentic clean": (make("Likely authentic", [0.05, 0.1, 0.08, 0.04]),
        ["No segment showed", "0.10"], ["flagging threshold, the strongest"], 2),
    "authentic clean + inconclusive": (make("Likely authentic", [0.05, 0.5, 0.4, 0.1]),
        ["No segment showed", "2 segments fell into the inconclusive band"], [], 3),
    "authentic, one flagged": (make("Likely authentic", [LOW, LOW, 0.7, LOW, LOW, LOW, LOW, LOW]),
        ["low overall, but 1 segment reached", "That segment covers 4.0 to 8.0 seconds"],
        ["No segment showed", "segments reached"], 3),
    "authentic, contiguous flags": (make("Likely authentic", [LOW, 0.7, 0.8, LOW, LOW, LOW, LOW, LOW, LOW]),
        ["2 segments reached", "contiguous, covering 2.0 to 8.0 seconds"], ["No segment showed"], 3),
    "authentic, scattered flags": (make("Likely authentic", [0.7, LOW, LOW, LOW, LOW, LOW, 0.9, LOW, LOW]),
        ["2 segments reached", "distributed across the recording"], ["No segment showed"], 3),
    "inconclusive": (make("Inconclusive", [0.5, 0.4, 0.6, 0.55]),
        ["mixed", "averaging 0.51", "4 segments fell into the inconclusive band"], [], 3),
    "synthetic contiguous": (make("Likely synthetic", [0.1, 0.8, 0.9, 0.85, 0.2]),
        ["3 of those segments", "contiguous, covering 2.0 to 10.0 seconds"], [], 3),
    "synthetic scattered": (make("Likely synthetic", [0.9, 0.1, 0.8, 0.2, 0.95]),
        ["3 of those segments", "distributed across the recording"], [], 3),
    "synthetic, exactly one flagged of several": (make("Likely synthetic", [0.9, 0.5, 0.5]),
        ["One of those segments", "That segment covers 0.0 to 4.0 seconds"], [], 3),
    "single-segment authentic": (make("Likely authentic", [0.1]),
        ["in a single four-second segment", "No segment showed"], ["overlapping"], 2),
    "single-segment flagged (synthetic)": (make("Likely synthetic", [0.9]),
        ["in a single four-second segment", "That segment showed characteristics",
         "a reading of 0.90 against a flagging threshold of 0.65."],
        ["overlapping", "One of those", "covers"], 2),
    "single-segment flagged (authentic headline)": (make("Likely authentic", [0.9]),
        ["That segment showed characteristics"], ["No segment showed", "covers"], 2),
}


def main():
    failures = []
    for name, (data, must, must_not, n_sent) in CASES.items():
        a = report.plain_language_summary(data)
        b = report.plain_language_summary(data)
        print(f"\n[{name}]")
        for s in a:
            print("  " + s)
        text = " ".join(a)
        try:
            assert a == b, "not deterministic"
            assert len(a) == n_sent, f"expected {n_sent} sentences, got {len(a)}"
            for m in must:
                assert m in text, f"missing {m!r}"
            for m in must_not:
                assert m not in text, f"unexpected {m!r}"
            if data["findings"]["n_suspect"] > 0:
                assert "No segment showed" not in text, "regression: 'No segment showed' with flags"
            assert "%" not in text, "probability stated as a percentage"
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"OK: {len(CASES)} cases passed")


if __name__ == "__main__":
    main()
