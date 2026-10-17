import io, hashlib
from datetime import datetime
import numpy as np
import streamlit as st
from scipy.io import wavfile
from matplotlib.figure import Figure
from chord_dsp import analyze, draw_plots, build_report, to_float

st.set_page_config(page_title="Guitar Chord Analyzer", page_icon="🎸", layout="centered")
st.title("🎸 Guitar Chord Analyzer")
st.caption("Data Communication project · formulas only (FFT, chroma, chord templates) · no training, no machine learning")

ss = st.session_state
ss.setdefault("k", 0)        # changes -> recorder/uploader reset
ss.setdefault("nonce", 0)    # changes -> forces a fresh analysis (new test noise)
ss.setdefault("rep", None); ss.setdefault("rep_key", None); ss.setdefault("pdf", None)

mode = st.radio("Input", ["🎤 Record with phone mic", "📁 Upload WAV file"], horizontal=True)
audio, source = None, ""
if mode.startswith("🎤"):
    audio = st.audio_input("Play ONE chord for 1–2 seconds, then stop the recording",
                           sample_rate=44100, key=f"rec{ss.k}")
    source = "Recording (device microphone)"
else:
    audio = st.file_uploader("WAV recording of one chord", type=["wav"], key=f"up{ss.k}")
    source = "WAV file: " + (audio.name if audio else "")

with st.expander("Settings / Test Mode"):
    reduce = st.checkbox("Noise reduction", True)
    noise = st.slider("Add artificial noise (test)", 0.0, 1.0, 0.0, 0.05)
    dist = st.slider("Add artificial distortion (test)", 0.0, 1.0, 0.0, 0.05)

c1, c2 = st.columns(2)
if c1.button("↻ Re-analyze", use_container_width=True):
    ss.rep = None; ss.rep_key = None; ss.pdf = None      # delete every detected result...
    ss.nonce += 1                                          # ...then produce a brand-new one below
if c2.button("🗑 Clear / record again", use_container_width=True):
    ss.k += 1; ss.nonce += 1; ss.rep = None; ss.rep_key = None; ss.pdf = None
    st.rerun()

if audio is not None:
    data_bytes = audio.getvalue()
    key = (hashlib.md5(data_bytes).hexdigest(), reduce, noise, dist, ss.nonce)
    if ss.rep_key != key:                                  # analyze only when something changed
        try:
            with st.spinner("Analyzing…"):
                sr, data = wavfile.read(io.BytesIO(data_bytes))
                raw = to_float(data)[-int(int(sr) * 5):]
                r = analyze(raw, int(sr), reduce, noise, dist)
                meta = dict(reduce=reduce, test=(noise > 0 or dist > 0), noise=noise, dist=dist, source=source)
                ss.rep = dict(raw=raw, sr=int(sr), result=r, meta=meta, time=datetime.now())
                ss.rep_key = key
                buf = io.BytesIO(); build_report(ss.rep).savefig(buf, format="pdf"); ss.pdf = buf.getvalue()
        except Exception as e:
            ss.rep = None; ss.pdf = None
            st.error("Could not analyze this audio: " + str(e))

if audio is None:                                          # recorder/uploader emptied -> drop old result
    ss.rep = None; ss.rep_key = None; ss.pdf = None

rep = ss.rep
if rep is None:
    st.info("Record or upload one chord, and the result appears here.")
else:
    r = rep["result"]
    st.markdown(f"<div style='text-align:center;font-size:3.4rem;font-weight:800;line-height:1.1'>{r['chord']}</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'>Detected pitch classes: " + " • ".join(r["notes"]) + "</div>",
                unsafe_allow_html=True)
    if r.get("reason"): st.warning("Note: " + r["reason"])
    a, b = st.columns(2)
    a.metric("Confidence", f"{r['confidence']:.0f}%"); b.metric("Signal quality", r["quality"])
    a.metric("Dominant frequency", f"{r['dominant']:.1f} Hz"); b.metric("Signal level", f"{r['db']:.1f} dBFS")
    fig = Figure(figsize=(7, 6), constrained_layout=True)
    draw_plots(fig.add_subplot(211), fig.add_subplot(212), rep["raw"], rep["sr"], r)
    st.pyplot(fig)
    name = r["chord"].replace("#", "sharp").replace(" ", "_")
    st.download_button("💾 Save report (PDF)", ss.pdf, file_name=f"chord_report_{name}_{rep['time']:%Y%m%d_%H%M%S}.pdf",
                       mime="application/pdf", use_container_width=True)
    st.caption("Confidence is a heuristic score, not a calibrated probability.")
