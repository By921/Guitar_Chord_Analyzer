import io, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile
from matplotlib.figure import Figure
from chord_dsp import analyze, draw_plots, build_report, to_float, tune, NOTES

st.set_page_config(page_title="Guitar Chord Analyzer", page_icon="🎸", layout="centered")

# Fixed finger-position chart (low E -> high e; -1 = muted, 0 = open). Plain lookup data, nothing is trained.
SHAPES = {"C Major":[-1,3,2,0,1,0],"D Major":[-1,-1,0,2,3,2],"E Major":[0,2,2,1,0,0],"G Major":[3,2,0,0,0,3],
 "A Major":[-1,0,2,2,2,0],"A Minor":[-1,0,2,2,1,0],"D Minor":[-1,-1,0,2,3,1],"E Minor":[0,2,2,0,0,0],
 "F Major":[1,3,3,2,1,1],"C7":[-1,3,2,3,1,0],"D7":[-1,-1,0,2,1,2],"E7":[0,2,0,1,0,0],"G7":[3,2,0,0,0,1],
 "A7":[-1,0,2,0,2,0],"Am7":[-1,0,2,0,1,0],"Dm7":[-1,-1,0,2,1,1],"Em7":[0,2,0,0,0,0],"Asus2":[-1,0,2,2,0,0],
 "Asus4":[-1,0,2,2,3,0],"Dsus2":[-1,-1,0,2,3,0],"Dsus4":[-1,-1,0,2,3,3],"Esus4":[0,2,2,2,0,0],
 "Csus2":[-1,3,0,0,3,3],"Csus4":[-1,3,3,0,1,1],"Gsus2":[3,0,0,0,3,3],"Gsus4":[3,3,0,0,1,3]}

st.markdown("""<style>
.hero{background:linear-gradient(135deg,#6d28d9,#2563eb 60%,#06b6d4);padding:22px;border-radius:18px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.9rem;color:#fff}.hero p{margin:6px 0 0;color:#fff;opacity:.92}
.pills span{display:inline-block;background:rgba(255,255,255,.2);color:#fff;border-radius:99px;padding:3px 11px;margin:10px 6px 0 0;font-size:.8rem}
.chordcard{background:linear-gradient(145deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:20px;padding:22px 14px;text-align:center;margin:10px 0}
.lbl{color:#94a3b8;font-size:.75rem;letter-spacing:.12em}
.chordname{font-size:3.6rem;font-weight:800;line-height:1.15;background:linear-gradient(90deg,#a78bfa,#38bdf8);-webkit-background-clip:text;background-clip:text;color:transparent}
.chip{display:inline-block;padding:4px 13px;border-radius:99px;background:#334155;color:#e2e8f0;margin:3px;font-weight:700}
.chip.on{background:#16a34a;color:#fff}
.meter{height:12px;border-radius:99px;background:#334155;overflow:hidden;margin:12px auto 4px;max-width:420px}.meter>div{height:100%;border-radius:99px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:10px 0}
.stat{background:#141b34;border:1px solid #2a3558;border-radius:14px;padding:12px;text-align:center;color:#e8ecf8}
.stat b{display:block;font-size:1.2rem}.stat span{font-size:.72rem;color:#94a3b8}
.badge{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:700;font-size:.8rem;color:#fff}
</style>
<div class="hero"><h1>🎸 Guitar Chord Analyzer</h1>
<p>Play a chord, get the answer. Data Communication project · formulas only, no training.</p>
<div class="pills"><span>FFT</span><span>Chroma</span><span>Chord templates</span><span>Tuner</span><span>PDF report</span></div></div>""",
            unsafe_allow_html=True)

ss = st.session_state
for k, v in dict(k=0, nonce=0, rep=None, rep_key=None, pdf=None, hist=[]).items(): ss.setdefault(k, v)

def read_wav(b):
    sr, data = wavfile.read(io.BytesIO(b)); return int(sr), to_float(data)

def dark(fig):
    fig.patch.set_facecolor("#0f172a")
    for ax in fig.axes:
        ax.set_facecolor("#0f172a"); ax.title.set_color("#e2e8f0")
        ax.xaxis.label.set_color("#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        lg = ax.get_legend()
        if lg:
            lg.get_frame().set_facecolor("#1e293b"); lg.get_frame().set_edgecolor("#334155")
            for t in lg.get_texts(): t.set_color("#e2e8f0")
    return fig

def fretboard(frets):
    fig = Figure(figsize=(3.3, 3.6)); ax = fig.add_axes([.04, .02, .92, .9]); ax.axis("off")
    played = [f for f in frets if f > 0]; base = 1 if not played or max(played) <= 4 else min(played)
    for s in range(6): ax.plot([s, s], [0, 4], color="#94a3b8", lw=1.6)
    for fr in range(5): ax.plot([0, 5], [fr, fr], color="#e2e8f0" if (fr == 0 and base == 1) else "#64748b", lw=4 if (fr == 0 and base == 1) else 1.2)
    for s, f in enumerate(frets):
        if f > 0: ax.scatter(s, f - base + .5, s=300, color="#a78bfa", zorder=3)
        elif f == 0: ax.scatter(s, -.45, s=90, facecolors="none", edgecolors="#e2e8f0", lw=1.5)
        else: ax.text(s, -.45, "×", color="#f87171", ha="center", va="center", fontsize=15)
        ax.text(s, 4.4, "EADGBe"[s], color="#94a3b8", ha="center", fontsize=10)
    if base > 1: ax.text(-.35, .5, f"{base}fr", color="#e2e8f0", fontsize=9, va="center", ha="right")
    ax.set_xlim(-.7, 5.4); ax.set_ylim(4.7, -1.0); ax.set_title("Finger positions", color="#e2e8f0", fontsize=11)
    fig.patch.set_facecolor("#0f172a"); return fig

def chroma_fig(r):
    fig = Figure(figsize=(4.2, 3.4), constrained_layout=True); ax = fig.add_subplot(111)
    ax.bar(NOTES, r["chroma"], color=["#22c55e" if i in r["pcs"] else "#64748b" for i in range(12)])
    ax.set_ylim(0, 1.05); ax.set_title("Note strength (chroma)"); ax.tick_params(axis="x", labelsize=8)
    return dark(fig)

tabA, tabT, tabH, tabI = st.tabs(["🎼 Analyze", "🎚 Tuner", "📜 History", "📘 How it works"])

# ---------------- Analyze ----------------
with tabA:
    mode = st.radio("Input", ["🎤 Record with phone mic", "📁 Upload WAV file"], horizontal=True)
    if mode.startswith("🎤"):
        audio = st.audio_input("Play ONE chord for 1–2 seconds, then stop the recording", sample_rate=44100, key=f"rec{ss.k}")
        source = "Recording (device microphone)"
    else:
        audio = st.file_uploader("WAV recording of one chord", type=["wav"], key=f"up{ss.k}")
        source = "WAV file: " + (audio.name if audio else "")
    with st.expander("⚙️ Settings / Test Mode"):
        reduce = st.checkbox("Noise reduction", True)
        noise = st.slider("Add artificial noise (test)", 0.0, 1.0, 0.0, 0.05)
        dist = st.slider("Add artificial distortion (test)", 0.0, 1.0, 0.0, 0.05)
    c1, c2 = st.columns(2)
    if c1.button("↻ Re-analyze"):
        ss.rep = ss.rep_key = ss.pdf = None; ss.nonce += 1          # delete everything, then analyze again below
    if c2.button("🗑 Clear / record again"):
        ss.k += 1; ss.nonce += 1; ss.rep = ss.rep_key = ss.pdf = None; st.rerun()

    if audio is None:
        ss.rep = ss.rep_key = ss.pdf = None
    else:
        b = audio.getvalue(); key = (hashlib.md5(b).hexdigest(), reduce, noise, dist, ss.nonce)
        if ss.rep_key != key:
            try:
                with st.spinner("Analyzing…"):
                    sr, raw = read_wav(b); raw = raw[-sr * 5:]
                    r = analyze(raw, sr, reduce, noise, dist)
                    meta = dict(reduce=reduce, test=(noise > 0 or dist > 0), noise=noise, dist=dist, source=source)
                    ss.rep = dict(raw=raw, sr=sr, result=r, meta=meta, time=datetime.now()); ss.rep_key = key
                    buf = io.BytesIO(); build_report(ss.rep).savefig(buf, format="pdf"); ss.pdf = buf.getvalue()
                    ss.hist.insert(0, {"Time": ss.rep["time"].strftime("%H:%M:%S"), "Source": source, "Chord": r["chord"],
                                       "Confidence": f"{r['confidence']:.0f}%", "Quality": r["quality"]})
            except Exception as e:
                ss.rep = ss.pdf = None; st.error("Could not analyze this audio: " + str(e))

    rep = ss.rep
    if rep is None:
        st.info("👆 Record or upload one chord and the result appears here.")
    else:
        r = rep["result"]; cf = r["confidence"]; tones = [NOTES[i] for i in r["pcs"]]
        col = "#22c55e" if cf >= 70 else "#f59e0b" if cf >= 48 else "#ef4444"
        chips = "".join(f'<span class="chip{" on" if n in tones else ""}">{n}</span>' for n in r["notes"] if n != "—") or '<span class="chip">—</span>'
        st.markdown(f"""<div class="chordcard"><div class="lbl">DETECTED CHORD</div><div class="chordname">{r['chord']}</div>
<div>{chips}</div><div class="meter"><div style="width:{cf:.0f}%;background:{col}"></div></div>
<div class="lbl">CONFIDENCE {cf:.0f}%</div></div>
<div class="grid"><div class="stat"><span>SIGNAL QUALITY</span><b>{r['quality']}</b></div>
<div class="stat"><span>DOMINANT FREQUENCY</span><b>{r['dominant']:.1f} Hz</b></div>
<div class="stat"><span>SIGNAL LEVEL</span><b>{r['db']:.1f} dBFS</b></div>
<div class="stat"><span>SOURCE</span><b style="font-size:.9rem">{'Test Mode' if rep['meta']['test'] else 'Normal'}</b></div></div>""",
                    unsafe_allow_html=True)
        if r.get("reason"): st.warning("Note: " + r["reason"])
        g1, g2 = st.columns(2)
        with g1:
            if r["chord"] in SHAPES: st.pyplot(fretboard(SHAPES[r["chord"]]))
            else: st.caption("No finger chart stored for this chord (barre or unusual shape).")
        with g2:
            if r["chroma"].any(): st.pyplot(chroma_fig(r))
        fig = Figure(figsize=(7, 5.6), constrained_layout=True)
        draw_plots(fig.add_subplot(211), fig.add_subplot(212), rep["raw"], rep["sr"], r); st.pyplot(dark(fig))
        name = r["chord"].replace("#", "sharp").replace(" ", "_")
        st.download_button("💾 Save report (PDF)", ss.pdf, file_name=f"chord_report_{name}_{rep['time']:%Y%m%d_%H%M%S}.pdf", mime="application/pdf")
        st.caption("Confidence is a heuristic score, not a calibrated probability.")

# ---------------- Tuner ----------------
with tabT:
    st.markdown("Pluck **one open string**, stop the recording, and see how far off it is.")
    ta = st.audio_input("Record one string", sample_rate=44100, key=f"tun{ss.k}")
    if ta is not None:
        try:
            sr, raw = read_wav(ta.getvalue()); res = tune(raw, sr)
        except Exception as e:
            res = None; st.error("Could not read the recording: " + str(e))
        else:
            if res is None: st.warning("Too quiet. Pluck louder or hold the phone closer.")
            else:
                c = res["cents"]; pos = 50 + max(-50, min(50, c))
                msg = "✔ In tune" if abs(c) < 5 else ("Too low — tighten the string" if c < 0 else "Too high — loosen the string")
                st.markdown(f"""<div class="chordcard"><div class="lbl">NEAREST STRING</div><div class="chordname">{res['string']}</div>
<div style="position:relative;height:16px;border-radius:99px;margin:22px 8px;background:linear-gradient(90deg,#ef4444,#f59e0b 35%,#22c55e 50%,#f59e0b 65%,#ef4444)">
<div style="position:absolute;left:{pos}%;top:-6px;width:6px;height:28px;background:#fff;border-radius:3px;transform:translateX(-50%);box-shadow:0 0 6px #000"></div></div>
<div style="color:#e8ecf8;font-size:1.1rem"><b>{res['freq']:.2f} Hz</b> · target {res['target']:.2f} Hz · <b>{c:+.1f} cents</b></div>
<div style="color:#e8ecf8;margin-top:6px">{msg}</div></div>""", unsafe_allow_html=True)

# ---------------- History ----------------
with tabH:
    if ss.hist:
        st.dataframe(pd.DataFrame(ss.hist))
        if st.button("Clear history"): ss.hist = []; st.rerun()
    else: st.info("Your analyses in this session will be listed here.")

# ---------------- How it works ----------------
with tabI:
    st.markdown("Every step is a fixed mathematical formula. **Nothing is trained**; chord templates are just lists of notes.")
    st.markdown("**1. Level**"); st.latex(r"RMS=\sqrt{\tfrac{1}{N}\sum x[n]^2},\quad dBFS=20\log_{10}(RMS)")
    st.markdown("**2. Hann window**"); st.latex(r"w[n]=0.5-0.5\cos\!\left(\tfrac{2\pi n}{N-1}\right)")
    st.markdown("**3. Fourier transform (FFT)**"); st.latex(r"X[k]=\sum_{n=0}^{N-1}x[n]\,e^{-j2\pi kn/N},\quad f=\tfrac{k\,f_s}{N}")
    st.markdown("**4. Frequency → note**"); st.latex(r"n=\mathrm{round}\!\left(69+12\log_2\tfrac{f}{440}\right),\quad pitch\ class=n \bmod 12")
    st.markdown("**5. Chord match**"); st.latex(r"S=\mathrm{mean}(C_{chord\ notes})-0.2\,\mathrm{mean}(C_{other\ notes})")
    st.markdown("**6. Tuner**"); st.latex(r"cents=1200\log_2\tfrac{f}{f_{target}}")
    st.caption("Also: a tone check (peak ÷ median ≥ 20) and a minimum level stop silence or hiss from being named as a chord.")
