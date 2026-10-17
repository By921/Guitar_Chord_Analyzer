"""chord_dsp.py - formula-only chord analysis (no training, no machine learning).
Shared DSP code: FFT, chroma, chord templates, plots and PDF report. No GUI / microphone libraries needed."""
from datetime import datetime
import numpy as np
from scipy import signal
from matplotlib.figure import Figure

NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CHORDS = {
"C Major":[0,4,7],"C Minor":[0,3,7],"C# Major":[1,5,8],
"D Major":[2,6,9],"D Minor":[2,5,9],"D# Major":[3,7,10],
"E Major":[4,8,11],"E Minor":[4,7,11],"F Major":[5,9,0],
"F Minor":[5,8,0],"F# Major":[6,10,1],"G Major":[7,11,2],
"G Minor":[7,10,2],"G# Major":[8,0,3],"A Major":[9,1,4],
"A Minor":[9,0,4],"A# Major":[10,2,5],"B Major":[11,3,6],
"B Minor":[11,2,6],"C7":[0,4,7,10],"D7":[2,6,9,0],
"E7":[4,8,11,2],"G7":[7,11,2,5],"A7":[9,1,4,7],
"Am7":[9,0,4,7],"Dm7":[2,5,9,0],"Em7":[4,7,11,2],
"Csus2":[0,2,7],"Dsus2":[2,4,9],"Esus2":[4,6,11],
"Gsus2":[7,9,2],"Asus2":[9,11,4],"Csus4":[0,5,7],
"Dsus4":[2,7,9],"Esus4":[4,9,11],"Gsus4":[7,0,2],"Asus4":[9,2,4]}
TEMPLATES = {}
for name, pcs in CHORDS.items():
    t = np.zeros(12); t[pcs] = 1; TEMPLATES[name] = t

def to_float(data):
    x = np.asarray(data)
    if x.ndim == 2: x = x.astype(np.float32).mean(axis=1) if np.issubdtype(x.dtype, np.integer) else x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype); x = x.astype(np.float32) / max(abs(info.min), info.max)
    else: x = x.astype(np.float32)
    return np.nan_to_num(x)

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    if not len(x): return x
    x = x - np.mean(x); p = np.max(np.abs(x))
    return x / p if p > 1e-8 else x

def effects(x, noise, distortion):
    y = x.copy()
    if noise:
        y += np.random.normal(0, np.sqrt(np.mean(y*y)+1e-12)*noise, len(y)).astype(np.float32)
    if distortion:
        y = np.tanh(y*(1+22*distortion)); y /= np.max(np.abs(y))+1e-9
    return y.astype(np.float32)

def denoise(x, strength=.45):
    if len(x) < 512: return x
    n=len(x); sp=np.fft.rfft(x*np.hanning(n)); mag=np.abs(sp); phase=np.angle(sp)
    floor=np.percentile(mag,22); threshold=floor*(1.5+7*strength)
    gain=np.clip(mag/(threshold+1e-12),.12,1)
    gain=np.where(mag<threshold,gain,1)
    return np.fft.irfft(mag*gain*np.exp(1j*phase),n=n).astype(np.float32)

def bandpass(x,sr):
    if len(x)<64: return x
    ny=sr/2; lo=65; hi=min(1500,ny*.92)
    try:
        sos=signal.butter(4,[lo,hi],btype="bandpass",fs=sr,output="sos")
        try: return signal.sosfiltfilt(sos,x).astype(np.float32)
        except ValueError: return signal.sosfilt(sos,x).astype(np.float32)
    except ValueError: return x

def pc_for_hz(hz):
    if hz<=0: return None
    return int(round(69+12*np.log2(hz/440)))%12

TONALITY_MIN=20.0        # spectrum peak / median. Pure noise scores ~8-9, real chords score far higher.
LIVE_LEVEL_GATE=0.002    # live mode: raw mic level (RMS) below this counts as silence

def _no_chord(sig,sr,chord,reason,quality,db,sp=None):
    """Result used when there is nothing chord-like to report (silence / noise only)."""
    n=len(sig)
    if sp is None: sp=np.abs(np.fft.rfft(sig*np.hanning(n)))
    return dict(processed=sig,freqs=np.fft.rfftfreq(n,1/sr),spectrum=sp,chord=chord,confidence=0.0,
                notes=["—"],quality=quality,dominant=0.0,db=db,hold=True,reason=reason)

def analyze(raw,sr,reduce_noise=True,noise=0,distortion=0,gate=0.0):
    x0=to_float(raw)
    if len(x0)<1024: raise ValueError("Audio is too short; use a clip longer than 0.1 seconds.")
    # Level is measured on the ORIGINAL signal. (Normalising first made room noise look as loud as a guitar.)
    level=float(np.sqrt(np.mean(x0*x0)+1e-12)); db=20*np.log10(level+1e-12)
    clipped=float(np.mean(np.abs(x0)>=.99))>.001
    if gate and level<gate:
        return _no_chord(x0,sr,"No sound","no sound picked up",
                         "TOO QUIET",db)
    x=norm(x0)
    if noise or distortion: x=effects(x,noise,distortion)
    processed=denoise(x) if reduce_noise else x.copy()
    processed=norm(bandpass(processed,sr))
    n=len(processed)
    sp=np.abs(np.fft.rfft(processed*np.hanning(n)))
    freqs=np.fft.rfftfreq(n,1/sr)
    mask=(freqs>=65)&(freqs<=min(1500,sr*.45))
    f,m=freqs[mask],sp[mask]
    # Is there any musical tone at all? Flat, hissy spectra are rejected instead of being forced into a chord.
    tonality=float(m.max()/(np.median(m)+1e-12)) if len(m) else 0.0
    if tonality<TONALITY_MIN:
        return _no_chord(processed,sr,"Uncertain","no clear musical tone (noise only?)",
                         "NOISY / UNCERTAIN",db,sp)
    chroma=np.zeros(12)
    for hz,amp in zip(f,m/(np.sqrt(f)+1)):
        pc=pc_for_hz(float(hz))
        if pc is not None: chroma[pc]+=amp
    if chroma.max()>0: chroma/=chroma.max()
    chord,conf,notes,strong="Uncertain",0.0,["—"],[]
    if chroma.max()>=1e-8:
        c=chroma/(chroma.max()+1e-12); scores={}
        for name,pcs in CHORDS.items():
            t=TEMPLATES[name]; tone=t>0
            scores[name]=max(0,float(c[tone].mean())-.20*float(c[~tone].mean()))
        ranked=sorted(scores.items(),key=lambda z:z[1],reverse=True)
        chord,best=ranked[0]; second=ranked[1][1]
        conf=float(np.clip(100*(.78*best+.22*max(0,best-second)*2),0,99))
        strong=[i for i in np.argsort(c)[::-1] if c[i]>=.38]
        notes=[NOTES[i] for i in strong][:5] or ["—"]
        if len(strong)<2: conf=min(conf,24.0)      # a single note is not a chord
        if conf<35: chord="Uncertain"      # too weak to name a chord
    quality="TOO QUIET" if level<.003 else ("POSSIBLE CLIPPING" if clipped else ("GOOD" if conf>=70 else ("FAIR" if conf>=48 else "NOISY / UNCERTAIN")))
    dominant=float(f[np.argmax(m)]) if len(m) else 0
    return dict(processed=processed, freqs=freqs, spectrum=sp,
                chord=chord,confidence=conf,notes=notes,quality=quality,
                dominant=dominant,db=db,hold=False,reason="")

def draw_plots(axw,axf,raw,sr,r):
    """Waveform + spectrum, used by both the live window and the saved report."""
    n=min(len(raw),int(sr*1.5)); t=np.arange(n)/sr
    axw.clear(); axw.plot(t,raw[:n],lw=.75,label="Input")
    proc=r["processed"]; n2=min(n,len(proc)); axw.plot(np.arange(n2)/sr,proc[:n2],lw=.75,alpha=.75,label="Filtered")
    axw.set_title("Waveform: input vs processed"); axw.set_xlabel("Time (s)"); axw.set_ylabel("Amplitude"); axw.grid(True,alpha=.25); axw.legend(loc="upper right")
    f,s=r["freqs"],r["spectrum"]; mask=f<=min(1500,sr/2)
    axf.clear(); axf.plot(f[mask],s[mask],lw=.8); axf.set_title("FFT Frequency Spectrum")
    axf.set_xlabel("Frequency (Hz)"); axf.set_ylabel("Magnitude"); axf.set_xlim(0,min(1500,sr/2)); axf.grid(True,alpha=.25)

def build_report(rep):
    """Build an A4 report page (chord, metrics, settings, both plots) from a saved analysis."""
    r,raw,sr,meta=rep["result"],rep["raw"],rep["sr"],rep["meta"]
    fig=Figure(figsize=(8.27,11.69))
    gs=fig.add_gridspec(3,1,height_ratios=[1.15,1,1],left=.11,right=.95,top=.96,bottom=.09,hspace=.5)
    a=fig.add_subplot(gs[0]); a.axis("off")
    a.text(0,1.0,"Guitar Chord Analysis Report",fontsize=20,fontweight="bold",va="top",transform=a.transAxes)
    a.text(0,.86,r["chord"],fontsize=40,fontweight="bold",va="top",color="#1f4e79",transform=a.transAxes)
    a.text(0,.64,"Detected pitch classes:  "+"   ".join(r["notes"]),fontsize=12,va="top",transform=a.transAxes)
    rows=[("Confidence",f'{r["confidence"]:.0f}%  (heuristic score)'),("Signal quality",r["quality"]),
          ("Dominant frequency",f'{r["dominant"]:.1f} Hz'),("Signal level",f'{r["db"]:.1f} dBFS'),
          ("Source",meta["source"]+(" [Test Mode]" if meta["test"] else "")),
          ("Analyzed on",rep["time"].strftime("%Y-%m-%d %H:%M:%S")),
          ("Noise reduction","On" if meta["reduce"] else "Off"),("Sample rate",f"{sr} Hz")]
    if meta["test"]: rows.insert(7,("Test effects",f'Noise {meta["noise"]:.2f}, Distortion {meta["dist"]:.2f}'))
    if r.get("reason"): rows.append(("Note",r["reason"]))
    y=.54
    for k,v in rows:
        a.text(0,y,k+":",fontsize=10.5,fontweight="bold",va="top",transform=a.transAxes)
        a.text(.30,y,v,fontsize=10.5,va="top",transform=a.transAxes); y-=.064
    axw=fig.add_subplot(gs[1]); axf=fig.add_subplot(gs[2])
    draw_plots(axw,axf,raw,sr,r)
    fig.text(.11,.025,"Educational prototype: confidence is a heuristic score, not a calibrated probability.",fontsize=8,color="#555555")
    return fig
