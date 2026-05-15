"""
app.py — NeuroScan AI  (Fixed Session State Pipeline)
Patient records voice → features extracted automatically → fed into model
No manual input, no file upload needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import time

from fusion_model import predict_stage, voice_features
from voice_extractor import extract_voice_features

# ------------------------------------------------------------------ #
#  PAGE CONFIG & SESSION STATE INITIALIZATION
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize persistent session storage to prevent reruns from wiping data
if "voice_input" not in st.session_state:
    st.session_state.voice_input = None
if "voice_feat_dict" not in st.session_state:
    st.session_state.voice_feat_dict = None
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------------------------------------------------------ #
#  GLOBAL CSS
# ------------------------------------------------------------------ #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #040d1a !important;
    color: #e2eaf7 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* HERO */
.hero {
    min-height: 92vh;
    background:
        radial-gradient(ellipse 90% 55% at 50% -5%, rgba(0,220,200,0.13) 0%, transparent 65%),
        radial-gradient(ellipse 50% 35% at 85% 85%, rgba(255,150,40,0.07) 0%, transparent 55%),
        linear-gradient(180deg, #040d1a 0%, #061525 55%, #040d1a 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 2rem; position: relative; overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 39px,
        rgba(0,220,200,0.015) 40px
    );
    pointer-events: none;
}
.eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.35em; color: #00dcc8; text-transform: uppercase;
    margin-bottom: 1.4rem;
    animation: fadeUp 0.7s 0.1s both;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.6rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.05; text-align: center;
    background: linear-gradient(140deg, #fff 0%, #b0d4ff 45%, #00dcc8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1.3rem;
    animation: fadeUp 0.7s 0.25s both;
}
.hero-sub {
    font-size: 1.05rem; color: #6a8fac; text-align: center;
    max-width: 540px; line-height: 1.75; margin-bottom: 2.5rem;
    animation: fadeUp 0.7s 0.4s both;
}
.hero-pills {
    display: flex; gap: 0.6rem; flex-wrap: wrap;
    justify-content: center; margin-bottom: 2.8rem;
    animation: fadeUp 0.7s 0.55s both;
}
.pill {
    padding: 0.35rem 0.9rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 500; letter-spacing: 0.04em;
    border: 1px solid;
}
.pill-t { background: rgba(0,220,200,0.08); border-color: rgba(0,220,200,0.28); color: #00dcc8; }
.pill-a { background: rgba(255,150,40,0.08); border-color: rgba(255,150,40,0.28); color: #ff9628; }
.pill-b { background: rgba(100,180,255,0.08); border-color: rgba(100,180,255,0.28); color: #64b4ff; }
.scroll-hint {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: #2a4a6a; letter-spacing: 0.2em; text-transform: uppercase;
    animation: fadeUp 0.7s 0.85s both, blink 2.5s 1.5s infinite;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes blink {
    0%,100% { opacity: 0.3; } 50% { opacity: 1; }
}

/* FORM AREA */
.fwrap { max-width: 1000px; margin: 0 auto; padding: 4rem 2rem 2rem; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,220,200,0.18), transparent); }

/* SECTION LABELS */
.slabel { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.28em; color:#00dcc8; text-transform:uppercase; margin-bottom:0.4rem; }
.stitle { font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:700; color:#e2eaf7; margin-bottom:0.4rem; }
.sdesc  { font-size:0.92rem; color:#4a6a8a; line-height:1.65; margin-bottom:1.8rem; }

/* CARD */
.card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px; padding: 1.8rem;
    margin-bottom: 1.4rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(0,220,200,0.18); }
.card-title {
    font-family:'Sora',sans-serif; font-weight:600; font-size:0.95rem;
    color:#c0d8f0; display:flex; align-items:center; gap:0.6rem; margin-bottom:1.3rem;
}
.step-dot {
    width:30px; height:30px; border-radius:50%;
    background: linear-gradient(135deg,#00dcc8,#0099aa);
    display:inline-flex; align-items:center; justify-content:center;
    font-family:'Sora',sans-serif; font-weight:700; font-size:0.8rem; color:#040d1a;
    flex-shrink:0;
}

/* WIDGETS */
.stSlider>div>div>div>div { background:#00dcc8 !important; }
.stSlider>div>div>div { background:rgba(0,220,200,0.12) !important; }
div[data-baseweb="select"]>div {
    background:rgba(255,255,255,0.04) !important;
    border-color:rgba(255,255,255,0.1) !important;
    color:#e2eaf7 !important; border-radius:10px !important;
}
.stNumberInput input, div[data-baseweb="input"]>div {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    color:#e2eaf7 !important; border-radius:10px !important;
}
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color:#6a8fac !important; font-size:0.84rem !important;
    font-weight:500 !important; letter-spacing:0.02em !important;
}
.stSelectSlider [data-testid="stMarkdownContainer"] p { color:#e2eaf7 !important; }

/* BUTTON */
.stButton>button {
    width:100%; padding:1rem 2rem !important;
    background: linear-gradient(135deg,#00dcc8 0%,#0088aa 100%) !important;
    color:#040d1a !important;
    font-family:'Sora',sans-serif !important; font-weight:700 !important;
    font-size:1rem !important; letter-spacing:0.06em !important;
    border:none !important; border-radius:14px !important;
    box-shadow:0 0 36px rgba(0,220,200,0.22) !important;
    transition:all 0.2s !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 0 56px rgba(0,220,200,0.38) !important;
}

/* VOICE FEATURE GRID */
.vf-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.5rem; margin-top:0.8rem; }
.vf-item { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.5rem 0.7rem; }
.vf-name { font-family:'DM Mono',monospace; font-size:0.65rem; color:#2a4a6a; margin-bottom:0.15rem; }
.vf-val  { font-size:0.82rem; color:#8aaccc; font-weight:500; }

/* RESULTS & METRICS */
.res-hero { border-radius:20px; padding:2.5rem 2rem; text-align:center; margin-bottom:1.8rem; position:relative; overflow:hidden; }
.res-hero.healthy  { background:linear-gradient(135deg,rgba(0,220,150,0.09),rgba(0,180,120,0.04)); border:1px solid rgba(0,220,150,0.28); }
.res-hero.mild     { background:linear-gradient(135deg,rgba(0,220,200,0.09),rgba(0,150,200,0.04)); border:1px solid rgba(0,220,200,0.28); }
.res-hero.moderate { background:linear-gradient(135deg,rgba(255,150,40,0.09),rgba(220,100,20,0.04)); border:1px solid rgba(255,150,40,0.28); }
.res-hero.severe   { background:linear-gradient(135deg,rgba(255,75,75,0.09),rgba(200,40,40,0.04)); border:1px solid rgba(255,75,75,0.28); }
.res-icon { font-size:3rem; margin-bottom:0.8rem; }
.res-eyebrow { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.28em; color:#3a5a7a; text-transform:uppercase; margin-bottom:0.4rem; }
.res-label  { font-family:'Sora',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; margin-bottom:0.7rem; }
.res-label.healthy  { color:#00dc96; }
.res-label.mild     { color:#00dcc8; }
.res-label.moderate { color:#ff9628; }
.res-label.severe   { color:#ff5050; }
.res-msg  { font-size:0.92rem; color:#6a8fac; max-width:460px; margin:0 auto; line-height:1.65; }
.res-conf { font-family:'DM Mono',monospace; font-size:0.8rem; color:#3a5a7a; margin-top:0.75rem; }

.pb-row { display:flex; align-items:center; gap:0.9rem; margin-bottom:0.65rem; }
.pb-lbl { font-family:'DM Mono',monospace; font-size:0.72rem; color:#4a6a8a; width:78px; flex-shrink:0; }
.pb-wrap { flex:1; height:7px; background:rgba(255,255,255,0.05); border-radius:100px; overflow:hidden; }
.pb-fill { height:100%; border-radius:100px; }
.pb-fill.mild     { background:linear-gradient(90deg,#00dcc8,#008899); }
.pb-fill.moderate { background:linear-gradient(90deg,#ff9628,#cc6600); }
.pb-fill.severe   { background:linear-gradient(90deg,#ff5050,#bb2020); }
.pb-val { font-family:'DM Mono',monospace; font-size:0.78rem; color:#6a8fac; width:42px; text-align:right; flex-shrink:0; }

.mt { width:100%; border-collapse:collapse; font-size:0.82rem; }
.mt th { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase; color:#2a4a6a; padding:0.45rem 0.7rem; text-align:left; border-bottom:1px solid rgba(255,255,255,0.05); }
.mt td { padding:0.55rem 0.7rem; color:#8aaccc; border-bottom:1px solid rgba(255,255,255,0.025); }
.mt td:first-child { color:#c0d8f0; font-weight:500; }
.tag { display:inline-block; padding:0.12rem 0.45rem; border-radius:4px; font-size:0.68rem; font-family:'DM Mono',monospace; }
.tag-t { background:rgba(0,220,200,0.12); color:#00dcc8; }
.tag-a { background:rgba(255,150,40,0.12); color:#ff9628; }
.tag-r { background:rgba(255,80,80,0.12);  color:#ff5050; }
.tag-g { background:rgba(0,220,150,0.12);  color:#00dc96; }
.tag-x { background:rgba(255,255,255,0.06); color:#4a6a8a; }

.ins { display:flex; gap:0.7rem; align-items:flex-start; padding:0.65rem 0.8rem; border-radius:10px; background:rgba(255,255,255,0.018); margin-bottom:0.45rem; }
.ins-ico { font-size:0.95rem; flex-shrink:0; margin-top:1px; }
.ins-txt { font-size:0.86rem; color:#6a8fac; line-height:1.55; }

div[data-testid="stMetric"] { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:0.9rem 1.1rem; }
div[data-testid="stMetricLabel"] { color:#3a5a7a !important; font-size:0.76rem !important; }
div[data-testid="stMetricValue"] { color:#e2eaf7 !important; font-family:'Sora',sans-serif !important; font-weight:700 !important; }

.footer { text-align:center; padding:2rem 1rem; color:#1a3a5a; font-size:0.76rem; font-family:'DM Mono',monospace; letter-spacing:0.03em; border-top:1px solid rgba(255,255,255,0.035); max-width:1000px; margin:0 auto; }
div[data-testid="stExpander"] { background:rgba(255,255,255,0.015) !important; border:1px solid rgba(255,255,255,0.05) !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  HERO
# ------------------------------------------------------------------ #
st.markdown("""
<div class="hero">
    <div class="eyebrow">Multi-Modal · AI-Powered · Clinical Decision Support</div>
    <div class="hero-title">NeuroScan AI<br>Parkinson's Detection</div>
    <div class="hero-sub">
        Fuses clinical biomarkers, voice acoustics recorded live, and
        motor tapping patterns to detect Parkinson's and classify its stage
        with interpretable AI.
    </div>
    <div class="hero-pills">
        <span class="pill pill-t">🧠 Clinical Model</span>
        <span class="pill pill-b">⚙️ Motor Tapping</span>
        <span class="pill pill-a">🎙️ Live Voice Recording</span>
        <span class="pill pill-t">4-Class Detection</span>
    </div>
    <div class="scroll-hint">▼ &nbsp; scroll to begin</div>
</div>
<div class="divider"></div>
<div class="fwrap">
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  STEP 1 — PATIENT INFO
# ------------------------------------------------------------------ #
st.markdown("""
<div class="slabel">Step 01</div>
<div class="stitle">Patient Information</div>
<div class="sdesc">Enter basic demographic and disease history details.</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title"><span class="step-dot">1</span>Demographics & History</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=65)
with c2:
    duration = st.number_input("Disease Duration (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
with c3:
    updrs = st.number_input("UPDRS III Motor Score (0–132)", min_value=0.0, max_value=132.0, value=25.0, step=1.0)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  STEP 2 — CLINICAL SYMPTOMS
# ------------------------------------------------------------------ #
st.markdown("""
<div class="slabel" style="margin-top:2rem">Step 02</div>
<div class="stitle">Clinical Motor Symptoms</div>
<div class="sdesc">Rate each symptom on the standard 0–4 UPDRS scale.</div>
""", unsafe_allow_html=True)

OPTS = [0, 1, 2, 3, 4]
def fg(x): return ["None","Mild","Moderate","Severe","Very Severe"][x]
def ff(x): return ["Normal","Slight Loss","Reduced","Masked","Fixed"][x]
def fw(x): return ["Normal","Slight","Slow","Needs Aid","Unable"][x]
def fp(x): return ["Normal","Slight","Unstable","Falls","Frequent Falls"][x]

st.markdown('<div class="card"><div class="card-title"><span class="step-dot">2</span>Motor & Non-Motor Assessment</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    speech   = st.select_slider("🗣️ Speech Difficulty",  options=OPTS, value=1, format_func=fg)
    tremor   = st.select_slider("🤚 Tremor at Rest",     options=OPTS, value=1, format_func=fg)
with col2:
    facial   = st.select_slider("😐 Facial Expression", options=OPTS, value=1, format_func=ff)
    rigidity = st.select_slider("💪 Muscle Rigidity",   options=OPTS, value=1, format_func=fg)
with col3:
    gait     = st.select_slider("🚶 Gait / Walking",    options=OPTS, value=1, format_func=fw)
    postural = st.select_slider("⚖️ Postural Stability",options=OPTS, value=1, format_func=fp)
st.markdown('</div>', unsafe_allow_html=True)

clinical_input = [age, duration, updrs, speech, facial, tremor, rigidity, gait, postural]

# ------------------------------------------------------------------ #
#  STEP 3 — LIVE VOICE RECORDING
# ------------------------------------------------------------------ #
st.markdown("""
<div class="slabel" style="margin-top:2rem">Step 03</div>
<div class="stitle">Live Voice Recording</div>
<div class="sdesc">
    Record the patient saying a sustained <strong style="color:#00dcc8">"Aaaaah"</strong> vowel sound for 5–7 seconds.
    The system automatically extracts all 22 MDVP acoustic biomarkers from the recording.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🎙️ Audio Recording Capture Panel</div>', unsafe_allow_html=True)
audio_file = st.audio_input("Record Voice Sample", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#2a4a6a;font-size:0.8rem;margin:0.5rem 0;">— or upload a .wav file directly —</p>', unsafe_allow_html=True)
uploaded_wav = st.file_uploader("Upload .wav voice recording", type=["wav"], label_visibility="collapsed")

# Process raw microphone/uploader assets into the Session State cache layer
if audio_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    try:
        with st.spinner("Processing microphone audio and aligning MDVP array..."):
            voice_feat_dict = extract_voice_features(tmp_path)
            # Map features to voice_features column sequence order from fusion_model
            st.session_state.voice_input = [float(voice_feat_dict.get(col, 0.0)) for col in voice_features]
            st.session_state.voice_feat_dict = voice_feat_dict
    except Exception as e:
        st.error(f"Live processing error: {e}")
    finally:
        os.unlink(tmp_path)

elif uploaded_wav is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_wav.read())
        tmp_path = tmp.name
    try:
        with st.spinner("Extracting voice biomarkers from file asset…"):
            voice_feat_dict = extract_voice_features(tmp_path)
            st.session_state.voice_input = [float(voice_feat_dict.get(col, 0.0)) for col in voice_features]
            st.session_state.voice_feat_dict = voice_feat_dict
    except Exception as e:
        st.error(f"File analysis error: {e}")
    finally:
        os.unlink(tmp_path)

# Always render visual indicators based on persistent Session State caches
if st.session_state.voice_feat_dict:
    st.success(f"✅ Voice features ready in pipeline memory — {len(st.session_state.voice_input)} elements aligned.")
    with st.expander("📊 Extracted Voice Biomarkers (22 MDVP features)"):
        items_html = "".join(
            f'<div class="vf-item"><div class="vf-name">{k}</div><div class="vf-val">{v:.5f}</div></div>'
            for k, v in st.session_state.voice_feat_dict.items()
        )
        st.markdown(f'<div class="vf-grid">{items_html}</div>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#2a4a6a;font-size:0.84rem;font-style:italic;text-align:center;padding:0.5rem;">No active voice matrix detected — structural weights will adapt smoothly to clinical parameters.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  PREDICT BUTTON & ACTION TRIGGER
# ------------------------------------------------------------------ #
st.markdown('<div style="margin-top:2.5rem;margin-bottom:1rem;">', unsafe_allow_html=True)
predict_clicked = st.button("🔬 Run NeuroScan Analysis")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close fwrap

if predict_clicked:
    with st.spinner("Running multi-modal fusion analysis…"):
        time.sleep(0.4)
    try:
        # Crucial Note: Testing with UPDRS < 10 bypasses this entirely to show Stage 0 (Healthy)
        stage, prob, debug = predict_stage(clinical_input, st.session_state.voice_input)
        st.session_state.result = (stage, prob, debug, clinical_input, st.session_state.voice_feat_dict)
    except ValueError as e:
        st.error(f"**Input Error:** {e}")
        st.session_state.result = None
    except Exception as e:
        st.error(f"**Analysis failed:** {e}")
        st.session_state.result = None

# ------------------------------------------------------------------ #
#  RESULTS GENERATION
# ------------------------------------------------------------------ #
if st.session_state.result:
    stage, prob, debug, ci, vfd = st.session_state.result
    age_r, dur_r, updrs_r, speech_r, facial_r, tremor_r, rigidity_r, gait_r, postural_r = ci

    st.markdown('<div class="divider"></div><div class="fwrap">', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Analysis Complete</div><div class="stitle">Diagnostic Result</div>', unsafe_allow_html=True)

    CFG = {
        0: ("healthy",  "✅", "Healthy",           "#00dc96",
            "No significant Parkinson's indicators detected. Routine annual screening advised."),
        1: ("mild",     "🟦", "Stage 1 — Mild",    "#00dcc8",
            "Early-stage Parkinson's. Symptoms are minimal and unilateral. Regular neurologist follow-up recommended."),
        2: ("moderate", "🟧", "Stage 2 — Moderate","#ff9628",
            "Moderate bilateral symptoms affecting daily activities. Medical consultation and physiotherapy advised."),
        3: ("severe",   "🔴", "Stage 3 — Severe",  "#ff5050",
            "Advanced motor impairment with postural instability. Immediate specialist care required."),
    }
    cls, icon, lbl, col, msg = CFG[stage]

    st.markdown(f"""
    <div class="res-hero {cls}">
        <div class="res-icon">{icon}</div>
        <div class="res-eyebrow">NeuroScan AI Prediction</div>
        <div class="res-label {cls}">{lbl}</div>
        <p class="res-msg">{msg}</p>
        <div class="res-conf">
            Confidence: {debug['max_prob']*100:.1f}%
            {' ⚠️ Low confidence — collect more data' if not debug['confident'] else ''}
            ; Voice: {'✅ Used' if debug['voice_used'] else '⊘ Skipped'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    lc, rc = st.columns([1, 1], gap="large")

    with lc:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.67rem;letter-spacing:0.22em;color:#2a4a6a;text-transform:uppercase;margin-bottom:1rem;">Stage Probabilities</p>', unsafe_allow_html=True)
        for lbl_p, cls_p, p in [("Mild","mild",prob[0]),("Moderate","moderate",prob[1]),("Severe","severe",prob[2])]:
            st.markdown(f"""
            <div class="pb-row">
                <span class="pb-lbl">{lbl_p}</span>
                <div class="pb-wrap"><div class="pb-fill {cls_p}" style="width:{p*100:.0f}%"></div></div>
                <span class="pb-val">{p*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("UPDRS III", f"{updrs_r:.0f}")
        m2.metric("Duration", f"{dur_r:.1f} yrs")
        m3, m4 = st.columns(2)
        m3.metric("Motor Sum", f"{speech_r+facial_r+tremor_r+rigidity_r+gait_r+postural_r}/24")
        m4.metric("Age", f"{age_r}")

    with rc:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.67rem;letter-spacing:0.22em;color:#2a4a6a;text-transform:uppercase;margin-bottom:1rem;">Model Contributions</p>', unsafe_allow_html=True)

        def ptag(arr):
            i = int(np.argmax(arr))
            tgs = [("Mild","tag-t"),("Moderate","tag-a"),("Severe","tag-r")]
            lb, tc = tgs[i]
            return f'<span class="tag {tc}">{lb} {arr[i]*100:.0f}%</span>'

        v_tag = ptag(debug['voice_prob']) if debug['voice_used'] and debug['voice_prob'] else '<span class="tag tag-x">Skipped</span>'
        rows = [
            ("🧠 Clinical", f"{debug['weights']['clinical']*100:.0f}%", ptag(debug['clinical_prob']), "XGBoost · 9 features"),
            ("⚙️ Tapping",  f"{debug['weights']['tap']*100:.0f}%",      ptag(debug['tap_prob']),      "RF · stage profiles"),
            ("🎙️ Voice",    f"{debug['weights']['voice']*100:.0f}%",     v_tag,                        "RF · 22 MDVP features"),
        ]
        rows_html = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td style='color:#2a4a6a;font-size:0.72rem;'>{r[3]}</td></tr>" for r in rows)
        st.markdown(f'<table class="mt"><thead><tr><th>Model</th><th>Weight</th><th>Says</th><th>Method</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if vfd:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.67rem;letter-spacing:0.22em;color:#2a4a6a;text-transform:uppercase;margin-bottom:0.6rem;">Key Voice Biomarkers</p>', unsafe_allow_html=True)
            key_feats = ['MDVP:Fo(Hz)','HNR','NHR','RPDE','PPE','DFA']
            items = "".join(f'<div class="vf-item"><div class="vf-name">{k}</div><div class="vf-val">{vfd.get(k,0):.4f}</div></div>' for k in key_feats if k in vfd)
            st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.4rem;">{items}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:0.5rem;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.67rem;letter-spacing:0.22em;color:#2a4a6a;text-transform:uppercase;margin-bottom:0.8rem;">💡 Clinical Insights</p>', unsafe_allow_html=True)

    insights = []
    if updrs_r > 35:   insights.append(("🔴", f"UPDRS III of {updrs_r:.0f} indicates significant motor impairment (>35)"))
    elif updrs_r > 20: insights.append(("🟡", f"UPDRS III of {updrs_r:.0f} reflects moderate motor involvement (20–35)"))
    else:              insights.append(("🟢", f"UPDRS III of {updrs_r:.0f} is in the mild range (<20)"))
    if tremor_r >= 3:   insights.append(("⚠️", "Severe resting tremor — a hallmark Parkinson's motor sign"))
    if rigidity_r >= 3: insights.append(("⚠️", "Marked muscle rigidity — associated with advanced motor involvement"))
    if gait_r >= 2:     insights.append(("🚶", "Gait disturbance present — assess fall risk"))
    if postural_r >= 2: insights.append(("⚖️", "Postural instability — physiotherapy recommended"))
    if dur_r > 5:       insights.append(("📅", f"Disease duration of {dur_r:.1f} yrs — greater neurodegeneration likely"))
    if vfd:
        hnr = vfd.get('HNR', 25)
        ppe = vfd.get('PPE', 0.2)
        if hnr < 15:   insights.append(("🎙️", f"Low HNR ({hnr:.1f} dB) — significant voice noise, consistent with PD dysphonia"))
        if ppe > 0.35: insights.append(("🎙️", f"Elevated PPE ({ppe:.3f}) — irregular pitch, a key PD voice biomarker"))
    if not debug['voice_used']:
        insights.append(("🎙️", "Voice not recorded — recording a sustained vowel would add the 3rd modality and improve accuracy"))
    if not debug['confident']:
        insights.append(("📊", "Confidence below 50% — borderline case, additional neurological assessment recommended"))
    if not insights:
        insights.append(("✅", "All indicators within mild range — early monitoring protocol advised"))

    for ico, txt in insights:
        st.markdown(f'<div class="ins"><span class="ins-ico">{ico}</span><span class="ins-txt">{txt}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 Technical Details — Model Internals"):
        st.json(debug)

    st.markdown('</div>', unsafe_allow_html=True)  # close fwrap

# ------------------------------------------------------------------ #
#  FOOTER
# ------------------------------------------------------------------ #
st.markdown("""
<div class="footer">
    ⚠️ &nbsp;NeuroScan AI is a research tool only — not a certified medical device.<br>
    Always consult a qualified neurologist for clinical diagnosis and treatment.<br><br>
    <span>Clinical · Motor Tapping · Live Voice · Multi-Modal Fusion</span>
</div>
""", unsafe_allow_html=True)