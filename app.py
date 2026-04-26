# import streamlit as st
# import pandas as pd
# import joblib

# # ---------------- CONFIG ----------------
# st.set_page_config(page_title="Parkinson Detection", layout="wide")

# # ---------------- SESSION ----------------
# if "page" not in st.session_state:
#     st.session_state.page = "home"

# def go_home():
#     st.session_state.page = "home"

# def go_app():
#     st.session_state.page = "app"

# # ---------------- STYLE ----------------
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: white;
# }
# .hero {
#     text-align: center;
#     padding: 120px 20px;
# }
# .card {
#     background: rgba(255,255,255,0.08);
#     padding: 20px;
#     border-radius: 15px;
#     margin-top: 20px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ================= LANDING =================
# if st.session_state.page == "home":

#     st.markdown("""
#     <div class="hero">
#         <h1>🧠 Parkinson Detection System</h1>
#         <p>AI + Clinical Hybrid Model for Stage Prediction</p>
#     </div>
#     """, unsafe_allow_html=True)

#     if st.button("🚀 Start"):
#         go_app()

#     st.stop()

# # ================= MAIN =================
# if st.session_state.page == "app":

#     if st.button("⬅ Back"):
#         go_home()

#     st.title("🧪 Parkinson Prediction System")

#     # ---------------- LOAD MODEL ----------------
#     try:
#         model = joblib.load("parkinson_model.pkl")
#         features = joblib.load("features.pkl")
#     except:
#         st.error("❌ Run train_model.py first")
#         st.stop()

#     # ================= DATASET =================
#     st.markdown("### 📂 Bulk Dataset Prediction")

#     file = st.file_uploader("Upload CSV", type=["csv"])

#     if file is not None:
#         df = pd.read_csv(file, header=1)
#         df.columns = df.columns.str.strip()

#         st.write("### 🔍 Dataset Preview")
#         st.dataframe(df.head())

#         # Rename
#         df = df.rename(columns={
#             "Age (years)": "Age",
#             "UPDRS III total (-)": "UPDRS III",
#             "Duration of disease from first symptoms (years)": "Disease Duration",
#             "18. Speech": "Speech Score",
#             "19. Facial Expression": "Facial Expression",
#             "20. Tremor at Rest - head": "Tremor (Head)",
#             "Levodopa equivalent (mg/day)": "Levodopa (mg/day)"
#         })

#         df_model = df[features].replace("-", pd.NA)
#         df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

#         if st.button("📊 Predict Dataset"):

#             stages = model.predict(df_model)

#             results = []

#             for i in range(len(df)):
#                 stage = stages[i]
#                 updrs_val = df_model.iloc[i]["UPDRS III"]

#                 # 🔥 HYBRID CORRECTION
#                 if updrs_val > 50:
#                     stage = max(stage, 3.5)
#                 elif updrs_val > 30:
#                     stage = max(stage, 2.5)

#                 # Classification
#                 if stage < 1:
#                     status = "Healthy"
#                     severity = "None"
#                 elif stage < 2:
#                     status = "Parkinson's"
#                     severity = "Stage 1 (Mild)"
#                 elif stage < 3:
#                     status = "Parkinson's"
#                     severity = "Stage 2 (Moderate)"
#                 elif stage < 4:
#                     status = "Parkinson's"
#                     severity = "Stage 3 (Advanced)"
#                 else:
#                     status = "Parkinson's"
#                     severity = "Stage 4-5 (Severe)"

#                 results.append({
#                     "Patient": i+1,
#                     "Age": df_model.iloc[i]["Age"],
#                     "UPDRS III": updrs_val,
#                     "Stage": round(stage, 2),
#                     "Status": status,
#                     "Severity": severity
#                 })

#             result_df = pd.DataFrame(results)

#             st.write("### 📈 Results")
#             st.dataframe(result_df)
#             st.bar_chart(result_df["Stage"])

#     # ================= INDIVIDUAL =================
#     st.markdown("---")
#     st.markdown("### 🧑‍⚕️ Individual Patient Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 0, 120, 30)
#         duration = st.number_input("Disease Duration", 0.0, 50.0, 0.0)
#         levodopa = st.number_input("Levodopa", 0.0, 2000.0, 0.0)

#     with col2:
#         updrs = st.number_input("UPDRS III", 0.0, 200.0, 0.0)
#         speech = st.number_input("Speech Score", 0.0, 10.0, 0.0)
#         facial = st.number_input("Facial Expression", 0.0, 10.0, 0.0)
#         tremor = st.number_input("Tremor", 0.0, 10.0, 0.0)

#     if st.button("🧠 Predict"):

#         data = pd.DataFrame([{
#             "Age": age,
#             "UPDRS III": updrs,
#             "Disease Duration": duration,
#             "Speech Score": speech,
#             "Facial Expression": facial,
#             "Tremor (Head)": tremor,
#             "Levodopa (mg/day)": levodopa
#         }])

#         stage = model.predict(data)[0]

#         # 🔥 HYBRID CORRECTION
#         if updrs > 50:
#             stage = max(stage, 3.5)
#         elif updrs > 30:
#             stage = max(stage, 2.5)

#         # Classification
#         if stage < 1:
#             status = "Healthy"
#             severity = "None"
#         elif stage < 2:
#             status = "Parkinson's"
#             severity = "Stage 1 (Mild)"
#         elif stage < 3:
#             status = "Parkinson's"
#             severity = "Stage 2 (Moderate)"
#         elif stage < 4:
#             status = "Parkinson's"
#             severity = "Stage 3 (Advanced)"
#         else:
#             status = "Parkinson's"
#             severity = "Stage 4-5 (Severe)"

#         st.markdown(f"""
#         <div class="card">
#         <h3>Result</h3>
#         <b>Status:</b> {status}<br>
#         <b>Stage:</b> {round(stage,2)}<br>
#         <b>Severity:</b> {severity}
#         </div>
#         """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Parkinson AI", layout="wide")

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_app():
    st.session_state.page = "app"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* HERO (centered, compact) */
.hero-wrap {
    display:flex;
    justify-content:center;
    align-items:center;
    padding: 60px 20px 30px 20px;
}
.hero {
    text-align:center;
    max-width: 900px;
}
.title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 18px;
    margin-top: 10px;
    margin-bottom: 20px;
    color: #dcdcdc;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border-radius: 28px;
    font-size: 16px;
    padding: 10px 22px;
    transition: 0.25s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 16px rgba(255,80,120,0.5);
}

/* CARD */
.card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 12px;
    margin-top: 10px;
}

/* FEATURE CARDS */
.feature-card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ================= LANDING =================
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero">
            <div class="title">🧠 Parkinson AI System</div>
            <div class="subtitle">
                Predict disease stage using clinical indicators in seconds
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered CTA
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🚀 Start Analysis", use_container_width=True):
            go_app()

    st.caption("AI-powered clinical support system • Multi-stage (0–5) prediction")

    # Features row
    st.markdown("### 🚀 Key Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-card"><h4>⚡ Fast</h4>Instant predictions</div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><h4>🧠 AI + Clinical</h4>Hybrid logic</div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><h4>📊 Insightful</h4>Charts & explanations</div>', unsafe_allow_html=True)

    st.stop()

# ================= MAIN =================
if st.session_state.page == "app":

    if st.button("⬅ Back"):
        go_home()

    st.title("🧪 Parkinson Prediction Dashboard")
    st.caption("Analyze individual patients or entire datasets")

    # ---------------- LOAD MODEL ----------------
    try:
        model = joblib.load("parkinson_model.pkl")
        features = joblib.load("features.pkl")
    except:
        st.error("❌ Model files not found. Run train_model.py first.")
        st.stop()

    # ---------------- TABS ----------------
    tab1, tab2 = st.tabs(["🧑 Individual Prediction", "📂 Dataset Analysis"])

    # ================= INDIVIDUAL =================
    with tab1:
        st.subheader("Patient Input")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 30)
            duration = st.number_input("Disease Duration", 0.0, 50.0, 0.0)
            levodopa = st.number_input("Levodopa (mg/day)", 0.0, 2000.0, 0.0)
        with col2:
            updrs = st.number_input("UPDRS III", 0.0, 200.0, 0.0)
            speech = st.number_input("Speech Score", 0.0, 10.0, 0.0)
            facial = st.number_input("Facial Expression", 0.0, 10.0, 0.0)
            tremor = st.number_input("Tremor (Head)", 0.0, 10.0, 0.0)

        if st.button("🧠 Predict"):
            data = pd.DataFrame([{
                "Age": age,
                "UPDRS III": updrs,
                "Disease Duration": duration,
                "Speech Score": speech,
                "Facial Expression": facial,
                "Tremor (Head)": tremor,
                "Levodopa (mg/day)": levodopa
            }])

            # ⏳ Loading
            with st.spinner("Analyzing patient data..."):
                stage = model.predict(data)[0]

            # 🔧 Hybrid correction
            if updrs > 50:
                stage = max(stage, 3.5)
            elif updrs > 30:
                stage = max(stage, 2.5)

            # 🎨 Color + severity
            if stage < 1:
                severity = "Healthy"
                color = "#00ff9f"
            elif stage < 2:
                severity = "Stage 1 (Mild)"
                color = "#9be564"
            elif stage < 3:
                severity = "Stage 2 (Moderate)"
                color = "#ffd166"
            elif stage < 4:
                severity = "Stage 3 (Advanced)"
                color = "#ff8c42"
            else:
                severity = "Stage 4–5 (Severe)"
                color = "#ff4b4b"

            # 📊 Hierarchy
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Stage", round(stage, 2))
            with m2:
                st.metric("Severity", severity)

            st.markdown(f"""
            <div class="card">
                <h3 style="color:{color}; margin:0;">{severity}</h3>
                <p style="margin:6px 0 0 0;">Stage ≈ {round(stage,2)}</p>
            </div>
            """, unsafe_allow_html=True)

            # 📊 Visual insight
            st.subheader("Stage Visualization")
            chart_df = pd.DataFrame({"Stage":[stage]})
            st.bar_chart(chart_df)

            # 🧠 Dynamic explainability
            st.subheader("🔍 What Influenced Prediction")
            influences = []
            if updrs > 30:
                influences.append("High UPDRS III indicates significant motor impairment")
            if tremor > 2:
                influences.append("Tremor contributes to progression")
            if facial > 2:
                influences.append("Reduced facial expression suggests motor symptoms")
            if speech > 2:
                influences.append("Speech impairment is a Parkinson’s indicator")
            if duration > 5:
                influences.append("Long disease duration increases stage severity")
            if levodopa > 500:
                influences.append("Higher medication dosage suggests advanced condition")

            if not influences:
                influences.append("All indicators are low → likely healthy or early-stage")

            for item in influences:
                st.write("•", item)

            st.caption("Input summary")
            st.json(data.to_dict())

    # ================= DATASET =================
    with tab2:
        st.subheader("Upload Dataset")
        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file, header=1)
            df.columns = df.columns.str.strip()
            st.write("Preview")
            st.write(df.head())

            # Rename to match training
            df = df.rename(columns={
                "Age (years)": "Age",
                "UPDRS III total (-)": "UPDRS III",
                "Duration of disease from first symptoms (years)": "Disease Duration",
                "18. Speech": "Speech Score",
                "19. Facial Expression": "Facial Expression",
                "20. Tremor at Rest - head": "Tremor (Head)",
                "Levodopa equivalent (mg/day)": "Levodopa (mg/day)"
            })

            # Clean
            df_model = df[features].replace("-", pd.NA)
            df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

            if st.button("📊 Predict Dataset"):
                with st.spinner("Processing dataset..."):
                    stages = model.predict(df_model)

                result_df = pd.DataFrame({
                    "Patient": range(1, len(stages)+1),
                    "Age": df_model["Age"],
                    "UPDRS III": df_model["UPDRS III"],
                    "Stage": stages.round(2)
                })

                st.dataframe(result_df)

                st.subheader("Stage Distribution")
                st.bar_chart(result_df["Stage"])
