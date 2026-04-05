# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(page_title="Parkinson Detection", layout="wide")

# # ---------------- CUSTOM CSS ----------------
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
#     color: white;
# }

# .title {
#     text-align: center;
#     font-size: 42px;
#     font-weight: bold;
#     margin-bottom: 10px;
# }

# .card {
#     background: rgba(255,255,255,0.08);
#     padding: 20px;
#     border-radius: 15px;
#     box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
# }

# .healthy {
#     color: #00ff9f;
#     font-weight: bold;
# }

# .parkinson {
#     color: #ff4b4b;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- TITLE ----------------
# st.markdown("<div class='title'>🧠 Parkinson Detection System</div>", unsafe_allow_html=True)
# st.markdown("---")

# # ---------------- LOAD MODELS ----------------
# clf = joblib.load("parkinson_classifier.pkl")
# reg = joblib.load("parkinson_stage_model.pkl")
# features = joblib.load("features.pkl")

# # ---------------- LAYOUT ----------------
# col1, col2 = st.columns([2, 1])

# # ================= LEFT SIDE =================
# with col1:

#     # ===== CSV UPLOAD =====
#     st.markdown("### 📂 Upload Patient Dataset")
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file:
#         df = pd.read_csv(uploaded_file, header=1)
#         df.columns = df.columns.str.strip()

#         st.markdown("### 🔍 Data Preview")
#         st.dataframe(df.head())

#         df = df.select_dtypes(include=['number'])
#         df = df.reindex(columns=features, fill_value=0)

#         if st.button("🚀 Run Prediction"):

#             pred_status = clf.predict(df)
#             pred_stage = reg.predict(df)

#             st.markdown("## 📊 Results")

#             for i in range(len(pred_status)):

#                 if pred_status[i] == 1:
#                     result = "Parkinson's"
#                     css_class = "parkinson"
#                     stage = round(pred_stage[i], 2)

#                     if stage <= 2:
#                         severity = "Mild"
#                         progress = 30
#                     elif stage <= 3:
#                         severity = "Moderate"
#                         progress = 60
#                     else:
#                         severity = "Severe"
#                         progress = 90
#                 else:
#                     result = "Healthy"
#                     css_class = "healthy"
#                     stage = 0
#                     severity = "None"
#                     progress = 10

#                 st.markdown(f"""
#                 <div class="card">
#                     <h4>Patient {i+1}</h4>
#                     <p>Status: <span class="{css_class}">{result}</span></p>
#                     <p>Stage: {stage}</p>
#                     <p>Severity: {severity}</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.progress(progress)

#     # ===== MANUAL INPUT =====
#     st.markdown("---")
#     st.markdown("## 🧑‍⚕️ Manual Patient Entry")

#     with st.container():

#         colA, colB = st.columns(2)

#         with colA:
#             age = st.number_input("Age (years)", 0, 120, 60)
#             duration = st.number_input("Disease Duration (years)", 0.0, 50.0, 1.0)
#             levodopa = st.number_input("Levodopa (mg/day)", 0.0, 2000.0, 0.0)
#             clonazepam = st.number_input("Clonazepam (mg/day)", 0.0, 10.0, 0.0)

#         with colB:
#             updrs = st.number_input("UPDRS III Score", 0.0, 200.0, 10.0)
#             speech = st.number_input("Speech Score", 0.0, 10.0, 0.0)
#             facial = st.number_input("Facial Expression", 0.0, 10.0, 0.0)
#             tremor = st.number_input("Tremor Score", 0.0, 10.0, 0.0)

#         if st.button("🧠 Predict for Single Patient"):

#             input_dict = {
#                 "Age (years)": age,
#                 "Duration of disease from first symptoms (years)": duration,
#                 "Levodopa equivalent (mg/day)": levodopa,
#                 "Clonazepam (mg/day)": clonazepam,
#                 "UPDRS III total (-)": updrs,
#                 "18. Speech": speech,
#                 "19. Facial Expression": facial,
#                 "20. Tremor at Rest - head": tremor
#             }

#             input_df = pd.DataFrame([input_dict])
#             input_df = input_df.select_dtypes(include=['number'])
#             input_df = input_df.reindex(columns=features, fill_value=0)

#             pred_status = clf.predict(input_df)[0]
#             pred_stage = reg.predict(input_df)[0]

#             if pred_status == 1:
#                 result = "Parkinson's"
#                 css_class = "parkinson"
#                 stage = round(pred_stage, 2)

#                 if stage <= 2:
#                     severity = "Mild"
#                     progress = 30
#                 elif stage <= 3:
#                     severity = "Moderate"
#                     progress = 60
#                 else:
#                     severity = "Severe"
#                     progress = 90
#             else:
#                 result = "Healthy"
#                 css_class = "healthy"
#                 stage = 0
#                 severity = "None"
#                 progress = 10

#             st.markdown(f"""
#             <div class="card">
#                 <h3>🧾 Prediction Result</h3>
#                 <p>Status: <span class="{css_class}">{result}</span></p>
#                 <p>Stage: {stage}</p>
#                 <p>Severity: {severity}</p>
#             </div>
#             """, unsafe_allow_html=True)

#             st.progress(progress)

# # ================= RIGHT SIDE =================
# with col2:

#     st.markdown("### 🧠 About Model")
#     st.markdown("""
#     <div class="card">
#     ✔ Detects Parkinson’s Disease  
#     ✔ Predicts Stage (Hoehn & Yahr)  
#     ✔ Machine Learning Model  

#     <br>

#     <b>Severity Levels:</b><br>
#     🟢 Mild (0–2)<br>
#     🟡 Moderate (2–3)<br>
#     🔴 Severe (3+)
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("### ⚙️ Instructions")
#     st.markdown("""
#     <div class="card">
#     1. Upload CSV file  
#     2. Click Predict  
#     3. Or enter patient manually  
#     4. View results instantly  
#     </div>
#     """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Parkinson Detection", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
}
.healthy { color: #00ff9f; }
.parkinson { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Parkinson Detection System</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD ----------------
clf = joblib.load("parkinson_classifier.pkl")
reg = joblib.load("parkinson_stage_model.pkl")
features = joblib.load("features.pkl")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2,1])

# ================= LEFT =================
with col1:

    # ===== CSV =====
    st.subheader("📂 Upload Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file, header=1)
        df.columns = df.columns.str.strip()

        st.dataframe(df.head())

        df = df.select_dtypes(include=['number'])
        df = df.reindex(columns=features, fill_value=0)

        if st.button("Predict Dataset"):
            pred = clf.predict(df)
            stage = reg.predict(df)

            for i in range(len(pred)):
                if pred[i] == 1:
                    result = "Parkinson's"
                    css = "parkinson"
                    stg = round(stage[i],2)

                    if stg <= 2:
                        sev = "Mild"
                    elif stg <= 3:
                        sev = "Moderate"
                    else:
                        sev = "Severe"
                else:
                    result = "Healthy"
                    css = "healthy"
                    stg = 0
                    sev = "None"

                st.markdown(f"""
                <div class="card">
                <b>Patient {i+1}</b><br>
                Status: <span class="{css}">{result}</span><br>
                Stage: {stg}<br>
                Severity: {sev}
                </div>
                """, unsafe_allow_html=True)

    # ===== MANUAL INPUT =====
    st.markdown("---")
    st.subheader("🧑‍⚕️ Manual Patient Entry (Important Fields Only)")

    colA, colB = st.columns(2)

    with colA:
        age = st.number_input("Age", 0, 120, 60)
        duration = st.number_input("Disease Duration", 0.0, 50.0, 1.0)
        levodopa = st.number_input("Levodopa (mg/day)", 0.0, 2000.0, 0.0)

    with colB:
        updrs = st.number_input("UPDRS III", 0.0, 200.0, 10.0)
        speech = st.number_input("Speech Score", 0.0, 10.0, 0.0)
        facial = st.number_input("Facial Expression", 0.0, 10.0, 0.0)
        tremor = st.number_input("Tremor (Head)", 0.0, 10.0, 0.0)

    if st.button("🧠 Predict Single Patient"):

        data = {
            "Age (years)": age,
            "Duration of disease from first symptoms (years)": duration,
            "Levodopa equivalent (mg/day)": levodopa,
            "UPDRS III total (-)": updrs,
            "18. Speech": speech,
            "19. Facial Expression": facial,
            "20. Tremor at Rest - head": tremor
        }

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=features, fill_value=0)

        pred = clf.predict(input_df)[0]
        stage = reg.predict(input_df)[0]

        if pred == 1:
            result = "Parkinson's"
            css = "parkinson"
            stg = round(stage,2)

            if stg <= 2:
                sev = "Mild"
            elif stg <= 3:
                sev = "Moderate"
            else:
                sev = "Severe"
        else:
            result = "Healthy"
            css = "healthy"
            stg = 0
            sev = "None"

        st.markdown(f"""
        <div class="card">
        <h3>Result</h3>
        Status: <span class="{css}">{result}</span><br>
        Stage: {stg}<br>
        Severity: {sev}
        </div>
        """, unsafe_allow_html=True)

# ================= RIGHT =================
with col2:
    st.subheader("ℹ️ Info")
    st.markdown("""
    <div class="card">
    ✔ Uses ML model  
    ✔ Predicts Parkinson’s  
    ✔ Estimates severity  

    <br>
    🟢 Mild (0–2)  
    🟡 Moderate (2–3)  
    🔴 Severe (3+)  
    </div>
    """, unsafe_allow_html=True)