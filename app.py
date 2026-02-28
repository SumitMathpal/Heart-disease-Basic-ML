import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="AI Heart Risk Analyzer",
    page_icon="❤️",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.glass {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
.big-font {
    font-size:22px !important;
    font-weight:bold;
}
.footer {
    text-align:center;
    margin-top:40px;
    font-size:14px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("Logistic_reg.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧠 AI Health Panel")
    st.markdown("### About Model")
    st.info("Model: Logistic Regression")
    st.markdown("Dataset: UCI Heart Dataset")
    st.markdown("Accuracy: ~85-90%")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by **Sumit Mathpal**")

# ================= MAIN TITLE =================
st.title("❤️ AI Powered Heart Risk Analyzer")
st.markdown("#### Enter Patient Clinical Data")

# ================= INPUT SECTION =================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])

    with col2:
        resting_bp = st.number_input("Resting BP", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting BS >120", [0, 1])

    with col3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max HR", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICT =================
if st.button("🔍 Analyze Risk"):

    with st.spinner("Analyzing Patient Data..."):
        time.sleep(2)

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    risk_percent = round(probability * 100, 2)

    # ================= RESULT CARD =================
    if prediction == 1:
        st.error(f"⚠ HIGH RISK DETECTED ({risk_percent}%)")
        st.progress(int(risk_percent))
        st.markdown("### 🔴 Recommendation:")
        st.write("- Immediate cardiology consultation advised.")
        st.write("- Reduce cholesterol intake.")
        st.write("- Start moderate supervised exercise.")
        st.write("- Avoid smoking & alcohol.")

    else:
        st.success(f"✅ LOW RISK ({risk_percent}%)")
        st.progress(int(risk_percent))
        st.markdown("### 🟢 Recommendation:")
        st.write("- Maintain healthy lifestyle.")
        st.write("- Regular exercise 30 mins daily.")
        st.write("- Balanced diet & stress management.")

    # ================= CONFIDENCE =================
    confidence = round((1 - abs(0.5 - probability)) * 100, 2)
    st.markdown(f"### 🧠 Model Confidence: {confidence}%")

# ================= FOOTER =================
st.markdown("""
<div class="footer">
AI Healthcare System © 2026 | Built with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)