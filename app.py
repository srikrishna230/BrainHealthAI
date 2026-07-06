import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from prediction.predict import predict
from reports.generate_report import generate_pdf

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Brain Health Monitoring",
    page_icon="",
    layout="wide"
)

# -----------------------
# Load CSS
# -----------------------
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Brain Health AI")

uploaded_file = st.sidebar.file_uploader(
    "Upload EEG CSV",
    type=["csv"]
)

# -----------------------
# Read Uploaded CSV
# -----------------------

df = None
prediction_result = None

if uploaded_file is not None:

    st.write("File uploaded:", uploaded_file.name)

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        st.write("Columns:", df.columns.tolist())
        st.write(df.head())

        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

        if st.sidebar.button("Predict", key="predict_btn"):

            st.session_state.prediction_result = predict(df)
            st.write(st.session_state.prediction_result)
            st.success("Prediction completed!")

        st.success("CSV Loaded Successfully")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
    st.sidebar.success("EEG File Loaded Successfully")

st.sidebar.divider()

epochs = st.sidebar.slider("Epochs", 10, 200, 50)

batch = st.sidebar.selectbox(
    "Batch Size",
    [8,16,32,64,128]
)

lr = st.sidebar.selectbox(
    "Learning Rate",
    [0.1,0.01,0.001,0.0001],
    index=2
)

import subprocess
import sys

if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        result = subprocess.run(
            [sys.executable, "training/train_model.py"],
            capture_output=True,
            text=True
        )

    if result.returncode == 0:
        st.success("✅ Model trained successfully!")
        st.text(result.stdout)
    else:
        st.error("❌ Training failed")
        st.text(result.stderr)

if st.sidebar.button("Generate PDF", key="pdf_btn"):

    if st.session_state.prediction_result is not None:

        generate_pdf(st.session_state.prediction_result)

        st.success("PDF Generated Successfully!")

    else:

        st.warning("Please click Predict first.")

        
if df is not None:
    print(df.head())
    print(df.columns)
else:
    print("df is None")

# -----------------------
# Header
# -----------------------
st.title("Brain Health Monitoring System")

# -----------------------
# Dataset Information
# -----------------------

if df is not None:

    st.subheader("Uploaded EEG Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(), use_container_width=True)

    st.divider()

st.caption("AI Powered EEG Analysis using TensorFlow")

st.divider()


# -----------------------
# Top Cards
# -----------------------
c1, c2, c3 = st.columns(3)

c4, c5, c6 = st.columns(3)

result = st.session_state.get("prediction_result")

if result is not None:

    c1.metric("Mental State", result["Prediction"])
    c2.metric("Confidence", f"{result['Confidence']:.1f}%")
    c3.metric("Stress", f"{result['Stress']:.1f}%")

    c4.metric("Anxiety", f"{result['Anxiety']:.1f}%")
    c5.metric("Attention", f"{result['Attention']:.1f}%")
    c6.metric("Happiness", f"{result['Happiness']:.1f}%")

else:

    c1.metric("Mental State", "--")
    c2.metric("Confidence", "--")
    c3.metric("Stress", "--")

    c4.metric("Anxiety", "--")
    c5.metric("Attention", "--")
    c6.metric("Happiness", "--")
st.divider()


if result is not None:

    st.subheader("Brain Health Indicators")

    st.write("Attention")
    st.progress(int(result["Attention"]))

    st.write("Happiness")
    st.progress(int(result["Happiness"]))

    st.write("Anxiety")
    st.progress(int(result["Anxiety"]))

    st.write("Stress")
    st.progress(int(result["Stress"]))

    st.write("Brain Health")
    st.progress(int(result["BrainHealth"]))


# -----------------------
# EEG Wave
# -----------------------
left,right=st.columns([2,1])

with left:

    st.subheader("EEG Waveform")

    x=np.linspace(0,10,500)

    y=np.sin(10*x)+0.2*np.random.randn(500)

    fig = go.Figure()

    if df is not None:

        fig.add_trace(
            go.Scatter(
                x=df.iloc[:,0],
                y=df.iloc[:,1],
                mode="lines",
                name="EEG Signal"
            )
        )

    else:
  
        x=np.linspace(0,10,500)

        y=np.sin(10*x)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Sample EEG"
            )
        ) 

    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    st.subheader("Mental State Distribution")

    result = st.session_state.get("prediction_result")

    if result is not None:

        values = result["Probabilities"]

    else:

        values = [33, 33, 34]

    labels = ["Neutral", "Anxiety", "Depression"]

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4
            )
        ]
    )

    fig2.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------
# Probability Distribution
# -----------------------

st.subheader("Probability Distribution")

result = st.session_state.get("prediction_result")

if result is not None:

    probs = result["Probabilities"]

else:

    probs = [33, 33, 34]

prob = pd.DataFrame({

    "State": [
        "Neutral",
        "Anxiety",
        "Depression"
    ],

    "Probability": probs

})

st.bar_chart(prob, x="State", y="Probability")


# -----------------------
# Extracted Features
# -----------------------

st.subheader("Extracted EEG Features")

if result is not None:

    feature_df = pd.DataFrame({

        "Feature": [
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
            "Gamma",
            "Hjorth Activity",
            "Hjorth Mobility",
            "Hjorth Complexity"
        ],

        "Value": [
            round(result.get("Delta", 0), 4),
            round(result.get("Theta", 0), 4),
            round(result.get("Alpha", 0), 4),
            round(result.get("Beta", 0), 4),
            round(result.get("Gamma", 0), 4),
            round(result.get("Activity", 0), 4),
            round(result.get("Mobility", 0), 4),
            round(result.get("Complexity", 0), 4)
        ]

    })

else:

    feature_df = pd.DataFrame({

        "Feature": [
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
            "Gamma",
            "Hjorth Activity",
            "Hjorth Mobility",
            "Hjorth Complexity"
        ],

        "Value": [0,0,0,0,0,0,0,0]

    })

st.dataframe(feature_df, use_container_width=True)


# -----------------------
# AI Recommendation
# -----------------------

st.subheader("AI Recommendation")

if result is not None:

    if result["Stress"] >= 70:

        st.error("🔴 High Stress detected. Relaxation, meditation and adequate sleep are recommended.")

    elif result["Anxiety"] >= 60:

        st.warning("🟠 Elevated Anxiety detected. Consider breathing exercises and stress management.")

    elif result["Depression"] >= 60:

        st.error("🔴 Depression indicators detected. Consider consulting a mental health professional if symptoms persist.")

    elif result["Happiness"] >= 70:

        st.success("😊 Positive emotional state detected. Brain activity appears healthy.")

    else:

        st.info("🟢 Brain activity appears balanced.")

else:

    st.info("Upload an EEG CSV file and click **Predict** to receive an AI analysis.")