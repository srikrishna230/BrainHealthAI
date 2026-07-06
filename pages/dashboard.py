import streamlit as st
import pandas as pd

def dashboard():

    st.title("🧠 EEG Brain Health Monitoring System")
    st.caption("TensorFlow Neural Network | Research Tool")

    st.divider()

    left, right = st.columns([1,4])

    with left:

        st.subheader("Upload & Controls")

        uploaded_file = st.file_uploader(
            "Upload EEG CSV",
            type=["csv"]
        )

        st.markdown("---")

        epochs = st.number_input(
            "Epochs",
            value=40,
            min_value=1
        )

        batch_size = st.number_input(
            "Batch Size",
            value=32,
            min_value=1
        )

        learning_rate = st.number_input(
            "Learning Rate",
            value=0.001,
            format="%.4f"
        )

        retrain = st.checkbox("Retrain Model")

        train = st.button("Train Model")

        predict = st.button("Predict")

        report = st.button("Generate PDF")

    with right:

        st.subheader("EEG Waveform")

        st.info("Upload a CSV file to visualize EEG signals.")

        st.markdown("---")

        c1,c2 = st.columns(2)

        with c1:

            st.subheader("Prediction")

            st.metric(
                label="Mental State",
                value="Waiting..."
            )

        with c2:

            st.subheader("Confidence")

            st.metric(
                label="Probability",
                value="0%"
            )

        st.markdown("---")

        st.subheader("Probability Distribution")

        st.info("Probability graph will appear here.")

        st.markdown("---")

        st.subheader("Mental State Distribution")

        st.info("Pie chart will appear here.")

        st.markdown("---")

        st.subheader("Extracted Features")

        st.dataframe(pd.DataFrame())

        st.markdown("---")

        st.subheader("Recommendation")

        st.success("Recommendations will appear after prediction.")