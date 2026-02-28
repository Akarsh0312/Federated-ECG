
import os
import requests
import subprocess
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# ---- STATE INIT ----
if "server_running" not in st.session_state:
    st.session_state.server_running = False

if "client_a_connected" not in st.session_state:
    st.session_state.client_a_connected = False

if "client_b_connected" not in st.session_state:
    st.session_state.client_b_connected = False


# ---- UI ----
st.title("Federated ECG System")

# -------- SYSTEM STATUS --------
st.subheader("System Status")

if st.button("Start Server (Real)"):
    subprocess.Popen(
        ["python", "../federated/server.py"],
        cwd=os.getcwd()
    )
    st.session_state.server_running = True

st.write(
    "🖥 Server:",
    "RUNNING" if st.session_state.server_running else "STOPPED"
)


if st.button("Connect Client A (Real)"):
    subprocess.Popen(
        ["python", "../federated/client_A.py"],
        cwd=os.getcwd()
    )
    st.session_state.client_a_connected = True

st.write(
    "🧑‍💻 Client A:",
    "CONNECTED" if st.session_state.client_a_connected else "NOT CONNECTED"
)

if st.button("Simulate Client B Connect"):
    st.session_state.client_b_connected = True

st.write(
    "🧑‍💻 Client B:",
    "CONNECTED" if st.session_state.client_b_connected else "NOT CONNECTED"
)

# -------- INFERENCE --------
# ---- CLASS MAPPING ----
CLASS_LABELS = {
    0: "Normal Beat (No Stress)",
    1: "Supraventricular Ectopic Beat",
    2: "Ventricular Ectopic Beat",
    3: "Fusion Beat",
    4: "Unknown / Other Arrhythmia",
    5: "Stress Related Abnormality"
}

STRESS_CLASSES = [1, 2, 3, 4, 5]
# ---- ECG Upload ----
st.subheader("Upload ECG File")

uploaded_file = st.file_uploader("Upload ECG (.csv file)", type=["csv"])


ecg_input = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("ECG CSV Loaded Successfully")    

    # take first row ECG signal (140 values)
    signal = df.iloc[0, :-1].values

    # st.write("### 📈 Original ECG Waveform")

    st.write("### 📈 Live ECG Monitor View")

    fig, ax = plt.subplots(figsize=(8, 3))

    # Black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Green ECG line
    ax.plot(signal, color="#00FF00", linewidth=1.5)

    # Grid like ECG monitor
    ax.grid(color="gray", linestyle="--", linewidth=0.3)

    # Remove axis labels for cleaner monitor look
    ax.set_xticks([])
    ax.set_yticks([])

    st.pyplot(fig)

    # convert 1D signal to 2D image
    signal = np.interp(signal, (signal.min(), signal.max()), (0, 255))
    signal = signal.astype(np.uint8)

    img = cv2.resize(signal.reshape(1, -1), (64, 64))

    ecg_input = img.reshape(1, 64, 64, 1)

    st.success("ECG CSV Processed Successfully")


# ---- INFERENCE ----

st.subheader("Inference")

if st.button("Run Inference (Global Model)"):

    if uploaded_file is None:
        st.error("Please upload ECG file first")
        st.stop()

    MODEL_URL = "http://localhost:8000/get-model/"

    response = requests.get(MODEL_URL)
    with open("downloaded_model.h5", "wb") as f:
        f.write(response.content)

    model = load_model("downloaded_model.h5")

    all_signals = df.iloc[:, :-1].values
    total_beats = len(all_signals)

    processed_images = []

    for row in all_signals:
        signal = row
        signal = np.interp(signal, (signal.min(), signal.max()), (0, 255))
        signal = signal.astype(np.uint8)
        img = cv2.resize(signal.reshape(1, -1), (64, 64))
        processed_images.append(img)

    processed_images = np.array(processed_images)
    processed_images = processed_images.reshape(-1, 64, 64, 1)

    # 🔥 ONE SHOT PREDICTION
    predictions = model.predict(processed_images, verbose=0)

    pred_classes = np.argmax(predictions, axis=1)

    normal_count = np.sum(pred_classes == 0)
    abnormal_count = np.sum(pred_classes != 0)

    abnormal_percent = (abnormal_count / total_beats) * 100

    st.write("## 📊 Full ECG Analysis Report")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Beats", total_beats)
    col2.metric("Normal Beats", normal_count)
    col3.metric("Abnormal Beats", abnormal_count)

    st.write(f"### ⚠️ Overall Arrhythmia Percentage: {abnormal_percent:.2f}%")

    if abnormal_percent < 20:
        st.success("🟢 LOW RISK – Mostly Normal Rhythm")
    elif abnormal_percent < 50:
        st.warning("🟡 MODERATE RISK – Monitor Required")
    else:
        st.error("🔴 HIGH RISK – Clinical Attention Recommended")