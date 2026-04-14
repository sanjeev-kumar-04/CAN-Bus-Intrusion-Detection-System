import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import plotly.graph_objects as go
from collections import deque

from models.model import CANIDS
from utils.preprocessing import hex_to_int, parse_timestamp
import joblib

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="CAN IDS Dashboard", layout="wide")
st.title("🚗 CAN Bus Intrusion Detection System")

# ----------------------------
# LOAD MODEL + SCALER
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CANIDS(input_dim=10).to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler = joblib.load("scaler.pkl")

# ----------------------------
# DATA BUFFER
# ----------------------------
SEQ_LEN = 5
buffer = deque(maxlen=SEQ_LEN)

# ----------------------------
# LOAD DATA
# ----------------------------
columns = ['Timestamp', 'CAN_ID', 'DLC'] + [f'DATA{i}' for i in range(8)] + ['Flag']

df = pd.read_csv("data/DoS_dataset.csv", names=columns, nrows=5000)
df = df.sample(frac=1).reset_index(drop=True)

# ----------------------------
# SESSION STATE
# ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "stop" not in st.session_state:
    st.session_state.stop = False
if "index" not in st.session_state:
    st.session_state.index = 0
if "attack_count" not in st.session_state:
    st.session_state.attack_count = 0
if "logs" not in st.session_state:
    st.session_state.logs = []

# ----------------------------
# PROCESS ROW
# ----------------------------
def process_row(row):
    data = [hex_to_int(row[f'DATA{i}']) for i in range(8)]
    features = data + [parse_timestamp(row['Timestamp']), hex_to_int(row['DLC'])]
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features = scaler.transform(features)
    features = np.clip(features, -3, 3)
    return features.flatten()

# ----------------------------
# SIDEBAR STATS (FIXED)
# ----------------------------
with st.sidebar:
    st.header("📊 System Stats")
    attack_stat = st.empty()
    total_stat = st.empty()
    normal_stat = st.empty()

# ----------------------------
# CUSTOM MESSAGE TESTER
# ----------------------------
st.subheader("Custom CAN Message Tester")

custom_timestamp = st.text_input("Timestamp", "1234")
custom_can_id = st.text_input("CAN_ID", "0x2F4")
custom_dlc = st.text_input("DLC", "8")

custom_data = [st.text_input(f"DATA{i}", "FF") for i in range(8)]

custom_output = st.empty()

if st.button("🚨 Test Custom Message"):
    row = {
        "Timestamp": custom_timestamp,
        "CAN_ID": custom_can_id,
        "DLC": custom_dlc,
    }
    for i in range(8):
        row[f"DATA{i}"] = custom_data[i]

    buffer.append(process_row(row))

    if len(buffer) < SEQ_LEN:
        padded = list(buffer) + [list(buffer)[0]] * (SEQ_LEN - len(buffer))
    else:
        padded = list(buffer)

    X = torch.FloatTensor(np.array(padded)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X).item()

    prediction = "🚨 ATTACK" if output > 0.5 else "✅ NORMAL"

    custom_output.markdown("### 🔍 Result")
    custom_output.write(f"Confidence: {output:.4f}")
    custom_output.write(f"Prediction: {prediction}")

    if output > 0.5:
        custom_output.error("🚨 ATTACK DETECTED")
    else:
        custom_output.success("✅ NORMAL TRAFFIC")

# ----------------------------
# START / STOP
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Simulation"):
        st.session_state.running = True

with col2:
    if st.button("🛑 Stop Simulation"):
        st.session_state.running = False


# ----------------------------
# LIVE DASHBOARD PLACEHOLDERS
# ----------------------------
st.markdown("---")
st.subheader("📡 Live Simulation Dashboard")

table_box = st.empty()
chart_box = st.empty()
status_box = st.empty()


# ----------------------------
# SINGLE STEP EXECUTION (IMPORTANT FIX)
# ----------------------------
if st.session_state.index < len(df):

    if st.session_state.running:
        row = df.iloc[st.session_state.index]
        st.session_state.index += 1

        buffer.append(process_row(row))

        if len(buffer) < SEQ_LEN:
            padded = list(buffer) + [list(buffer)[0]] * (SEQ_LEN - len(buffer))
        else:
            padded = list(buffer)

        X = torch.FloatTensor(np.array(padded)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(X).item()

        prediction = "🚨 ATTACK" if output > 0.5 else "✅ NORMAL"

        if output > 0.5:
            st.session_state.attack_count += 1

        st.session_state.logs.append({
            "msg": st.session_state.index,
            "type": row["Flag"],
            "prediction": prediction,
            "confidence": output
        })

    # ----------------------------
    # STATUS UPDATE
    # ----------------------------
    status_box.info(f"Processing packet #{st.session_state.index}")

    # ----------------------------
    # SIDEBAR STATS
    # ----------------------------
    attack_stat.metric("🚨 Attacks Detected", st.session_state.attack_count)
    total_stat.metric("📡 Total Messages", st.session_state.index)
    normal_stat.metric(
        "✅ Normal Messages",
        st.session_state.index - st.session_state.attack_count
    )

    # ----------------------------
    # LIVE TABLE
    # ----------------------------
    table_box.dataframe(
        pd.DataFrame(st.session_state.logs[-10:]),
        width="stretch"
    )

    # ----------------------------
    # LIVE GRAPH
    # ----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[x["confidence"] for x in st.session_state.logs[-20:]],
        mode="lines+markers",
        name="Confidence"
    ))

    fig.update_layout(
        title="Live Attack Confidence",
        yaxis=dict(range=[0, 1])
    )

    chart_box.plotly_chart(fig, width="stretch")

    if st.session_state.running:
    # process one step
        time.sleep(0.3)
        st.rerun()
    