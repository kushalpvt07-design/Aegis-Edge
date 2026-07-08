import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import plotly.graph_objects as go
import time

# ==========================================
# 1. DASHBOARD CONFIGURATION
# ==========================================
st.set_page_config(page_title="CNC Edge AI Monitor", layout="wide", initial_sidebar_state="collapsed")
st.title("🏭 Real-Time Edge AI Machine Monitor")
st.markdown("Live sub-100ms telemetry from Kafka & Spark ONNX Inference")

# ==========================================
# 2. INITIALIZE KAFKA CONSUMER
# ==========================================
@st.cache_resource
def create_consumer():
    return KafkaConsumer(
        'inference_alerts',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )

consumer = create_consumer()

# ==========================================
# 3. UI PLACEHOLDERS
# ==========================================
alert_box = st.empty()
col1, col2, col3 = st.columns(3)
metric_vib = col1.empty()
metric_temp = col2.empty()
metric_risk = col3.empty()
chart_placeholder = st.empty()

# Initialize data history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['vibration', 'temperature', 'anomaly_score'])

# ==========================================
# 4. REAL-TIME EVENT LOOP
# ==========================================
for message in consumer:
    data = message.value
    
    # Extract the most recent value from the 50-step arrays
    current_vib = data.get('vibration_window', [0])[-1]
    current_temp = data.get('temperature_window', [0])[-1]
    raw_score = data.get('anomaly_score', 0)
    risk_score = raw_score * 100
    
    # Determine Status dynamically
    status = 'TRIGGERED' if risk_score > 85.0 else 'SAFE'

    # Append to history
    new_row = pd.DataFrame([{
        'vibration': current_vib,
        'temperature': current_temp,
        'anomaly_score': raw_score
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(100)
    df = st.session_state.history

    # Update Metrics
    metric_vib.metric("Vibration (Hz)", f"{current_vib:.2f}")
    metric_temp.metric("Temperature (°C)", f"{current_temp:.2f}")
    metric_risk.metric("AI Risk Score", f"{risk_score:.1f}%")

    # Update Alert Banner
    if status == 'TRIGGERED':
        alert_box.error(f"🚨 CATASTROPHIC FAILURE PREDICTED. KILL SWITCH ENGAGED! Risk: {risk_score:.1f}%", icon="🚨")
    else:
        alert_box.success("✅ Machine Operating Normally.", icon="⚙️")

    # Draw Live Charts using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['vibration'], mode='lines', name='Vibration', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(y=df['anomaly_score']*100, mode='lines', name='Risk %', line=dict(color='red')))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400, xaxis_title="Time (Ticks)", yaxis_title="Magnitude")
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    time.sleep(0.05)
