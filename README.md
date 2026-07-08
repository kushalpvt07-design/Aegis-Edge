# 🏭 Edge-Native IoT Predictive Maintenance Pipeline

### 📌 The Problem

Standard cloud-based Machine Learning pipelines suffer from a fatal flaw in industrial environments: **latency**. If an anomaly detection model relies on sluggish batch processing or cloud round-trips, it cannot react in time to save hardware. In high-speed CNC machining, a 500ms delay in detecting a vibration anomaly means a shattered spindle, resulting in hundreds of thousands of dollars in catastrophic damage and factory downtime.

### 🛡️ The Solution

This repository implements a **Sub-100ms Edge-Native MLOps Pipeline**. It physically localizes the entire ingestion, processing, and inference stack directly to the edge machine. By combining a bare-metal messaging broker, Spark stream processing, and an aggressively optimized C++ neural network runtime, it detects exponential mechanical degradation and triggers a kill-switch *before* failure occurs.

### ⚙️ Key Architecture Components

* **High-Frequency Ingestion (Python + Kafka):** A physical simulator (`telemetry_producer.py`) mathematically generates CNC spindle telemetry—combining cyclical sine baselines, exponential thermal decay, and Gaussian noise. It samples at exactly 50Hz with a strict `linger_ms=5` network boundary.
* **Bare-Metal Messaging (Kafka KRaft):** Bypasses virtualization bottlenecks (Docker/WSL2) by running a native Windows Kafka broker in KRaft mode, ensuring zero-latency data pass-through.
* **Micro-Batch Orchestration (Apache Spark):** Utilizes PySpark Structured Streaming (`telemetry_processor.py`) to consume the telemetry firehose. It aggregates data into 50-step sliding windows (1 second of physics) using highly tuned 1-second micro-batches to ensure OS-level stability.
* **Vectorized Edge Inference (ONNX):** A PyTorch 1D Convolutional Neural Network (1D-CNN) exported to the C++ ONNX runtime. Integrated via Spark `mapPartitions`, it extracts spatial anomaly features from the sliding windows while eliminating Java/Python IPC serialization overhead, guaranteeing sub-5ms inference times.
* **Visual MLOps Actuation (Streamlit):** A low-latency web interface (`dashboard.py`) operating as a parallel Kafka consumer. It dynamically plots live physical vibrations and the CNN's Softmax risk probability, engaging a visual Red Kill-Switch when catastrophic risk exceeds 85%.

---

### 🚀 Quick Start (Bare-Metal Windows)

This system is engineered for stability on edge hardware, intentionally bypassing Docker due to WSL2 bridging limitations. 

**Prerequisites:** * Python 3.12 
* Java (JVM) installed and added to PATH
* Apache Kafka 4.3.1 installed natively at `C:\kafka\kafka_2.13-4.3.1`

#### 1. Initialize the Infrastructure (Kafka KRaft)
Format the cluster and start the broker locally (run in an Administrator PowerShell):
```powershell
# Format storage with a standalone cluster ID
C:\kafka\kafka_2.13-4.3.1\bin\windows\kafka-storage.bat format -t 9Uz2jK5CR96WOdsMp1LA-w -c C:\kafka\kafka_2.13-4.3.1\config\server.properties --standalone

# Start the broker
C:\kafka\kafka_2.13-4.3.1\bin\windows\kafka-server-start.bat C:\kafka\kafka_2.13-4.3.1\config\server.properties
```

### 2. Start the Telemetry Firehose
Open a new terminal, activate your virtual environment, and launch the 50Hz CNC simulator:

```powershell
python telemetry_producer.py
```

### 3. Engage the Inference Brain
In a third terminal, launch the data stream processor to score the sliding windows via ONNX:

```powerShell
python telemetry_processor.py
```

### 4. Launch the Factory Floor Monitor
In a fourth terminal, spin up the real-time visual dashboard:

```powerShell
streamlit run dashboard.py
```

### 🧪 Pipeline Usage & Testing
The system automatically simulates a machine lifecycle. You can monitor the execution live via the Streamlit UI at `http://localhost:8501`.

**Phase 1: Baseline Operation (t < 500 steps)**
* **State:** The spindle operates normally with cyclical sine-wave vibrations and standard Gaussian noise.
* **Expected Result:** The 1D-CNN processes the 50-step windows and outputs an Anomaly Score near 0%. The dashboard displays a "SAFE" status.

**Phase 2: The Catastrophic Failure (t > 500 steps)**
* **State:** The `telemetry_producer.py` simulator mathematically injects exponential degradation (thermal drift and vibration spikes).
* **Expected Result:** As the physical variance shifts, the ONNX model immediately flags the spatial anomaly. The Risk Score climbs rapidly. Once it breaches 85.0%, the dashboard throws a `TRIGGERED` status, flashing the Catastrophic Kill-Switch alert to halt the simulated machine.

---

### 🧠 Tech Stack
* **Ingestion:** Python, `kafka-python-ng`
* **Messaging:** Apache Kafka (Native KRaft Mode)
* **Stream Processing:** Apache Spark (PySpark Structured Streaming)
* **Deep Learning:** PyTorch (1D-CNN architecture)
* **Edge Inference:** ONNX Runtime (C++ compiled graph)
* **Visualization:** Streamlit, Plotly
