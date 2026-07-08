import json
import time
import math
import random
from kafka import KafkaProducer

# --- Enterprise Configuration ---
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'cnc_spindle_telemetry'
SAMPLING_RATE_HZ = 50
SLEEP_INTERVAL =1.0 / SAMPLING_RATE_HZ  # 0.02 seconds for exactly 50Hz

# Initialize Producer with strict latency-optimized boundaries
try:
    print(f"Connecting to Kafka broker at {KAFKA_BROKER}...")
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        linger_ms=5,  # Force sub-5ms packet dispatch
        batch_size=16384,
        acks=1,
        bootstrap_timeout_ms=2000,  # type: ignore # Fail fast if broker is unavailable
    )
    print("Kafka connection established successfully.")
except Exception as e:
    print(f"\n[WARNING] Could not connect to Kafka broker: {e}")
    print("Falling back to MOCK PRODUCER (Dry-run mode)...")
    
    class MockProducer:
        def __init__(self):
            self.count = 0
            
        def send(self, topic, value):
            self.count += 1
            if self.count % 50 == 0:
                vib_val = value.get('vibration_window', [0])[-1]
                temp_val = value.get('temperature_window', [0])[-1]
                print(f"[MOCK-SEND] Topic: {topic} | Step: {self.count} | Vibration: {vib_val:.4f} | Temp: {temp_val:.4f}")
                
        def flush(self):
            pass
            
        def close(self):
            pass
            
    producer = MockProducer()


def simulate_spindle_telemetry(t, degradation_factor=0.0):
    """
    Simulates mechanical physics: Cyclical baseline, exponential decay, and Gaussian noise.
    """
    # Cyclical baseline (Normal Operation)
    base_vibration = math.sin(t * 0.1) * 2.0
    base_temp = 65.0 + math.cos(t * 0.05) * 5.0
    
    # Exponential degradation (Catastrophic Failure approaching)
    vib_spike = math.exp(degradation_factor) - 1
    temp_drift = math.exp(degradation_factor * 0.5) - 1
    
    # Gaussian noise (Sensor inaccuracy)
    vib_noise = random.gauss(0, 0.5)
    temp_noise = random.gauss(0, 1.0)
    
    return {
        "timestamp": time.time(),
        "vibration": round(base_vibration + vib_spike + vib_noise, 4),
        "temperature": round(base_temp + temp_drift + temp_noise, 4)
    }


def run_telemetry_stream(topic=TOPIC_NAME):
    from collections import deque
    
    print(f"Starting 50Hz Telemetry Stream on topic: {topic}...")
    t = 0.0
    degradation = 0.0
    
    vib_window = deque(maxlen=50)
    temp_window = deque(maxlen=50)
    
    try:
        while True:
            # Generate current machine state
            point = simulate_spindle_telemetry(t, degradation)
            
            vib_window.append(point["vibration"])
            temp_window.append(point["temperature"])
            
            # Dispatch to Kafka only when we have a full 50-sample window
            if len(vib_window) == 50:
                payload = {
                    "timestamp": point["timestamp"],
                    "vibration_window": list(vib_window),
                    "temperature_window": list(temp_window)
                }
                producer.send(topic, value=payload)
            
            # Step time forward
            t += 1.0
            # Slowly introduce exponential failure over time
            if t > 500:  
                degradation += 0.005 
                
            # Sleep for 0.02 seconds to maintain exactly 50Hz sampling rate
            time.sleep(SLEEP_INTERVAL)
            
            # Periodically print progress status
            step_int = int(t)
            if step_int % 500 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Stream Active: {step_int} packets transmitted.")
            
    except KeyboardInterrupt:
        print("\nTelemetry stream halted.")
    finally:
        producer.flush()
        producer.close()
        print("Kafka producer resources cleaned up.")


if __name__ == "__main__":
    run_telemetry_stream()
