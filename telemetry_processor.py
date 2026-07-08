import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import urllib.request
import pyspark
import pandas as pd
import numpy as np
import onnxruntime as ort
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, FloatType

# 1. Global placeholder for the ONNX Runtime Session
_ort_session = None
_input_name = None

# 2. Define the Vectorized Pandas UDF
@pandas_udf(FloatType())
def predict_anomaly_udf(vibration_series: pd.Series, temp_series: pd.Series) -> pd.Series:
    """
    Executes sub-5ms ONNX inference over a batch of 50-step sliding windows.
    Applies Softmax to extract the anomaly risk percentage.
    """
    global _ort_session, _input_name
    if _ort_session is None:
        # Lazily load ONNX session once per Spark executor process (bypasses Driver pickling error)
        import onnxruntime as ort
        _ort_session = ort.InferenceSession("spindle_anomaly_cnn.onnx")
        _input_name = _ort_session.get_inputs()[0].name
        
    # Extract the arrays from the Pandas Series
    vib_data = np.stack(vibration_series.values)
    temp_data = np.stack(temp_series.values)
    
    # Shape becomes (B, 2, 50)
    input_tensor = np.stack([vib_data, temp_data], axis=1).astype(np.float32)
    # Add this line to prevent overflow:
    input_tensor = np.clip(input_tensor, -1e6, 1e6) 
    
    # Run the ONNX C++ Execution Graph -> Returns shape (B, 2) Logits
    logits = _ort_session.run(None, {_input_name: input_tensor})[0]
    
    # Apply Softmax to convert logits to probabilities along the class axis (axis=1)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # stability fix
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Extract the probability for Class 1 (Anomalous)
    anomaly_risk_scores = probabilities[:, 1]
    
    return pd.Series(anomaly_risk_scores)

def setup_hadoop_winutils():
    # Only run on Windows
    if os.name != 'nt':
        return
        
    hadoop_dir = os.path.abspath("hadoop")
    bin_dir = os.path.join(hadoop_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    winutils_path = os.path.join(bin_dir, "winutils.exe")
    hadoop_dll_path = os.path.join(bin_dir, "hadoop.dll")
    
    # Download winutils.exe and hadoop.dll if not present
    base_url = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
    
    if not os.path.exists(winutils_path):
        print("Downloading winutils.exe for Windows compatibility...")
        urllib.request.urlretrieve(f"{base_url}/winutils.exe", winutils_path)
        
    if not os.path.exists(hadoop_dll_path):
        print("Downloading hadoop.dll for Windows compatibility...")
        urllib.request.urlretrieve(f"{base_url}/hadoop.dll", hadoop_dll_path)
        
    os.environ["HADOOP_HOME"] = hadoop_dir
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"HADOOP_HOME configured to: {hadoop_dir}")


def initialize_spark_processor():
    # Configure Windows compatibility if running on Windows
    setup_hadoop_winutils()

    # Define local checkpoint directory to avoid Windows-specific path resolution issues with /tmp
    checkpoint_path = os.path.abspath("spark_checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Initialize a highly tuned Spark session
    # We restrict memory overhead so it doesn't bog down the edge environment.
    # We dynamically include the spark-sql-kafka connector matching the installed PySpark version.
    spark = SparkSession.builder \
        .appName("Spindle_Telemetry_Processor") \
        .config("spark.master", "local[2]") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.streaming.checkpointLocation", checkpoint_path) \
        .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "true") \
        .config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}") \
        .getOrCreate()

    # Suppress the massive wall of INFO logs Spark generates
    spark.sparkContext.setLogLevel("WARN")

    # Define the exact schema of our physics simulator
    telemetry_schema = StructType([
        StructField("timestamp", DoubleType(), True),
        StructField("vibration_window", ArrayType(DoubleType()), True),
        StructField("temperature_window", ArrayType(DoubleType()), True)
    ])

    print("Connecting to Kafka Telemetry Stream...")

    # 1. Read directly from the Kafka Topic
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "cnc_spindle_telemetry") \
        .option("startingOffsets", "latest") \
        .load()

    # 2. Parse the JSON payload
    parsed_df = df.select(
        from_json(col("value").cast("string"), telemetry_schema).alias("data")
    ).select("data.*")

    # 3. Feature Engineering / Windowing Prep
    # Include timestamp so it passes through to the dashboard
    features_df = parsed_df.select("timestamp", "vibration_window", "temperature_window")

    # 4. Pipeline Integration Example
    print("Applying Vectorized ONNX Inference...")
    scored_df = features_df.withColumn(
        "anomaly_score", 
        predict_anomaly_udf(col("vibration_window"), col("temperature_window"))
    )
    
    # Convert the entire processed row into a single JSON payload
    kafka_output_stream = scored_df.selectExpr("to_json(struct(*)) AS value")

    # 5. Write Stream - Micro-batching is more stable on Windows than continuous mode
    print("Executing Micro-batch Processing Engine & Routing to Kafka...")
    query = kafka_output_stream \
        .writeStream \
        .outputMode("append") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "inference_alerts") \
        .option("checkpointLocation", "./spark_checkpoints_alerts") \
        .trigger(processingTime="5 seconds") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    initialize_spark_processor()
