import os
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
    Expects input series to contain arrays of length 50.
    """
    global _ort_session, _input_name
    if _ort_session is None:
        # Lazily load ONNX session once per Spark executor process (bypasses Driver pickling error)
        import onnxruntime as ort
        _ort_session = ort.InferenceSession("spindle_anomaly_cnn.onnx")
        _input_name = _ort_session.get_inputs()[0].name
        
    vib_data = np.stack(vibration_series.values)
    temp_data = np.stack(temp_series.values)
    
    # Shape becomes (B, 2, 50)
    input_tensor = np.stack([vib_data, temp_data], axis=1).astype(np.float32)
    ort_outs = _ort_session.run(None, {_input_name: input_tensor})
    
    # Extract the anomaly scores (index 1) and return as a Pandas Series
    # We use [:, 1] instead of flatten() to preserve batch size and extract the correct logit
    predictions = ort_outs[0][:, 1]
    return pd.Series(predictions)

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
    features_df = parsed_df.select("vibration_window", "temperature_window")

    # 4. Pipeline Integration Example
    print("Applying Vectorized ONNX Inference...")
    scored_df = features_df.withColumn(
        "anomaly_score", 
        predict_anomaly_udf(col("vibration_window"), col("temperature_window"))
    )
    
    # Trigger the kill switch if the network outputs a score higher than 0.85
    kill_switch_df = scored_df.filter(col("anomaly_score") > 0.85)

    # 4. Write Stream - Continuous Processing Mode
    # Using trigger(continuous='50 milliseconds') bypasses micro-batching entirely
    print("Executing Continuous Processing Engine...")
    query = kill_switch_df \
        .writeStream \
        .format("console") \
        .trigger(continuous="50 milliseconds") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    initialize_spark_processor()
