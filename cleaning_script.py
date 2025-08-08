#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

# ---- Spark <-> MinIO (localhost) ----
spark = SparkSession.builder \
    .appName("Clean_Parquet_MinIO") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9000") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .getOrCreate()

# ---- chemins sur MinIO ----
INPUT_PATH  = "s3a://prices/strat1/"        # <-- mets "s3a://prices/ma1_raw/" si c’est là
OUTPUT_PATH = "s3a://prices/strat1_clean/"

print(f"[INFO] Read:  {INPUT_PATH}")
df = spark.read.parquet(INPUT_PATH)

print("\n== SCHEMA =="); df.printSchema()
print("\n== SAMPLE =="); df.show(10, truncate=False)

# Normalisation minimale
if "time" in df.columns:   df = df.withColumn("time", to_timestamp(col("time")))
if "symbol" in df.columns: df = df.withColumn("symbol", col("symbol").cast("string"))
if "ma1" in df.columns:    df = df.withColumn("ma1", col("ma1").cast("double"))

before = df.count()
df = df.dropna().dropDuplicates()
after = df.count()
print(f"\n[INFO] rows: {before} -> {after}")

print(f"[INFO] Write: {OUTPUT_PATH}")
df.write.mode("overwrite").parquet(OUTPUT_PATH)
print("✅ Done")
spark.stop()

