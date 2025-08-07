#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, window, avg, from_unixtime
from pyspark.sql.types import DoubleType

spark = SparkSession.builder \
    .appName("Strategy1SlidingWindow") \
    .config("spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4") \
    .getOrCreate()

# ===== S3A pour MinIO =====
hconf = spark._jsc.hadoopConfiguration()
hconf.set("fs.s3a.impl",                        "org.apache.hadoop.fs.s3a.S3AFileSystem")
hconf.set("fs.s3a.endpoint",                    os.getenv("MINIO_ENDPOINT"))
hconf.set("fs.s3a.access.key",                  os.getenv("MINIO_ACCESS_KEY"))
hconf.set("fs.s3a.secret.key",                  os.getenv("MINIO_SECRET_KEY"))
hconf.set("fs.s3a.path.style.access",           "true")
hconf.set("fs.s3a.aws.credentials.provider",    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
# Désactivez SSL si votre endpoint est en http://
hconf.set("fs.s3a.connection.ssl.enabled",      "false")

# Lecture Kafka
raw = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", os.getenv("KAFKA_BROKER")) \
    .option("subscribe", "crypto-prices") \
    .option("startingOffsets", "earliest") \
    .load()

parsed = raw.selectExpr("CAST(value AS STRING) as v") \
    .select(
        split(col("v"), ",")[0].cast(DoubleType()).alias("epoch_s"),
        split(col("v"), ",")[1].alias("symbol"),
        split(col("v"), ",")[2].cast(DoubleType()).alias("price")
    ) \
    .withColumn("ts", from_unixtime(col("epoch_s")).cast("timestamp"))

df = parsed.withWatermark("ts", "2 minutes")

strat1 = df.groupBy(
        window(col("ts"), "1 minute", "10 seconds"),
        col("symbol")
    ) \
    .agg(avg("price").alias("ma1")) \
    .select(
        col("window.start").alias("time"),
        col("symbol"),
        col("ma1")
    )

# Sink console pour debug
strat1.writeStream \
    .format("console") \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .option("truncate", False) \
    .start()

# Sink Parquet dans MinIO
strat1.writeStream \
    .format("parquet") \
    .option("path",               "s3a://prices/strat1") \
    .option("checkpointLocation", "s3a://prices/strat1-chkpt") \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

spark.streams.awaitAnyTermination()

