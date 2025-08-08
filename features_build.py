#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------- Spark <-> MinIO ----------
spark = SparkSession.builder \
    .appName("Build_Features_MA1") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9000") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .getOrCreate()

INPUT_PATH  = "s3a://prices/strat1_clean/"
OUTPUT_PATH = "s3a://prices/features_ma1/"

print(f"[INFO] Read:  {INPUT_PATH}")
df = spark.read.parquet(INPUT_PATH).select("time", "symbol", "ma1") \
    .dropna(subset=["time", "symbol", "ma1"])

# On ordonne temporellement par symbol
w = Window.partitionBy("symbol").orderBy(F.col("time").cast("long")) \
          .rowsBetween(Window.unboundedPreceding, 0)
w_tail_5   = Window.partitionBy("symbol").orderBy(F.col("time").cast("long")).rowsBetween(-5, 0)
w_tail_10  = Window.partitionBy("symbol").orderBy(F.col("time").cast("long")).rowsBetween(-10, 0)
w_tail_20  = Window.partitionBy("symbol").orderBy(F.col("time").cast("long")).rowsBetween(-20, 0)

# Lags
df = df.withColumn("ma1_prev", F.lag("ma1").over(Window.partitionBy("symbol").orderBy("time")))
# Diff & % change
df = df.withColumn("ma1_diff", F.when(F.col("ma1_prev").isNotNull(), F.col("ma1") - F.col("ma1_prev")))
df = df.withColumn("ma1_pct",  F.when(F.col("ma1_prev") > 0, (F.col("ma1")/F.col("ma1_prev") - 1)))

# Slope locale (régression linéaire simple sur les 6 derniers points ~ 1 min si pas de trous)
# approx slope ≈ cov(t, ma1)/var(t) avec t = index relatif
t = F.sequence(F.lit(0), F.lit(5))
# Moyennes glissantes
df = df.withColumn("ma1_mean_6",  F.avg("ma1").over(w_tail_5)) \
       .withColumn("ma1_std_6",   F.stddev_samp("ma1").over(w_tail_5)) \
       .withColumn("ma1_mean_20", F.avg("ma1").over(w_tail_20)) \
       .withColumn("ma1_std_20",  F.stddev_samp("ma1").over(w_tail_20))

# z-score court
df = df.withColumn("zscore_6",
                   F.when(F.col("ma1_std_6") > 0, (F.col("ma1") - F.col("ma1_mean_6"))/F.col("ma1_std_6")))

# Volatilité % sur ~2 minutes (20 points si batch 10s)
df = df.withColumn("pct_abs", F.abs(F.col("ma1_pct"))) \
       .withColumn("vol_20", F.avg("pct_abs").over(w_tail_20))

# Moyennes “short/long” (proxy momentum)
df = df.withColumn("ma_short_10", F.avg("ma1").over(w_tail_10)) \
       .withColumn("ma_long_20",  F.avg("ma1").over(w_tail_20)) \
       .withColumn("ma_cross",    F.when(F.col("ma_short_10") > F.col("ma_long_20"), F.lit(1)).otherwise(F.lit(0)))

# Label simple pour backtest: direction future à 3 pas (≈30s)
df = df.withColumn("ma1_fwd3", F.lead("ma1", 3).over(Window.partitionBy("symbol").orderBy("time"))) \
       .withColumn("label_up3", F.when(F.col("ma1_fwd3") > F.col("ma1"), F.lit(1)).otherwise(F.lit(0)))

# Nettoyage NA et tri final
out_cols = [
    "time","symbol","ma1","ma1_prev","ma1_diff","ma1_pct",
    "ma1_mean_6","ma1_std_6","ma1_mean_20","ma1_std_20",
    "zscore_6","vol_20","ma_short_10","ma_long_20","ma_cross",
    "ma1_fwd3","label_up3"
]
df_out = df.select(*out_cols).dropna()

print("\n== FEATURES SAMPLE ==")
df_out.orderBy("time").show(10, truncate=False)

print(f"[INFO] Write: {OUTPUT_PATH}")
df_out.write.mode("overwrite").parquet(OUTPUT_PATH)
print("✅ Features written")
spark.stop()

