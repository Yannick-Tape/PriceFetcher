# river_trainer.py (version corrigée)

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F, types as T, Window

# ---- Hyperparamètres (via env) ----
FEATURE_PATH   = os.getenv("FEATURE_PATH", "s3a://prices/features_ma1/")
LABEL_H        = int(os.getenv("LABEL_HORIZON_STEPS", "6"))
THRESHOLD      = float(os.getenv("THRESHOLD", "0.60"))
FEE_BPS        = float(os.getenv("FEE_BPS", "1.0"))
LOG_PATH       = os.getenv("LOG_PATH", "s3a://prices/experiments/river_decisions")
APP_NAME       = os.getenv("APP_NAME", "river_trainer_simple")

# ---- MinIO ----
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_KEY      = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

spark = (
    SparkSession.builder.appName(APP_NAME)
    .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key", MINIO_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print(f"[INFO] Read features from: {FEATURE_PATH}")
print(f"[INFO] LABEL_H={LABEL_H} | THRESHOLD={THRESHOLD} | FEE_BPS={FEE_BPS}")

# 1) Load & sort
df = spark.read.parquet(FEATURE_PATH)
for c in ["time", "symbol", "ma1"]:
    if c not in df.columns:
        raise ValueError(f"Colonne obligatoire manquante: {c}")
df = df.withColumn("time", F.col("time").cast(T.TimestampType()))

# 2) Label = signe du retour futur sur ma1 (corrigé: lead(..., offset))
w = Window.orderBy("time")
df = df.withColumn("ma1_fwd", F.lead("ma1", LABEL_H).over(w))
df = df.withColumn("ret_fwd", (F.col("ma1_fwd") / F.col("ma1") - 1.0))
df = df.withColumn("label", (F.col("ret_fwd") > 0).cast("int"))
df = df.orderBy("time").where(F.col("ma1_fwd").isNotNull())

# 3) Auto‑features numériques
exclude = {"time", "symbol", "label", "ma1_fwd", "ret_fwd"}
num_cols = [f for f, t in df.dtypes if t in ("double", "float", "int", "bigint") and f not in exclude]
if "ma1" not in num_cols:
    num_cols = ["ma1"] + num_cols
print(f"[INFO] features: {num_cols}")

# 4) Pull to driver (ok petit volume, pédagogique)
sel_cols = ["time", "label", "ret_fwd"] + num_cols
rows_iter = df.select(*sel_cols).toLocalIterator()

# 5) Modèle River
from river import linear_model, preprocessing, metrics
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
acc = metrics.Accuracy()

# 6) Online loop + PnL
decisions = []
fee = FEE_BPS / 10000.0
n = 0
n_trades = 0
pnl = 0.0

for r in rows_iter:
    n += 1
    x = {c: (float(getattr(r, c)) if getattr(r, c) is not None else 0.0) for c in num_cols}
    y = int(r.label)
    ret_fwd = float(r.ret_fwd) if r.ret_fwd is not None else 0.0

    proba = model.predict_proba_one(x).get(1, 0.5)
    signal = 1 if proba >= THRESHOLD else 0

    trade_pnl = 0.0
    if signal == 1:
        n_trades += 1
        trade_pnl = ret_fwd - fee
    pnl += trade_pnl

    decisions.append({
        "time": r.time.isoformat() if r.time else None,
        "proba_up": proba,
        "signal": signal,
        "y_true": y,
        "ret_fwd": ret_fwd,
        "trade_pnl": trade_pnl
    })

    acc = acc.update(y, signal)
    model = model.learn_one(x, y)

# 7) Résumé
acc_val = acc.get()
print(f"[RESULT] n={n} | accuracy={acc_val:.4f} | trades={n_trades} | pnl(sum)={pnl:.6f}")

# 8) Sauvegarde journal CSV sur MinIO
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(LOG_PATH, f"run_{ts}_th{int(THRESHOLD*100)}_bps{int(FEE_BPS)}")

dec_df = spark.createDataFrame(
    decisions,
    schema=T.StructType([
        T.StructField("time", T.StringType(), True),
        T.StructField("proba_up", T.DoubleType(), True),
        T.StructField("signal", T.IntegerType(), True),
        T.StructField("y_true", T.IntegerType(), True),
        T.StructField("ret_fwd", T.DoubleType(), True),
        T.StructField("trade_pnl", T.DoubleType(), True),
    ])
)
(dec_df.coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(out_dir))

print(f"[INFO] decisions saved to: {out_dir}")
print("✅ Done")

