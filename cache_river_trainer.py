#!/usr/bin/env python3
# river_trainer.py
from dotenv import load_dotenv
load_dotenv()

import os
from collections import deque

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lag, lead
from pyspark.sql.window import Window

from river import preprocessing, linear_model, metrics

# ----------------------------
# 0) Paramètres
# ----------------------------
FEATURES_PATH = os.getenv("FEATURES_PATH", "s3a://prices/features_ma1/")
TIME_COL = os.getenv("TIME_COL", "time")
SYMBOL_COL = os.getenv("SYMBOL_COL", "symbol")
MA_COL = os.getenv("MA_COL", "ma1")
LABEL_H = int(os.getenv("LABEL_HORIZON_STEPS", "3"))  # nb de lignes dans le futur pour le label
ROLL_WIN = int(os.getenv("ROLLING_WINDOW", "500"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "1000"))

print(f"[INFO] Read features from: {FEATURES_PATH}")
print(f"[INFO] LABEL_HORIZON_STEPS={LABEL_H}, ROLLING_WINDOW={ROLL_WIN}")

# ----------------------------
# 1) Spark + MinIO (S3A)
# ----------------------------
spark = SparkSession.builder.appName("RiverTrainerMA1").getOrCreate()
hconf = spark._jsc.hadoopConfiguration()
hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hconf.set("fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"))
hconf.set("fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
hconf.set("fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
hconf.set("fs.s3a.path.style.access", "true")
hconf.set("fs.s3a.aws.credentials.provider",
          "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
if hconf.get("fs.s3a.endpoint", "").startswith("http://"):
    hconf.set("fs.s3a.connection.ssl.enabled", "false")

# ----------------------------
# 2) Lecture & création du label/feat si manquants
# ----------------------------
df = spark.read.parquet(FEATURES_PATH)

# tri chrono
if TIME_COL in df.columns:
    df = df.orderBy(col(TIME_COL).asc())

# fenêtre par symbole (ou globale si pas de symbole)
w = Window.partitionBy(SYMBOL_COL).orderBy(col(TIME_COL).asc()) if SYMBOL_COL in df.columns \
    else Window.orderBy(col(TIME_COL).asc())

# si pas de label => on le crée avec lead(MA_COL, LABEL_H)
if "label" not in df.columns and "y" not in df.columns:
    if MA_COL not in df.columns:
        raise ValueError(f"Impossible de créer un label: colonne {MA_COL!r} absente.")
    df = (
        df
        .withColumn("ma_fwd", lead(col(MA_COL), LABEL_H).over(w))
        .withColumn("label", when(col("ma_fwd") > col(MA_COL), 1).otherwise(0))
        .drop("ma_fwd")
    )

# features minimales : ma1 + deltas (si présentes)
if MA_COL in df.columns:
    df = (
        df
        .withColumn("delta1", col(MA_COL) - lag(col(MA_COL), 1).over(w))
        .withColumn("delta3", col(MA_COL) - lag(col(MA_COL), 3).over(w))
    )

# garder uniquement colonnes utiles (numériques)
ban = {TIME_COL, SYMBOL_COL}
label_col = "label" if "label" in df.columns else ("y" if "y" in df.columns else None)
if label_col is None:
    raise ValueError("Pas de label détecté et impossible d'en créer un. Vérifie tes colonnes.")

numeric_types = {"double", "float", "int", "bigint", "long", "decimal", "short"}
feature_cols = [c for (c, t) in df.dtypes if c not in ban | {label_col} and t in numeric_types]
if not feature_cols:
    # fallback : si seules colonnes utiles = MA_COL/deltas
    for c in [MA_COL, "delta1", "delta3"]:
        if c in df.columns:
            feature_cols.append(c)
    if not feature_cols:
        raise ValueError("Aucune feature numérique trouvée.")

print(f"[INFO] label_col = {label_col}")
print(f"[INFO] feature_cols = {feature_cols}")

# virer les lignes sans label ou features
cols_to_keep = feature_cols + [label_col]
df = df.select(*cols_to_keep).na.drop()

# ----------------------------
# 3) Modèle River + métriques
# ----------------------------
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
acc_global = metrics.Accuracy()
from collections import deque
roll_buf = deque(maxlen=ROLL_WIN)

def update_metrics(y_true, y_pred):
    acc_global.update(y_true, y_pred)
    roll_buf.append(1 if int(y_true) == int(y_pred) else 0)
    return acc_global.get(), (sum(roll_buf) / len(roll_buf))

# ----------------------------
# 4) One-pass training (online)
# ----------------------------
n = 0
for row in df.toLocalIterator():
    y = int(row[label_col])
    x = {c: float(row[c]) if row[c] is not None else 0.0 for c in feature_cols}

    # predict -> update metrics -> learn
    y_hat = int(model.predict_one(x) or 0)
    g, r = update_metrics(y, y_hat)
    model.learn_one(x, y)

    n += 1
    if n % LOG_EVERY == 0:
        print(f"[PROGRESS] n={n} | acc_global={g:.4f} | acc_rolling({ROLL_WIN})={r:.4f}")

final_roll = (sum(roll_buf) / len(roll_buf)) if roll_buf else 0.0
print(f"[RESULT] n={n} | acc_global={acc_global.get():.4f} | acc_rolling({ROLL_WIN})={final_roll:.4f}")
print("✅ Done")

