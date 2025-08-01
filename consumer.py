#!/usr/bin/env python3

import csv
import os
from kafka import KafkaConsumer
from datetime import datetime

# Adresse du broker Kafka exposé par ta VM
BROKER = "34.255.191.210:9092"
TOPIC = "crypto-prices"
GROUP_ID = "price-writers"
CSV_FILE = "prices.csv"

# Initialisation du CSV si nécessaire
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "symbol", "price"])

# Création d’un consumer kafka-python
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BROKER,
    group_id=GROUP_ID,
    auto_offset_reset="earliest",
    value_deserializer=lambda v: v.decode("utf-8")
)

print("✍️ Consumer kafka-python démarré (broker:", BROKER, ")")
for msg in consumer:
    ts_str, sym, price_str = msg.value.split(",")
    dt = datetime.fromtimestamp(float(ts_str)).strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([dt, sym, f"{float(price_str):.2f}"])
    print(f"→ écrit dans CSV : {dt},{sym},{price_str}")


# C’est comme un ouvrier qui attend des colis dans un entrepôt (Kafka)
# et range chaque colis (message) dans un fichier (CSV)


