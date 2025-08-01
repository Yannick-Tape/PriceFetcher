#!/usr/bin/env python3

import time
import requests
from kafka import KafkaProducer

API_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"
BROKER = "34.255.191.210:9092"
TOPIC = "crypto-prices"

# Création d’un producteur Kafka 100% Python
producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: v.encode("utf-8")  # on envoie des chaînes UTF-8
)

def fetch_price():
    """Récupère le prix actuel depuis Binance."""
    resp = requests.get(API_URL, params={"symbol": SYMBOL})
    return resp.json()["price"]

if __name__ == "__main__":
    print("🔄 Producer kafka-python démarré")
    while True:
        ts = time.time()
        price = fetch_price()
        payload = f"{ts},{SYMBOL},{price}"
        # envoi asynchrone du message
        producer.send(TOPIC, payload)
        producer.flush()         # force l’envoi avant de dormir
        print(f"→ envoyé : {payload}")
        time.sleep(10)           # pause 10 s



# À chaque boucle, on capte le prix et on l’envoie à Kafka comme si on 
# postait dans un canal, pour que d’autres services puissent lire

