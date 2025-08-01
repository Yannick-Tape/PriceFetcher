#!/usr/bin/env python3
import time                       # Gestion des pauses
import requests                   # Requêtes HTTP
import csv                        # Lecture/écriture CSV
import os                         # Vérification de l'existence de fichiers
from datetime import datetime     # Horodatage

API_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"                # Paire BTC/USD sur Binance
CSV_FILE = "prices.csv"           # Fichier de sortie

def fetch_price():
    """Interroge l'API Binance et retourne le prix actuel."""
    response = requests.get(API_URL, params={"symbol": SYMBOL})  # Appel HTTP
    data = response.json()                                        # Conversion JSON
    return float(data["price"])                                 # Extraction du prix

def init_csv():
    """Crée le CSV avec l’en-tête si nécessaire."""
    if not os.path.exists(CSV_FILE):                              # Fichier absent ?
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "price"])  # En-tête

def append_csv(timestamp, symbol, price):
    """Ajoute une ligne au CSV."""
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, symbol, f"{price:.2f}"])     # Ligne de données

def main():
    init_csv()                                                    # Prépare le CSV
    print("Phase 1 – Stockage local : écriture dans prices.csv")
    while True:
        try:
            price = fetch_price()                                   # Récupère le prix
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")     # Format timestamp
            print(f"[{now}] {SYMBOL} = {price:.2f} USDT")          # Affiche en console
            append_csv(now, SYMBOL, price)                          # Enregistre dans le CSV
        except Exception as e:
            print("Erreur lors de la récupération ou de l'écriture :", e)
        time.sleep(10)                                              # Pause 10 secondes

if __name__ == "__main__":
    main()                                                         # Point d'entrée

