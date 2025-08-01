
# ici main.py est une API FastAPI minimale qui lit la dernière ligne de prices.csv et la renvoie en JSON via /price/latest


from fastapi import FastAPI, HTTPException
import csv
import os

app = FastAPI()
CSV_FILE = "prices.csv"

@app.get("/price/latest")
def get_latest_price():
    """
    Renvoie la dernière ligne du CSV sous forme JSON : {timestamp, symbol, price}
    """
    if not os.path.exists(CSV_FILE):
        raise HTTPException(status_code=404, detail="Fichier CSV non trouvé")

    with open(CSV_FILE, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise HTTPException(status_code=404, detail="Aucune donnée disponible")
        last = rows[-1]
    return {"timestamp": last["timestamp"], "symbol": last["symbol"], "price": float(last["price"])}
