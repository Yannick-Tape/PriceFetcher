#!/usr/bin/env bash

# 1. Ajout d’une ligne « test » à la fin du CSV
echo "TEST_TIMESTAMP,TESTSYM,999.99" >> prices.csv

# 2. Appel de l’API et filtration du prix renvoyé
RESPONSE=$(curl -s http://34.255.191.210:8000/price/latest)
PRICE=$(echo "$RESPONSE" | grep -oP '(?<="price":)[0-9]+\.[0-9]+')

# 3. Vérification simple
if [[ "$PRICE" == "999.99" ]]; then
  echo "✅ Succès : l’API reflète bien la nouvelle ligne du CSV."
else
  echo "❌ Échec : l’API n’a pas renvoyé 999.99 (elle a renvoyé $PRICE)."
fi

