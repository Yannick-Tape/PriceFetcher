#!/usr/bin/env bash
# test_kafka_pipeline.sh
# Usage: ./test_kafka_pipeline.sh

TOPIC="crypto-prices"
BROKER_IP="34.255.191.210:9092"       # IP publique de la VM
CSV="prices.csv"
TEST_SYMBOL="AUTOTEST"
TEST_PRICE="123.45"
TS=$(date +%s)                       # timestamp unique

# 1. Publier le message de test dans Kafka via le conteneur
echo "🔄 Envoi du message de test dans Kafka ($BROKER_IP)..."
echo "${TS},${TEST_SYMBOL},${TEST_PRICE}" | \
  docker exec -i $(docker-compose ps -q kafka) \
    kafka-console-producer \
      --broker-list ${BROKER_IP} \
      --topic ${TOPIC} > /dev/null

# 2. Attendre que le consumer écrive dans prices.csv
echo "⏳ Attente de 5 secondes…"
sleep 5

# 3. Vérifier la présence dans le CSV
echo "🔍 Vérification dans ${CSV}…"
if tail -n 10 "${CSV}" | grep -q "${TEST_SYMBOL},${TEST_PRICE}"; then
  echo "✅ Succès : la ligne de test a été écrite."
  RESULT=0
else
  echo "❌ Échec : la ligne de test n'a pas été trouvée."
  RESULT=1
fi

# 4. Nettoyer la ligne de test
echo "🧹 Nettoyage de la ligne de test…"
sed -i "/${TS},${TEST_SYMBOL},${TEST_PRICE}/d" "${CSV}"

exit ${RESULT}

