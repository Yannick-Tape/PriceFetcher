#!/usr/bin/env bash
set -e

# Charger les variables d'environnement depuis .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

### Configuration (modifie si besoin) ###
CONTAINER_NAME=minio
IMAGE=quay.io/minio/minio:latest
ROOT_USER=minioadmin
ROOT_PASS=minioadmin
MC_ALIAS=myminio
BUCKET_NAME=prices
# Utiliser l'adresse IP de la VM définie dans .env
MINIO_HOST=${VM_IP}
MINIO_PORT=9000
########################################

# 1. Supprimer un ancien container s’il existe
docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$" && {
  echo "🗑  Suppression de l'ancien container ${CONTAINER_NAME}..."
  docker rm -f ${CONTAINER_NAME}
}

# 2. Lancer MinIO
echo "�� Démarrage de MinIO (${CONTAINER_NAME})..."
docker run -d --name ${CONTAINER_NAME} \
  -p ${MINIO_PORT}:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=${ROOT_USER} \
  -e MINIO_ROOT_PASSWORD=${ROOT_PASS} \
  ${IMAGE} server /data --console-address ":9001"

# 3. Attendre que MinIO réponde
echo -n "⏳ Attente de MinIO sur http://${MINIO_HOST}:${MINIO_PORT} "
until curl -sI http://${MINIO_HOST}:${MINIO_PORT} 2>/dev/null | head -1 | grep -q "HTTP/"; do
  echo -n "."
  sleep 1
done
echo " OK"

# 4. Configurer l’alias mc
echo "🔧 Configuration de l’alias mc (${MC_ALIAS})..."
mc alias set ${MC_ALIAS} http://${MINIO_HOST}:${MINIO_PORT} ${ROOT_USER} ${ROOT_PASS} --api S3v4

# 5. Créer le bucket
echo "�� Création du bucket '${BUCKET_NAME}'..."
mc mb ${MC_ALIAS}/${BUCKET_NAME} && \
  echo "✅ Bucket '${BUCKET_NAME}' créé." || \
  echo "ℹ️  Le bucket '${BUCKET_NAME}' existe peut-être déjà."

echo "🎉 MinIO est prêt à l’usage !"

