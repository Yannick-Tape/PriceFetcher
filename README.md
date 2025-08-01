# 🤖 CryptoBot

## 📌 Objectif principal
Construire un **bot de trading crypto automatisé**, capable de collecter des données de marché, de les traiter en temps réel, de générer des signaux basés sur des modèles de machine learning et de fournir une interface pour le déploiement et le suivi en production.

---

## 🔄 Flux de données (end-to-end)

**📥 Collecte :**  
Appels à l’API Binance pour récupérer en continu les cours et le carnet d’ordres.

**🚚 Transport :**  
Messages Kafka pour publier chaque tick de marché.

**⚙️ Traitement :**  
Jobs Spark (PySpark) qui consomment Kafka, enrichissent/filtrent les données et stockent les résultats dans MinIO (data lake).

**🧠 Modélisation :**  
Entraînement et inférence de modèles (XGBoost, LSTM) sur les données historisées.

**🌐 API & UI :**  
FastAPI expose les signaux et les métriques ; Dash fournit un dashboard interactif.

**📌 Orchestration & suivi :**  
Airflow/MLflow pilotent les pipelines ETL, l’entraînement des modèles et versionnent les expériences.

**🚀 Déploiement :**  
Conteneurisation Docker + déploiement Kubernetes pour assurer scalabilité et haute disponibilité.

**📊 Monitoring :**  
Prometheus collecte les métriques ; Grafana affiche l’état du système et des stratégies en temps réel.

---

## 🛠️ Principales technologies

| Catégorie                | Technologies                              |
|--------------------------|-------------------------------------------|
| **Infrastructure**       | Docker, Docker Compose, Kubernetes        |
| **Ingestion & Streaming**| Kafka, Zookeeper                          |
| **Traitement Big Data**  | Spark Streaming (PySpark)                 |
| **Stockage**             | MinIO (S3-compatible)                     |
| **Modélisation IA**      | XGBoost, LSTM, SHAP (interprétabilité)    |
| **APIs & Orchestration** | FastAPI, Airflow, MLflow                  |
| **Visualisation & Monitoring** | Dash, Prometheus, Grafana           |

---

## 🚩 Découpage par phases

- ⚙️ **Infra de base** : VM Ubuntu + réseau + Docker Compose
- 🔗 **Ingestion** : API Binance + premiers topics Kafka
- 🔄 **Streaming** : Spark → MinIO
- 📊 **Data Lake** : préparation des features
- 🤖 **Modélisation** : entraînement / inférence ML
- 🚀 **API** : FastAPI pour exposer les signaux
- 🔄 **Orchestration** : Airflow / suivi MLflow
- 📈 **Dashboards** : Prometheus/Grafana
- 🐳 **Conteneurisation complète**
- ☸️ **Production** : Kubernetes

---

## ✅ Résultat attendu

Un système **modulaire et reproductible** qui :

- Capture et traite les données de marché en temps réel.
- Génère des signaux basés sur une stratégie hybride (Big Data + ML).
- Assure une orchestration et un monitoring automatisés pour un déploiement fiable en production.
