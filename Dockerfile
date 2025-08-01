#  - Expose l'API FastAPI avec Uvicorn
#            -base Python 3.9, installation de FastAPI et Uvicorn, copie du CSV et du script, exposition du port 8000 et démarrage du serveur.
FROM python:3.9-slim
WORKDIR /app
COPY main.py /app/main.py
COPY prices.csv /app/prices.csv
RUN pip install fastapi uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

