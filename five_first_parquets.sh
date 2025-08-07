#!/usr/bin/env bash
set -e

# 1. Prépare un répertoire temporaire
TMPDIR=$(mktemp -d)
echo "⏳ Répertoire temporaire créé : $TMPDIR"

# 2. Récupère les 5 premiers noms de fichiers parquet dont la taille n'est pas 466B
FILES=$(mc ls myminio/prices/strat1 \
  | grep '\.parquet$' \
  | grep -v ' 466B ' \
  | head -5 \
  | awk '{print $NF}')

# 3. Pour chaque fichier : on le télécharge puis on affiche son contenu
for F in $FILES; do
  echo -e "\n=== $F ==="
  mc cp myminio/prices/strat1/$F "$TMPDIR/$F"

  python3 <<EOF
import pandas as pd

# Chargement du Parquet
df = pd.read_parquet("$TMPDIR/$F", engine="pyarrow")

# Affichage
print(df)
EOF

done

# 4. (Optionnel) Nettoyage  
# rm -rf "$TMPDIR"

