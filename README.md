# Chatbot Neurologie — RAG

Chatbot d'information santé spécialisé en neurologie, basé sur un pipeline RAG
(retrieval hybride sémantique + BM25 avec fusion RRF, génération via Groq).

## Structure du projet

```
chatbot-neuro-rag/
├── app.py                    # Interface Streamlit (léger, appelle le backend)
├── backend/
│   ├── pipeline.py           # Chunking, embeddings, FAISS, BM25, retrieval RRF
│   └── generation.py         # Construction du prompt + appel LLM (Groq)
├── requirements.txt
├── data/
│   └── corpus_neurologie.json   # À ajouter toi-même (voir ci-dessous)
```

## ⚠️ Étape manquante avant de pousser sur GitHub

Le fichier `data/corpus_neurologie.json` (ton corpus scrapé, 19 pathologies)
n'est PAS inclus ici — il est sur ton Google Drive, généré depuis Colab.

**Avant de push :**
1. Télécharge `corpus_neurologie.json` depuis ton Drive (ou reprends-le depuis
   la cellule Colab : `files.download(chemin_corpus)`)
2. Place-le dans le dossier `data/` de ce projet, à la racine, sous le nom exact
   `corpus_neurologie.json`

## Déploiement sur Streamlit Community Cloud

1. Pousse tout ce dossier sur un dépôt GitHub public
2. Va sur https://share.streamlit.io
3. Connecte ton compte GitHub, sélectionne le dépôt et `app.py` comme fichier principal
4. Dans les "Secrets" de l'app Streamlit Cloud (Settings → Secrets), ajoute :
   ```
   GROQ_API_KEY = "ta_clé_groq_ici"
   ```
5. Déploie — le premier chargement peut prendre quelques minutes (téléchargement
   du modèle d'embedding)

## Lancer en local (optionnel, pour tester avant de déployer)

```bash
pip install -r requirements.txt
export GROQ_API_KEY="ta_clé_ici"   # sur Windows : set GROQ_API_KEY=ta_clé_ici
streamlit run app.py
```
