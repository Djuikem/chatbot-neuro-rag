# Chatbot RAG — Assistant d'information santé en neurologie

## Documentation technique du projet

---

## 1. Contexte et objectif

Ce projet est un chatbot conversationnel capable de répondre à des questions de santé
portant sur des pathologies neurologiques (migraine, épilepsie, Parkinson, Alzheimer,
sclérose en plaques, etc.), en français et en anglais.

Il repose sur une architecture **RAG (Retrieval-Augmented Generation)** : plutôt que
de laisser un modèle de langage répondre uniquement à partir de ses connaissances
générales (avec le risque d'hallucination que cela comporte sur un sujet médical),
le système va d'abord **chercher les informations pertinentes dans une base de
connaissances vérifiée**, puis demande au modèle de langage de formuler une réponse
**strictement basée sur ces sources**.

---

## 2. Vue d'ensemble de l'architecture

```
Question utilisateur
        │
        ▼
┌───────────────────┐
│   1. RETRIEVAL     │  → cherche les passages pertinents dans le corpus
│  (FAISS + BM25)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2. CONSTRUCTION    │  → assemble les passages trouvés + la question
│    DU PROMPT        │     dans un texte structuré pour le LLM
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  3. GÉNÉRATION      │  → le LLM (via l'API Groq) rédige la réponse,
│   (LLM via Groq)    │     en se basant uniquement sur les sources fournies
└───────────────────┘
        │
        ▼
   Réponse finale + liens vers les sources
```

Le corpus de connaissances (19 pathologies neurologiques) a été construit par
scraping du site **MSD Manuals** (version grand public, en français), puis
structuré et découpé en unités exploitables ("chunks").

---

## 3. Détail des composants

### 3.1 Constitution du corpus (scraping)

Les fiches de 19 pathologies neurologiques ont été extraites automatiquement
depuis MSD Manuals avec les bibliothèques `requests` et `BeautifulSoup`. Chaque
fiche est structurée en sections (introduction, causes, symptômes, diagnostic,
traitement, prévention) ainsi qu'en tableaux de médicaments, reconstruits
proprement malgré des structures HTML complexes (cellules fusionnées).

### 3.2 Chunking (découpage du texte)

Chaque section est découpée en segments d'environ 200 mots, avec un
chevauchement de 50 mots entre segments consécutifs. Ce chevauchement évite
qu'une idée importante soit coupée entre deux chunks sans contexte.

### 3.3 Embeddings et indexation (FAISS)

Chaque chunk est transformé en vecteur numérique (384 dimensions) grâce au
modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2`
(bibliothèque `sentence-transformers`). Ces vecteurs sont indexés avec
**FAISS** (bibliothèque de recherche vectorielle développée par Meta), qui
permet de retrouver rapidement les chunks les plus proches sémantiquement
d'une question donnée.

### 3.4 Retrieval hybride (recherche)

Le système combine deux approches de recherche :
- **Recherche sémantique** (FAISS) : capture le sens général de la question,
  même si les mots exacts diffèrent du corpus.
- **Recherche lexicale** (BM25) : capture la présence de mots-clés précis.

Les deux classements sont fusionnés par **Reciprocal Rank Fusion (RRF)**, une
méthode qui combine des classements plutôt que des scores bruts (évitant les
problèmes d'échelles incompatibles entre les deux méthodes).

Un **seuil de pertinence adaptatif** (relatif au meilleur score trouvé, plutôt
qu'une valeur fixe) filtre les chunks dont le lien avec la question est trop
faible — ce qui réduit la contamination entre pathologies partageant du
vocabulaire médical proche (ex : "traitement", "anticonvulsivants").

### 3.5 Génération de la réponse (LLM)

Les chunks retenus sont assemblés dans un prompt qui contraint le modèle de
langage à :
- répondre **uniquement** à partir des sources fournies ;
- dire explicitement qu'il ne sait pas si l'information est absente du corpus,
  plutôt que d'inventer une réponse ;
- citer la pathologie source dans sa phrase ;
- ajouter systématiquement un rappel que l'information est générale et ne
  remplace pas un avis médical professionnel.

Une détection simple par mots-clés identifie les questions qui semblent
décrire une situation vécue en temps réel (ex : "je fais une crise
maintenant") et adapte la réponse pour que les consignes de sécurité
critiques (quand appeler les secours) arrivent en priorité, avant les détails.

---

## 4. Pourquoi une clé API Groq ?

### Le rôle de cette clé

L'étape de génération (rédaction de la réponse finale) nécessite d'appeler un
**grand modèle de langage (LLM)**. Or, entraîner ou héberger soi-même un tel
modèle est hors de portée d'un projet étudiant (puissance de calcul énorme,
coûts élevés). La solution standard est d'appeler ce modèle via une **API** —
un service en ligne qui exécute le modèle et renvoie la réponse. La clé API
sert à authentifier ces appels (identifier qui fait la requête, et permettre
au fournisseur de suivre l'usage).

### Pourquoi Groq précisément, et pas une autre solution ?

Trois solutions ont été testées au cours du projet :

| Solution | Résultat |
|---|---|
| **API Claude (Anthropic)** | Fonctionne très bien, mais nécessite d'ajouter des crédits payants dès la première requête — écarté pour éviter tout coût |
| **Hugging Face (Inference API gratuite)** | Instable : plusieurs modèles testés se sont révélés indisponibles ou redirigés vers des offres payantes tierces (erreurs `model_not_found`, `model_not_supported`) |
| **Groq** | Fonctionne de façon stable, tier gratuit généreux, sans carte bancaire requise |

**Groq** est une entreprise qui propose l'inférence de modèles de langage
open-source (comme les modèles de la famille GPT-OSS ou Llama) sur du matériel
spécialisé (LPU — *Language Processing Unit*), ce qui lui permet d'offrir des
réponses **très rapides** tout en maintenant un tier gratuit exploitable pour
un usage de test/projet étudiant, contrairement à des services qui exigent des
crédits payants dès la première utilisation.

**Dans ce projet**, le modèle utilisé via Groq est `openai/gpt-oss-120b`, un
modèle open-source de 120 milliards de paramètres, suffisamment capable pour
produire des réponses cohérentes, structurées et fidèles aux sources fournies.

### Limite à noter

Le tier gratuit de Groq a des limites de débit (nombre de requêtes par
minute/jour). Pour un usage à plus grande échelle que le cadre de ce projet, un
passage à un tier payant serait nécessaire.

---

## 5. Fonctionnalités additionnelles

- **Support multilingue** (français / anglais) : le corpus reste en français,
  mais la réponse générée est traduite selon la langue choisie par
  l'utilisateur ; l'historique de conversation est conservé séparément pour
  chaque langue.
- **Liens cliquables vers les sources** : chaque réponse est accompagnée de
  liens directs vers les fiches MSD Manuals utilisées, pour vérification.
- **Gestion propre des questions hors-sujet** : si aucune information
  pertinente n'est trouvée dans le corpus, le système répond directement sans
  appeler le LLM, évitant tout risque d'invention de réponse hors du domaine
  couvert (neurologie).
- **Interface conversationnelle** (Streamlit) avec historique de session,
  sélecteur de langue, et bouton de réinitialisation.

---

## 6. Déploiement

Le projet est structuré en trois parties :
- `app.py` — interface utilisateur (Streamlit)
- `backend/pipeline.py` — retrieval (chunking, embeddings, FAISS, BM25, RRF)
- `backend/generation.py` — construction du prompt et appel au LLM (Groq)

Le code est versionné sur **GitHub** et déployé sur **Streamlit Community
Cloud** (hébergement gratuit), avec la clé API Groq stockée de façon sécurisée
dans les secrets de la plateforme (jamais écrite en dur dans le code).

---

## 7. Limites et perspectives

- Le corpus est limité à 19 pathologies neurologiques ; une extension à
  d'autres domaines médicaux nécessiterait un nouveau travail de collecte.
- Le tier gratuit d'hébergement (Streamlit Cloud) impose des limites de
  ressources (throttling en cas d'usage intensif).
- La détection d'urgence repose sur une simple recherche de mots-clés, pas sur
  un modèle de classification — une amélioration possible serait d'entraîner
  un classifieur dédié.
- Un déploiement à grande échelle (application publique, chatbot WhatsApp)
  nécessiterait une infrastructure payante et poserait des questions de
  responsabilité légale qui dépassent le cadre de ce projet académique.
