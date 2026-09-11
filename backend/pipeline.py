"""
Pipeline de retrieval : chargement du corpus, chunking, embeddings,
indexation FAISS, recherche hybride (sémantique + BM25) avec fusion RRF
et seuil de pertinence adaptatif.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def decouper_en_chunks(texte, taille_chunk=200, chevauchement=50):
    """Découpe un texte en chunks de taille fixe (en mots) avec chevauchement."""
    mots = texte.split()
    chunks = []
    debut = 0
    while debut < len(mots):
        fin = debut + taille_chunk
        chunks.append(" ".join(mots[debut:fin]))
        if fin >= len(mots):
            break
        debut += taille_chunk - chevauchement
    return chunks


def construire_chunks_corpus(corpus, taille_chunk=200, chevauchement=50):
    """Transforme le corpus brut (sections + médicaments par pathologie) en liste de chunks."""
    chunks_corpus = []
    for nom_pathologie, data in corpus.items():
        for nom_section, paragraphes in data["sections"].items():
            texte_section = " ".join(paragraphes)
            if not texte_section.strip():
                continue
            for i, chunk_texte in enumerate(decouper_en_chunks(texte_section, taille_chunk, chevauchement)):
                chunks_corpus.append({
                    "pathologie": nom_pathologie,
                    "section": nom_section,
                    "type": "texte",
                    "sous_index": i,
                    "texte": chunk_texte,
                    "source_url": data["url_source"]
                })
        for i, ligne_medicament in enumerate(data["medicaments"]):
            chunks_corpus.append({
                "pathologie": nom_pathologie,
                "section": "Médicaments",
                "type": "medicament",
                "sous_index": i,
                "texte": ligne_medicament,
                "source_url": data["url_source"]
            })
    return chunks_corpus


def normaliser(vecteurs):
    """Normalise des vecteurs (norme L2 = 1), nécessaire pour que le produit scalaire
    FAISS (IndexFlatIP) équivale à une similarité cosinus."""
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    return vecteurs / normes


class PipelineRAG:
    """Encapsule le modèle d'embedding, l'index FAISS et BM25 pour un corpus donné."""

    def __init__(self, chemin_corpus_json, nom_modele="paraphrase-multilingual-MiniLM-L12-v2"):
        with open(chemin_corpus_json, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        self.chunks_corpus = construire_chunks_corpus(corpus)
        self.model = SentenceTransformer(nom_modele)

        textes_enrichis = [
            f"{c['pathologie']} - {c['section']}: {c['texte']}" for c in self.chunks_corpus
        ]

        embeddings = self.model.encode(textes_enrichis, show_progress_bar=False)
        self.embeddings_normalises = normaliser(embeddings).astype("float32")

        dimension = self.embeddings_normalises.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings_normalises)

        textes_tokenises = [t.lower().split() for t in textes_enrichis]
        self.bm25 = BM25Okapi(textes_tokenises)

    def rechercher(self, question, k=5, k_rrf=60, marge_relative=0.15, seuil_absolu_min=0.3):
        """Recherche hybride : fusion par rang (RRF) entre similarité sémantique et BM25,
        avec un seuil sémantique adaptatif (relatif au meilleur score) pour écarter
        les chunks hors-sujet (ex: contamination entre pathologies partageant du vocabulaire)."""
        n_total = len(self.chunks_corpus)

        vecteur_question = self.model.encode([question])
        vecteur_question = normaliser(vecteur_question).astype("float32")
        scores_sem, indices_sem = self.index.search(vecteur_question, n_total)

        score_semantique_par_idx = {idx: s for idx, s in zip(indices_sem[0], scores_sem[0])}
        rang_semantique = {idx: r for r, idx in enumerate(indices_sem[0])}

        meilleur_score = float(scores_sem[0][0])
        seuil_dynamique = max(meilleur_score - marge_relative, seuil_absolu_min)

        scores_bm25 = self.bm25.get_scores(question.lower().split())
        rang_bm25 = {idx: r for r, idx in enumerate(np.argsort(scores_bm25)[::-1])}

        scores_rrf = []
        for idx in range(n_total):
            score_sem = score_semantique_par_idx.get(idx, 0)
            if score_sem < seuil_dynamique:
                continue
            r_sem = rang_semantique.get(idx, n_total)
            r_bm25 = rang_bm25.get(idx, n_total)
            score_combine = 1 / (k_rrf + r_sem) + 1 / (k_rrf + r_bm25)
            scores_rrf.append((idx, score_combine))

        scores_rrf.sort(key=lambda x: x[1], reverse=True)

        return [self.chunks_corpus[idx] for idx, _ in scores_rrf[:k]]
