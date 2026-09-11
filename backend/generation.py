"""
Génération de réponse : construction du prompt contraint (anti-hallucination,
garde-fou éthique) et appel au LLM via l'API Groq.
"""

import os
import re
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELE_LLM = "openai/gpt-oss-120b"


INSTRUCTIONS_LANGUE = {
    "fr": {
        "consigne_langue": "Réponds en français.",
        "consigne_base": "Tu es un assistant d'information santé spécialisé en neurologie. Réponds à la question UNIQUEMENT à partir des sources fournies ci-dessous. Si l'information n'est pas présente dans les sources, dis clairement que tu ne sais pas plutôt que d'inventer une réponse.\n\nCite le nom de la pathologie source dans ta réponse.\n\nPrécise systématiquement à la fin de ta réponse que ceci est une information générale et ne remplace pas l'avis d'un professionnel de santé.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
        "reponse_label": "RÉPONSE",
    },
    "en": {
        "consigne_langue": "Answer in English, even though the sources below are in French — translate the relevant information yourself.",
        "consigne_base": "You are a health information assistant specialized in neurology. Answer the question ONLY based on the sources provided below. If the information is not present in the sources, clearly say you don't know rather than inventing an answer.\n\nCite the name of the source pathology in your answer.\n\nAlways state at the end of your answer that this is general information and does not replace the advice of a healthcare professional.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
        "reponse_label": "ANSWER",
    },
}


def construire_prompt(question, chunks_retrouves, langue="fr"):
    """Assemble le contexte récupéré et la question dans un prompt structuré,
    en contraignant le LLM à ne répondre qu'à partir des sources fournies,
    et dans la langue demandée (les sources elles-mêmes restent en français)."""
    textes = INSTRUCTIONS_LANGUE.get(langue, INSTRUCTIONS_LANGUE["fr"])

    contexte = "\n\n".join([
        f"[Source {i+1} — {c['pathologie']} / {c['section']}]\n{c['texte']}"
        for i, c in enumerate(chunks_retrouves)
    ])

    return f"""{textes['consigne_base']}

{textes['consigne_langue']}

{textes['sources_label']}:
{contexte}

{textes['question_label']}: {question}

{textes['reponse_label']}:"""


def lier_citations_sources(texte_reponse, chunks_retrouves):
    """Remplace chaque mention 'Source N' dans le texte généré par un lien Markdown
    cliquable vers la pathologie correspondante (ex: 'Source 3' -> '[Source 3](url)').
    L'ordre des chunks_retrouves correspond à la numérotation utilisée dans le prompt
    (Source 1 = chunks_retrouves[0], etc.)."""

    def remplacer(match):
        numero = int(match.group(1))
        index = numero - 1
        if 0 <= index < len(chunks_retrouves):
            url = chunks_retrouves[index]["source_url"]
            return f"[Source {numero}]({url})"
        return match.group(0)  # numéro hors limites, on laisse le texte inchangé

    # Capture "Source 3", "[Source 3]", "(Source 3)" — avec ou sans crochets/parenthèses
    return re.sub(r"\[?Source\s+(\d+)\]?", remplacer, texte_reponse)


def generer_reponse(question, pipeline, langue="fr", k=5, max_tokens=800, temperature=0.3):
    """Pipeline complet : retrieval + génération. Retourne la réponse texte
    et la liste des pathologies utilisées comme sources.
    langue : "fr" ou "en" — la recherche (retrieval) reste en français,
    seule la réponse générée change de langue."""
    chunks_retrouves = pipeline.rechercher(question, k=k)
    prompt = construire_prompt(question, chunks_retrouves, langue=langue)

    reponse = client_groq.chat.completions.create(
        model=MODELE_LLM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    texte_reponse = lier_citations_sources(reponse.choices[0].message.content, chunks_retrouves)

    pathologies_utilisees = list(set(c["pathologie"] for c in chunks_retrouves))

    sources_uniques = {}
    for c in chunks_retrouves:
        if c["pathologie"] not in sources_uniques:
            sources_uniques[c["pathologie"]] = c["source_url"]

    return {
        "reponse": texte_reponse,
        "pathologies": pathologies_utilisees,
        "sources_urls": sources_uniques,
        "sources_detaillees": [(c["pathologie"], c["section"]) for c in chunks_retrouves]
    }
