"""
Génération de réponse : construction du prompt contraint (anti-hallucination,
garde-fou éthique) et appel au LLM via l'API Groq.
"""

import os
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELE_LLM = "openai/gpt-oss-120b"


def construire_prompt(question, chunks_retrouves):
    """Assemble le contexte récupéré et la question dans un prompt structuré,
    en contraignant le LLM à ne répondre qu'à partir des sources fournies."""
    contexte = "\n\n".join([
        f"[Source {i+1} — {c['pathologie']} / {c['section']}]\n{c['texte']}"
        for i, c in enumerate(chunks_retrouves)
    ])

    return f"""Tu es un assistant d'information santé spécialisé en neurologie. Réponds à la question UNIQUEMENT à partir des sources fournies ci-dessous. Si l'information n'est pas présente dans les sources, dis clairement que tu ne sais pas plutôt que d'inventer une réponse.

Cite le nom de la pathologie source dans ta réponse.

Précise systématiquement à la fin de ta réponse que ceci est une information générale et ne remplace pas l'avis d'un professionnel de santé.

SOURCES:
{contexte}

QUESTION: {question}

RÉPONSE:"""


def generer_reponse(question, pipeline, k=5, max_tokens=800, temperature=0.3):
    """Pipeline complet : retrieval + génération. Retourne la réponse texte
    et la liste des pathologies utilisées comme sources."""
    chunks_retrouves = pipeline.rechercher(question, k=k)
    prompt = construire_prompt(question, chunks_retrouves)

    reponse = client_groq.chat.completions.create(
        model=MODELE_LLM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    pathologies_utilisees = list(set(c["pathologie"] for c in chunks_retrouves))

    return {
        "reponse": reponse.choices[0].message.content,
        "pathologies": pathologies_utilisees,
        "sources_detaillees": [(c["pathologie"], c["section"]) for c in chunks_retrouves]
    }
