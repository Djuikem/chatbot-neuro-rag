"""
Génération de réponse : construction du prompt contraint (anti-hallucination,
garde-fou éthique) et appel au LLM via l'API Groq.
"""

import os
import re
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELE_LLM = "openai/gpt-oss-120b"


MOTS_CLES_URGENCE = {
    "fr": [
        "en ce moment", "là maintenant", "maintenant même", "je suis en train de",
        "j'ai une crise", "je fais une crise", "ça m'arrive là", "aidez-moi",
        "aide-moi vite", "urgence", "je ne me sens pas bien là", "juste là",
    ],
    "en": [
        "right now", "happening now", "i'm having a", "i am having a",
        "help me now", "emergency", "i feel bad right now", "currently having",
    ],
}


def detecter_urgence(question, langue="fr"):
    """Détection simple par mots-clés d'une situation vécue en direct
    (par opposition à une question générale/informative sur une pathologie).
    Volontairement simple (pas de ML) : suffisant pour adapter le ton de la
    réponse, pas destiné à être un système de triage médical fiable."""
    question_lower = question.lower()
    mots_cles = MOTS_CLES_URGENCE.get(langue, MOTS_CLES_URGENCE["fr"])
    return any(mot in question_lower for mot in mots_cles)


INSTRUCTIONS_LANGUE = {
    "fr": {
        "consigne_langue": "Réponds en français.",
        "consigne_base": "Tu es un assistant d'information santé spécialisé en neurologie. Réponds à la question UNIQUEMENT à partir des sources fournies ci-dessous. Si l'information n'est pas présente dans les sources, dis clairement que tu ne sais pas plutôt que d'inventer une réponse.\n\nCite le nom de la pathologie directement dans ta phrase (ex: \"la migraine peut être traitée par...\"). N'utilise PAS de notation du type [Source 1] ou (Source 2) — les liens vers les sources seront affichés séparément après ta réponse.\n\nPrécise systématiquement à la fin de ta réponse que ceci est une information générale et ne remplace pas l'avis d'un professionnel de santé.",
        "consigne_urgence": "\n\nATTENTION : cette question semble décrire une situation vécue EN CE MOMENT, pas juste une question générale. Commence ta réponse par UNE SEULE PHRASE COURTE indiquant dans quels cas appeler immédiatement les services d'urgence de son pays (ex: crise qui dure plus de 5 minutes, personne ne reprend pas connaissance, blessure). Ensuite seulement, donne les gestes de premiers secours détaillés.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
        "reponse_label": "RÉPONSE",
    },
    "en": {
        "consigne_langue": "Answer in English, even though the sources below are in French — translate the relevant information yourself.",
        "consigne_base": "You are a health information assistant specialized in neurology. Answer the question ONLY based on the sources provided below. If the information is not present in the sources, clearly say you don't know rather than inventing an answer.\n\nCite the condition's name directly in your sentence (e.g. \"migraine can be treated with...\"). Do NOT use notation like [Source 1] or (Source 2) — links to the sources will be displayed separately after your answer.\n\nAlways state at the end of your answer that this is general information and does not replace the advice of a healthcare professional.",
        "consigne_urgence": "\n\nWARNING: this question seems to describe a situation happening RIGHT NOW, not just a general question. Start your answer with ONE SHORT SENTENCE stating when to call local emergency services immediately (e.g. seizure lasting more than 5 minutes, person not regaining consciousness, injury). Only after that, give the detailed first-aid steps.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
        "reponse_label": "ANSWER",
    },
}


def construire_prompt(question, chunks_retrouves, langue="fr", urgence=False):
    """Assemble le contexte récupéré et la question dans un prompt structuré,
    en contraignant le LLM à ne répondre qu'à partir des sources fournies,
    et dans la langue demandée (les sources elles-mêmes restent en français).
    Si urgence=True, on demande au LLM de prioriser l'information critique
    (quand appeler les secours) en tout début de réponse."""
    textes = INSTRUCTIONS_LANGUE.get(langue, INSTRUCTIONS_LANGUE["fr"])

    contexte = "\n\n".join([
        f"[Source {i+1} — {c['pathologie']} / {c['section']}]\n{c['texte']}"
        for i, c in enumerate(chunks_retrouves)
    ])

    consigne_complete = textes["consigne_base"]
    if urgence:
        consigne_complete += textes["consigne_urgence"]

    return f"""{consigne_complete}

{textes['consigne_langue']}

{textes['sources_label']}:
{contexte}

{textes['question_label']}: {question}

{textes['reponse_label']}:"""


def nettoyer_citations_residuelles(texte_reponse):
    """Filet de sécurité : si le LLM ignore la consigne et laisse quand même des
    mentions de type [Source N] ou (Source N) — y compris avec plusieurs numéros,
    ex: [Source 1, 2] — on les retire proprement. On se limite volontairement aux
    citations ENTRE CROCHETS/PARENTHÈSES : c'est le format que produit presque
    toujours le LLM, et ça évite de casser la grammaire d'une phrase en supprimant
    un mot qui n'est pas entre crochets (ex: 'Voir Source 4' sans crochets)."""
    pattern = r"[\[\(]\s*Sources?\s*\d+(?:\s*(?:,|et|\/)\s*\d+)*\s*[\]\)]"
    texte_nettoye = re.sub(pattern, "", texte_reponse, flags=re.IGNORECASE)

    # Nettoyage des espaces/ponctuation orphelins laissés par la suppression
    texte_nettoye = re.sub(r"\s+([.,;:])", r"\1", texte_nettoye)
    # Fusionne une virgule suivie d'une autre ponctuation (ex: "repos,." -> "repos.")
    texte_nettoye = re.sub(r",\s*([.,;:])", r"\1", texte_nettoye)
    texte_nettoye = re.sub(r"[ \t]{2,}", " ", texte_nettoye)

    return texte_nettoye.strip()


MESSAGES_HORS_SUJET = {
    "fr": "Je ne trouve aucune information pertinente sur ce sujet dans ma base de connaissances (spécialisée en neurologie). Pose-moi plutôt une question sur une pathologie neurologique (migraine, épilepsie, Parkinson, Alzheimer, etc.).",
    "en": "I couldn't find any relevant information on this topic in my knowledge base (specialized in neurology). Try asking me about a neurological condition instead (migraine, epilepsy, Parkinson's, Alzheimer's, etc.).",
}


def generer_reponse(question, pipeline, k=5, max_tokens=800, temperature=0.3, langue="fr"):
    """Pipeline complet : retrieval + génération. Retourne la réponse texte
    et la liste des pathologies utilisées comme sources.
    langue : "fr" ou "en" — la recherche (retrieval) reste en français,
    seule la réponse générée change de langue.
    Si aucun chunk pertinent n'est trouvé (question hors-sujet), on renvoie
    directement un message adapté SANS appeler le LLM et SANS disclaimer
    médical (qui n'aurait pas de sens pour une question hors santé)."""
    chunks_retrouves = pipeline.rechercher(question, k=k)

    if not chunks_retrouves:
        return {
            "reponse": MESSAGES_HORS_SUJET.get(langue, MESSAGES_HORS_SUJET["fr"]),
            "pathologies": [],
            "sources_urls": {},
            "sources_detaillees": []
        }

    urgence = detecter_urgence(question, langue=langue)
    prompt = construire_prompt(question, chunks_retrouves, langue=langue, urgence=urgence)

    reponse = client_groq.chat.completions.create(
        model=MODELE_LLM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    texte_reponse = nettoyer_citations_residuelles(reponse.choices[0].message.content)

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
