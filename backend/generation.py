"""
Génération de réponse : détection de salutations/urgence, construction des
messages de conversation (avec mémoire des échanges précédents) et appel au
LLM via l'API Groq.
"""

import os
import re
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELE_LLM = "openai/gpt-oss-120b"


# --- Détection de salutations pures (bonjour, hello, salut...) ---

MOTS_SALUTATION = {
    "fr": ["bonjour", "bonsoir", "salut", "coucou", "hello", "bjr", "yo"],
    "en": ["hello", "hi", "hey", "good morning", "good evening", "yo"],
}

MESSAGES_ACCUEIL = {
    "fr": "Bonjour et bienvenue ! 😊 Je suis un assistant d'information santé spécialisé en neurologie. Pose-moi une question sur une pathologie comme la migraine, l'épilepsie, la maladie de Parkinson, Alzheimer, la sclérose en plaques, et bien d'autres — je ferai de mon mieux pour t'aider.",
    "en": "Hello and welcome! 😊 I'm a health information assistant specialized in neurology. Ask me a question about a condition like migraine, epilepsy, Parkinson's disease, Alzheimer's, multiple sclerosis, and more — I'll do my best to help.",
}


def detecter_salutation(question, langue="fr"):
    """Détecte si le message est une salutation PURE (rien d'autre à côté),
    ex: 'bonjour', 'hello !', 'salut ça va ?' — pour répondre chaleureusement
    sans passer par le retrieval/LLM (inutile pour une simple salutation)."""
    texte = question.lower().strip()
    texte = re.sub(r"[!?.,;:]+$", "", texte).strip()
    mots_cles = MOTS_SALUTATION.get(langue, MOTS_SALUTATION["fr"])

    # Salutation pure : le message commence par un mot de salutation ET reste très court
    # (moins de 5 mots), pour ne pas déclencher ce cas sur "bonjour, quels sont les
    # symptômes de la migraine ?" qui contient une vraie question à traiter normalement.
    commence_par_salutation = any(texte.startswith(mot) for mot in mots_cles)
    return commence_par_salutation and len(texte.split()) <= 5


# --- Détection d'urgence (situation vécue en direct) ---

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


# --- Instructions système (persona + règles) ---

INSTRUCTIONS_LANGUE = {
    "fr": {
        "consigne_langue": "Réponds en français.",
        "consigne_base": (
            "Tu es un assistant d'information santé spécialisé en neurologie, "
            "avec un ton chaleureux et accueillant. Réponds à la question "
            "UNIQUEMENT à partir des sources fournies dans le message utilisateur. "
            "Si l'information n'est pas présente dans les sources, dis clairement "
            "que tu ne sais pas plutôt que d'inventer une réponse.\n\n"
            "Cite le nom de la pathologie directement dans ta phrase "
            "(ex: \"la migraine peut être traitée par...\"). N'utilise PAS de "
            "notation du type [Source 1] ou (Source 2) — les liens vers les "
            "sources seront affichés séparément après ta réponse.\n\n"
            "IMPORTANT sur la façon de parler des sources : ces documents "
            "viennent de TA propre base de connaissances, PAS de l'utilisateur. "
            "N'écris donc JAMAIS des formulations comme \"les sources que vous "
            "avez fournies/partagées\" ou \"les documents que vous m'avez donnés\" "
            "— cela laisse croire à tort que c'est l'utilisateur qui a apporté "
            "ces informations. Utilise plutôt des formulations comme \"d'après "
            "les informations disponibles\", \"selon ma base de connaissances\", "
            "ou simplement \"le traitement de la migraine comprend...\" sans "
            "mentionner explicitement l'existence de sources dans la phrase.\n\n"
            "Tu peux t'appuyer sur les échanges précédents de la conversation "
            "pour comprendre le contexte d'une question de suivi (ex: si "
            "l'utilisateur a demandé les symptômes de la migraine puis demande "
            "'et les traitements ?', comprends qu'il parle toujours de la migraine).\n\n"
            "Précise systématiquement à la fin de ta réponse que ceci est une "
            "information générale et ne remplace pas l'avis d'un professionnel de santé."
        ),
        "consigne_urgence": "\n\nATTENTION : cette question semble décrire une situation vécue EN CE MOMENT, pas juste une question générale. Commence ta réponse par UNE SEULE PHRASE COURTE indiquant dans quels cas appeler immédiatement les services d'urgence de son pays (ex: crise qui dure plus de 5 minutes, personne ne reprend pas connaissance, blessure). Ensuite seulement, donne les gestes de premiers secours détaillés.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
    },
    "en": {
        "consigne_langue": "Answer in English, even though the sources may be in French — translate the relevant information yourself.",
        "consigne_base": (
            "You are a health information assistant specialized in neurology, "
            "with a warm and welcoming tone. Answer the question ONLY based on "
            "the sources provided in the user message. If the information is "
            "not present in the sources, clearly say you don't know rather than "
            "inventing an answer.\n\n"
            "Cite the condition's name directly in your sentence (e.g. "
            "\"migraine can be treated with...\"). Do NOT use notation like "
            "[Source 1] or (Source 2) — links to the sources will be displayed "
            "separately after your answer.\n\n"
            "IMPORTANT about how to refer to the sources: these documents come "
            "from YOUR OWN knowledge base, NOT from the user. Never write "
            "phrasings like \"the sources you provided/shared\" or \"the "
            "documents you gave me\" — this wrongly implies the user supplied "
            "this information. Instead use phrasings like \"based on the "
            "available information\", \"according to my knowledge base\", or "
            "simply state the fact directly (e.g. \"migraine treatment "
            "includes...\") without mentioning sources explicitly in the sentence.\n\n"
            "You can rely on the previous turns of the conversation to "
            "understand follow-up questions (e.g. if the user asked about "
            "migraine symptoms and then asks 'and treatments?', understand "
            "they're still talking about migraine).\n\n"
            "Always state at the end of your answer that this is general "
            "information and does not replace the advice of a healthcare professional."
        ),
        "consigne_urgence": "\n\nWARNING: this question seems to describe a situation happening RIGHT NOW, not just a general question. Start your answer with ONE SHORT SENTENCE stating when to call local emergency services immediately (e.g. seizure lasting more than 5 minutes, person not regaining consciousness, injury). Only after that, give the detailed first-aid steps.",
        "sources_label": "SOURCES",
        "question_label": "QUESTION",
    },
}


def construire_messages_chat(question, chunks_retrouves, historique, langue="fr", urgence=False):
    """Construit la liste de messages (format chat multi-tours) envoyée au LLM :
    1. Un message système avec les instructions et la persona
    2. L'historique des échanges précédents (pour la mémoire conversationnelle)
    3. Le tour actuel : les sources retrouvées + la question posée

    historique : liste de dicts {"role": "user"|"assistant", "content": str},
    dans l'ordre chronologique, SANS les liens de sources (juste le texte dit)."""
    textes = INSTRUCTIONS_LANGUE.get(langue, INSTRUCTIONS_LANGUE["fr"])

    consigne_complete = textes["consigne_base"] + "\n\n" + textes["consigne_langue"]
    if urgence:
        consigne_complete += textes["consigne_urgence"]

    messages = [{"role": "system", "content": consigne_complete}]

    # On ajoute l'historique récent (déjà limité en amont par l'appelant)
    for tour in historique:
        messages.append({"role": tour["role"], "content": tour["content"]})

    contexte = "\n\n".join([
        f"[Source {i+1} — {c['pathologie']} / {c['section']}]\n{c['texte']}"
        for i, c in enumerate(chunks_retrouves)
    ])

    tour_actuel = f"{textes['sources_label']}:\n{contexte}\n\n{textes['question_label']}: {question}"
    messages.append({"role": "user", "content": tour_actuel})

    return messages


def nettoyer_citations_residuelles(texte_reponse):
    """Filet de sécurité : si le LLM ignore la consigne et laisse quand même des
    mentions de type [Source N] ou (Source N) — y compris avec plusieurs numéros,
    ex: [Source 1, 2] — on les retire proprement. On se limite volontairement aux
    citations ENTRE CROCHETS/PARENTHÈSES : c'est le format que produit presque
    toujours le LLM, et ça évite de casser la grammaire d'une phrase en supprimant
    un mot qui n'est pas entre crochets (ex: 'Voir Source 4' sans crochets)."""
    pattern = r"[\[\(]\s*Sources?\s*\d+(?:\s*(?:,|et|\/)\s*\d+)*\s*[\]\)]"
    texte_nettoye = re.sub(pattern, "", texte_reponse, flags=re.IGNORECASE)

    texte_nettoye = re.sub(r"\s+([.,;:])", r"\1", texte_nettoye)
    texte_nettoye = re.sub(r",\s*([.,;:])", r"\1", texte_nettoye)
    texte_nettoye = re.sub(r"[ \t]{2,}", " ", texte_nettoye)

    return texte_nettoye.strip()


def corriger_attribution_sources(texte_reponse):
    """Filet de sécurité : si le LLM dit malgré la consigne que les sources
    viennent de l'utilisateur ('les sources que vous avez fournies/partagées',
    'the sources you provided/shared'...), on corrige vers une formulation
    neutre plutôt que de laisser cette confusion trompeuse dans la réponse."""
    remplacements = [
        (r"les (?:sources|documents|informations) que (?:vous|tu) (?:avez|as) (?:fournies?|partagées?|données?|donnés?)",
         "les informations disponibles"),
        (r"the sources (?:you|that you) (?:provided|shared|gave me)",
         "the available information"),
        (r"the documents (?:you|that you) (?:provided|shared|gave me)",
         "the available information"),
    ]
    texte_corrige = texte_reponse
    for motif, remplacement in remplacements:
        texte_corrige = re.sub(motif, remplacement, texte_corrige, flags=re.IGNORECASE)
    return texte_corrige


MESSAGES_HORS_SUJET = {
    "fr": "Je ne trouve aucune information pertinente sur ce sujet dans ma base de connaissances (spécialisée en neurologie). Pose-moi plutôt une question sur une pathologie neurologique (migraine, épilepsie, Parkinson, Alzheimer, etc.).",
    "en": "I couldn't find any relevant information on this topic in my knowledge base (specialized in neurology). Try asking me about a neurological condition instead (migraine, epilepsy, Parkinson's, Alzheimer's, etc.).",
}


def _construire_requete_retrieval(question, historique):
    """Pour les questions de suivi très courtes/ambiguës (ex: 'et les traitements ?'),
    on enrichit la requête envoyée au retrieval avec la dernière question de
    l'utilisateur, pour aider à retrouver la bonne pathologie malgré l'absence
    de mots-clés explicites dans la nouvelle question."""
    if len(question.split()) > 6 or not historique:
        return question

    derniere_question_utilisateur = None
    for tour in reversed(historique):
        if tour["role"] == "user":
            derniere_question_utilisateur = tour["content"]
            break

    if derniere_question_utilisateur:
        return f"{derniere_question_utilisateur} {question}"
    return question


def generer_reponse(question, pipeline, historique=None, k=5, max_tokens=800,
                     temperature=0.3, langue="fr"):
    """Pipeline complet : retrieval + génération, avec mémoire conversationnelle.

    historique : liste de dicts {"role": "user"|"assistant", "content": str}
    représentant les échanges précédents DE LA CONVERSATION ACTUELLE (déjà
    filtrés par langue et nettoyés des liens de sources par l'appelant).
    Sert à la fois à donner de la mémoire au LLM et à enrichir le retrieval
    pour les questions de suivi courtes.
    """
    historique = historique or []

    # Cas 1 : salutation pure -> réponse chaleureuse directe, sans retrieval/LLM
    if detecter_salutation(question, langue=langue):
        return {
            "reponse": MESSAGES_ACCUEIL.get(langue, MESSAGES_ACCUEIL["fr"]),
            "pathologies": [],
            "sources_urls": {},
            "sources_detaillees": []
        }

    requete_retrieval = _construire_requete_retrieval(question, historique)
    chunks_retrouves = pipeline.rechercher(requete_retrieval, k=k)

    # Cas 2 : rien de pertinent trouvé -> message hors-sujet direct
    if not chunks_retrouves:
        return {
            "reponse": MESSAGES_HORS_SUJET.get(langue, MESSAGES_HORS_SUJET["fr"]),
            "pathologies": [],
            "sources_urls": {},
            "sources_detaillees": []
        }

    urgence = detecter_urgence(question, langue=langue)

    # On ne garde que les derniers échanges pour ne pas alourdir le prompt
    # indéfiniment au fil d'une longue conversation (6 derniers messages = 3 tours)
    historique_recent = historique[-6:]

    messages = construire_messages_chat(
        question, chunks_retrouves, historique_recent, langue=langue, urgence=urgence
    )

    reponse = client_groq.chat.completions.create(
        model=MODELE_LLM,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    texte_reponse = nettoyer_citations_residuelles(reponse.choices[0].message.content)
    texte_reponse = corriger_attribution_sources(texte_reponse)

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
