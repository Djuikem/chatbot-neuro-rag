import re
import streamlit as st
from backend.pipeline import PipelineRAG
from backend.generation import generer_reponse

st.set_page_config(page_title="Chatbot Neurologie", page_icon="🧠", layout="centered")

TEXTES_INTERFACE = {
    "fr": {
        "titre": "Chatbot Neurologie",
        "sous_titre": "Assistant d'information santé — ne remplace pas l'avis d'un professionnel",
        "placeholder": "Pose ta question sur une pathologie neurologique...",
        "spinner": "Recherche en cours...",
        "pathologies_label": "Sources consultées",
        "vider_historique": "🗑️ Vider l'historique",
    },
    "en": {
        "titre": "Neurology Chatbot",
        "sous_titre": "Health information assistant — does not replace professional medical advice",
        "placeholder": "Ask your question about a neurological condition...",
        "spinner": "Searching...",
        "pathologies_label": "Sources consulted",
        "vider_historique": "🗑️ Clear history",
    },
}

# --- Style personnalisé : utilise les variables de thème de Streamlit pour
# s'adapter automatiquement au mode clair/sombre choisi par l'utilisateur
# (menu ⋮ en haut à droite -> Settings -> Choose app theme), plutôt que
# d'imposer des couleurs fixes qui casseraient le mode clair. ---
st.markdown("""
<style>
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 8px;
    }
    div[data-baseweb="segmented-control"] {
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def charger_pipeline():
    """Construit le pipeline RAG une seule fois (mis en cache pour toute la session)."""
    return PipelineRAG("data/corpus_neurologie.json")


pipeline = charger_pipeline()

# --- Sélecteur de langue façon "pilules" dans la barre latérale ---
with st.sidebar:
    st.markdown("### 🧠 Neuro-RAG")
    st.caption("Français / English")

    langue_choisie = st.segmented_control(
        label="Langue",
        options=["fr", "en"],
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇬🇧 English",
        default="fr",
        label_visibility="collapsed",
    )
    langue_choisie = langue_choisie or "fr"

    st.divider()
    st.caption("🎨 Le thème clair/sombre se change dans le menu ⋮ en haut à droite → Settings.")

    textes_bouton = TEXTES_INTERFACE[langue_choisie]
    if st.button(textes_bouton["vider_historique"], use_container_width=True):
        st.session_state.messages = [
            m for m in st.session_state.get("messages", []) if m.get("langue") != langue_choisie
        ]
        st.rerun()

textes = TEXTES_INTERFACE[langue_choisie]

st.title(textes["titre"])
st.caption(textes["sous_titre"])

if "messages" not in st.session_state:
    st.session_state.messages = []

messages_a_afficher = [m for m in st.session_state.messages if m.get("langue") == langue_choisie]

for msg in messages_a_afficher:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def extraire_contenu_brut(texte_affiche):
    """Retire la section '---\\n📚 Sources...' du texte affiché, pour ne
    transmettre que le contenu réellement dit par l'assistant comme mémoire
    de conversation au LLM (les liens de sources n'apportent rien au contexte
    conversationnel et alourdiraient inutilement le prompt)."""
    return re.split(r"\n\n---\n📚", texte_affiche)[0].strip()


if question := st.chat_input(textes["placeholder"]):
    st.session_state.messages.append({
        "role": "user", "content": question, "langue": langue_choisie
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(textes["spinner"]):
            # On construit l'historique (mémoire conversationnelle) à partir
            # des messages déjà affichés dans cette langue, nettoyés des liens
            # de sources, et sans le message qu'on vient d'ajouter.
            historique_pour_llm = [
                {"role": m["role"], "content": extraire_contenu_brut(m["content"])}
                for m in messages_a_afficher
            ]

            resultat = generer_reponse(
                question, pipeline, historique=historique_pour_llm, langue=langue_choisie
            )

            texte_final = resultat["reponse"]

            if resultat["sources_urls"]:
                liens_sources = ", ".join(
                    f"[{nom}]({url})" for nom, url in resultat["sources_urls"].items()
                )
                texte_final += f"\n\n---\n📚 *{textes['pathologies_label']} : {liens_sources}*"

            st.markdown(texte_final)

    st.session_state.messages.append({
        "role": "assistant", "content": texte_final, "langue": langue_choisie
    })
