import streamlit as st
from backend.pipeline import PipelineRAG
from backend.generation import generer_reponse

st.set_page_config(page_title="Chatbot Neurologie", page_icon="🧠")

TEXTES_INTERFACE = {
    "fr": {
        "titre": "🧠 Chatbot Neurologie",
        "sous_titre": "Assistant d'information santé — ne remplace pas l'avis d'un professionnel",
        "placeholder": "Pose ta question sur une pathologie neurologique...",
        "spinner": "Recherche en cours...",
        "pathologies_label": "Pathologies consultées",
        "selecteur_label": "🌐 Langue / Language",
    },
    "en": {
        "titre": "🧠 Neurology Chatbot",
        "sous_titre": "Health information assistant — does not replace professional medical advice",
        "placeholder": "Ask your question about a neurological condition...",
        "spinner": "Searching...",
        "pathologies_label": "Conditions consulted",
        "selecteur_label": "🌐 Langue / Language",
    },
}


@st.cache_resource
def charger_pipeline():
    """Construit le pipeline RAG une seule fois (mis en cache pour toute la session)."""
    return PipelineRAG("data/corpus_neurologie.json")


pipeline = charger_pipeline()

# --- Sélecteur de langue (dans la barre latérale, visible dès l'arrivée) ---
langue_choisie = st.sidebar.radio(
    "🌐 Langue / Language",
    options=["fr", "en"],
    format_func=lambda x: "Français" if x == "fr" else "English",
)

textes = TEXTES_INTERFACE[langue_choisie]

st.title(textes["titre"])
st.caption(textes["sous_titre"])

# --- Historique de conversation, réinitialisé si on change de langue ---
if "messages" not in st.session_state or st.session_state.get("langue_precedente") != langue_choisie:
    st.session_state.messages = []
    st.session_state.langue_precedente = langue_choisie

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input(textes["placeholder"]):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(textes["spinner"]):
            resultat = generer_reponse(question, pipeline, langue=langue_choisie)
            texte_final = (
                resultat["reponse"]
                + f"\n\n---\n📚 *{textes['pathologies_label']} : {', '.join(resultat['pathologies'])}*"
            )
            st.markdown(texte_final)

    st.session_state.messages.append({"role": "assistant", "content": texte_final})
