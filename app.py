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
        "drapeau": "🇫🇷",
    },
    "en": {
        "titre": "Neurology Chatbot",
        "sous_titre": "Health information assistant — does not replace professional medical advice",
        "placeholder": "Ask your question about a neurological condition...",
        "spinner": "Searching...",
        "pathologies_label": "Sources consulted",
        "vider_historique": "🗑️ Clear history",
        "drapeau": "🇬🇧",
    },
}

# --- Style personnalisé ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1420 0%, #131826 100%);
    }
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #7C9EFF 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.2rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 8px;
    }
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(124, 158, 255, 0.08);
    }
    .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: rgba(167, 139, 250, 0.06);
    }
    section[data-testid="stSidebar"] {
        background-color: #0d111c;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    div[data-baseweb="segmented-control"] {
        margin-top: 0.3rem;
    }
    .badge-langue {
        display: inline-block;
        font-size: 0.75rem;
        opacity: 0.55;
        margin-bottom: 2px;
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

    textes_bouton = TEXTES_INTERFACE[langue_choisie]
    if st.button(textes_bouton["vider_historique"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

textes = TEXTES_INTERFACE[langue_choisie]

st.title(textes["titre"])
st.caption(textes["sous_titre"])

# --- Historique persistant, PAS réinitialisé au changement de langue ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Petit badge discret indiquant la langue dans laquelle ce message a été échangé,
        # utile puisque l'historique peut désormais mélanger français et anglais.
        st.markdown(
            f"<span class='badge-langue'>{msg.get('drapeau', '')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])

if question := st.chat_input(textes["placeholder"]):
    st.session_state.messages.append({
        "role": "user", "content": question, "drapeau": textes["drapeau"]
    })
    with st.chat_message("user"):
        st.markdown(f"<span class='badge-langue'>{textes['drapeau']}</span>", unsafe_allow_html=True)
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(textes["spinner"]):
            resultat = generer_reponse(question, pipeline, langue=langue_choisie)

            texte_final = resultat["reponse"]

            # On n'ajoute la section "Sources consultées" que s'il y a réellement
            # des sources (pas pour les réponses hors-sujet, où sources_urls est vide)
            if resultat["sources_urls"]:
                liens_sources = ", ".join(
                    f"[{nom}]({url})" for nom, url in resultat["sources_urls"].items()
                )
                texte_final += f"\n\n---\n📚 *{textes['pathologies_label']} : {liens_sources}*"

            st.markdown(f"<span class='badge-langue'>{textes['drapeau']}</span>", unsafe_allow_html=True)
            st.markdown(texte_final)

    st.session_state.messages.append({
        "role": "assistant", "content": texte_final, "drapeau": textes["drapeau"]
    })
