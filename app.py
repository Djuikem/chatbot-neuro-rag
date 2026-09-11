import streamlit as st
from backend.pipeline import PipelineRAG
from backend.generation import generer_reponse

st.set_page_config(page_title="Chatbot Neurologie", page_icon="🧠")


@st.cache_resource
def charger_pipeline():
    """Construit le pipeline RAG une seule fois (mis en cache pour toute la session)."""
    return PipelineRAG("data/corpus_neurologie.json")


pipeline = charger_pipeline()

st.title("🧠 Chatbot Neurologie")
st.caption("Assistant d'information santé — ne remplace pas l'avis d'un professionnel")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Pose ta question sur une pathologie neurologique..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            resultat = generer_reponse(question, pipeline)
            texte_final = (
                resultat["reponse"]
                + f"\n\n---\n📚 *Pathologies consultées : {', '.join(resultat['pathologies'])}*"
            )
            st.markdown(texte_final)

    st.session_state.messages.append({"role": "assistant", "content": texte_final})
