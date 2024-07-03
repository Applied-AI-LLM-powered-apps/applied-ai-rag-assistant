import streamlit as st
import sys

from utils.auth import check_password
from utils.config_loader import load_config
from utils.utilsllm import get_model_provider, get_model_name, get_embeddings_model_name

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
vectordb = config['VECTORDB']['vectordb']

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

model_provider = get_model_provider()
model_name = get_model_name(provider=model_provider)
embeddings_model = get_embeddings_model_name(provider=model_provider)

def main():
    st.title(f"""Bienvenue sur {app_name} ! 👋""")

    st.markdown(
        f"""
        **{app_name}** utilise '**{model_provider}**' avec comme _modèle de langage_ '**{model_name}**'.
        
        La _base de connaissance_ est sur '**{vectordb}**' avec comme _modèle d'embedding_ : '**{embeddings_model}**'.
    """
    )
    st.write(sys.version)


if __name__ == "__main__":

    if not check_password():
        # Do not continue if check_password is not True.
        st.stop()

    main()
