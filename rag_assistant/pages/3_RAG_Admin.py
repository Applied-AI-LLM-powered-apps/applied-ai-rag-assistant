import streamlit as st

import json

from utils.auth import check_password
from langchain_community.vectorstores import OpenSearchVectorSearch

from utils.constants import DocumentType, ChunkType, Metadata, CollectionType
from utils.utilsdoc import get_store, empty_store, extract_unique_name, get_collection_count, get_metadatas, delete_documents_by_type_and_name
from utils.utilsfile import list_files, delete_file
from utils.config_loader import load_config


config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""Gestion des connaissances 📄""")

    # collection_name = st.selectbox("Collection", ["Default", "RAG"])

    count = get_collection_count(collection_name)
    if count > 0:
        st.write(f"Il y a **{count}** morceaux (chunks) dans la collection '**{collection_name}**'.")
    else:
        st.write("La collection est vide.")
        st.page_link("pages/2_Load_Document.py", label="Charger les connaissances")

    st.subheader("Fichier(s) chargé(s)")

    unique_filenames = extract_unique_name(collection_name,  Metadata.FILENAME.value)

    for name in unique_filenames:
        st.markdown(f"""- {name}""")

    st.subheader("Sujet(s) disponible(s):")
    unique_topic_names = extract_unique_name(collection_name, Metadata.TOPIC.value)
    for name in unique_topic_names:
        st.markdown(f"""- {name}""")

    # st.subheader("Document Type")
    # unique_document_types = extract_unique_name(collection_name, 'document_type')
    # for name in unique_document_types:
    #     st.markdown(f"""- {name}""")

    with st.form("search"):
        st.subheader("Chercher dans la Base de Connaissance:")
        search = st.text_input("Texte (*)")

        topic_name = st.selectbox("Sujet", unique_topic_names, index=None)
        filename = st.selectbox("Nom du Fichier", unique_filenames, index=None)
        document_type = st.selectbox("Type de Document", [e.value for e in DocumentType], index=None)
        chunk_type = st.selectbox("Type de Morceau", [e.value for e in ChunkType], index=None)
        #document_type = st.selectbox("Document Type", unique_document_types, index=None)

        filters = []
        if filename:
            filters.append({Metadata.FILENAME.value: filename})
        if document_type:
            filters.append({Metadata.DOCUMENT_TYPE.value: document_type})
        if topic_name:
            filters.append({Metadata.TOPIC.value: topic_name})
        if document_type:
            filters.append({Metadata.DOCUMENT_TYPE.value: document_type})
        if chunk_type:
            filters.append({Metadata.CHUNK_TYPE.value: chunk_type})
        if st.form_submit_button("Recherche"):
            # add check for empty string as it is not supported by bedrock (or anthropic?)
            if search != "":
                if len(filters) > 1:
                    where = {"$and": filters}
                elif len(filters) == 1:
                    where = filters[0]
                else:
                    where = {}
                store = get_store()
                if isinstance(store, OpenSearchVectorSearch):
                    result_filters = []
                    for os_filter in filters:
                        for key in os_filter.keys():
                            result_filters.append({"match": {f"metadata.{key}": os_filter[key]}})
                    result = store.similarity_search(search, k=5, boolean_filter=result_filters)
                else:
                    result = store.similarity_search(search, k=5, filter=where)
                st.write(result)
            else:
                st.write("Veuillez entrer un texte.")

    st.subheader("Administration des Données")

    col1, col2 = st.columns(2)
    with col1:
        file_name_to_delete = st.selectbox("Choisir un fichier", unique_filenames, index=None)
        if st.button("Supprimer les données du fichier"):
            delete_documents_by_type_and_name(collection_name=collection_name, type=Metadata.FILENAME.value, name=file_name_to_delete)
            delete_file(file_name_to_delete, CollectionType.DOCUMENTS.value)

        chunk_type_to_delete = st.selectbox("Choisir un type de morceau (chunk)", [e.value for e in ChunkType], index=None)
        if st.button("Supprimer les données de ce type"):
            delete_documents_by_type_and_name(collection_name=collection_name, type=Metadata.CHUNK_TYPE.value,
                                              name=chunk_type_to_delete)

    with col2:
        topic_name_to_delete = st.selectbox("Choisir un sujet", unique_topic_names, index=None)
        if st.button("Supprimer les données de ce sujet"):
            delete_documents_by_type_and_name(collection_name=collection_name, type=Metadata.TOPIC.value, name=topic_name_to_delete)

    if st.button("Supprimer la collection"):
        empty_store(collection_name=collection_name)

    with st.expander("Voir toutes les meta-données", expanded=False):
        st.subheader("Méta-données")
        metadatas = get_metadatas(collection_name=collection_name)
        st.code(json.dumps(metadatas, indent=4, sort_keys=True), language="json")


if __name__ == "__main__":
    if not check_password():
        # Do not continue if check_password is not True.
        st.stop()
    main()
