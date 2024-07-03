from typing import Optional

import streamlit as st
import os
import io

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document as LIDocument

from utils.auth import check_password
from utils.constants import DocumentType, SupportedFileType, Metadata, CollectionType
from utils.config_loader import load_config
from utils.utilsdoc import load_doc, load_store
from utils.utilsrag_li import build_summary_index

from utils.utilsvision import load_image
from utils.utilsfile import put_file

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""Chargement des Connaissances 📄""")

    # with st.form("Upload File"):
    topic_name = st.text_input("Thème du document (ex: API, Cloud, Data, Architecture, Sécurité, ...)")

    file_type = st.radio("Type de document", [e.value for e in DocumentType], index=None)

    pdfs = st.file_uploader("Document(s) à transmettre", type=[e.value for e in SupportedFileType],
                            accept_multiple_files=True)

    disabled = True
    if (file_type is not None) and (topic_name is not None) and (pdfs is not None) and (len(pdfs)):
        disabled = False


    with st.container():
        st.subheader("Traitement des images")
        analyse_images = st.checkbox("Analyser les images")
        image_only = st.checkbox("Traiter uniquement les images (test mode)", disabled=(not analyse_images))
        restart_image_analysis = st.checkbox("Relancer l'analyse d'image (test mode)", disabled=(not analyse_images))

    with st.container():
        st.subheader("Autres options")
        generate_summary = st.checkbox("Générer le sommaire", disabled=True)
        upload_only = st.checkbox("Enregistrement des documents uniquement")

    if st.button("Transmettre", disabled=disabled):


        upload_files(analyse_images, file_type, generate_summary, image_only, pdfs, restart_image_analysis, topic_name, upload_only)


def upload_files(analyse_images, file_type, generate_summary, image_only, pdfs, restart_image_analysis, topic_name, upload_only):
    file_paths = []
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    for pdf in pdfs:
        file_path = os.path.join(upload_directory, pdf.name)
        with open(file_path, 'wb') as f:
            f.write(pdf.read())
        file_paths.append(file_path)
        put_file(io.BytesIO(pdf.getvalue()), pdf.name, CollectionType.DOCUMENTS.value)
    metadata = {Metadata.DOCUMENT_TYPE.value: file_type, Metadata.TOPIC.value: topic_name}
    docs = []
    if not image_only:
        docs += load_doc(pdfs, metadata)
    if analyse_images:
        image_docs = load_image(pdfs, metadata, restart_image_analysis)
        docs += image_docs
    if not upload_only:
        load_store(docs, collection_name=collection_name)
    if generate_summary:
        docs_li = docs_prepare(
            #input_files=file_paths,
            input_dir=upload_directory
        )
        summary_index = build_summary_index(docs_li)


def docs_prepare(input_files: Optional[list[str]] = None, input_dir: Optional[str] = None) -> list[LIDocument]:
    documents = SimpleDirectoryReader(
        input_files=input_files,
        input_dir=input_dir,
        required_exts=["."+e.value for e in SupportedFileType]
    ).load_data()
    return documents


if __name__ == "__main__":
    if not check_password():
        # Do not continue if check_password is not True.
        st.stop()
    main()
