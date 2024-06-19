import streamlit as st
import os

from llama_index.core import SummaryIndex, VectorStoreIndex, StorageContext, load_index_from_storage, \
    get_response_synthesizer, DocumentSummaryIndex, SimpleDirectoryReader
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import Document as LIDocument

from utils.auth import check_password
from utils.constants import DocumentType, SupportedFileType, Metadata
from utils.config_loader import load_config
from utils.utilsdoc import load_doc, load_store, get_child_chunk_size
from utils.utilsllm import load_llamaindex_model

from utils.utilsvision import load_image

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']
LLAMA_INDEX_ROOT_DIR = config["LLAMA_INDEX"]["LLAMA_INDEX_ROOT_DIR"]
SUMMARY_INDEX_DIR = config["LLAMA_INDEX"]["SUMMARY_INDEX_DIR"]
summary_index_folder = f"{LLAMA_INDEX_ROOT_DIR}/{SUMMARY_INDEX_DIR}"

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
        generate_summary = st.checkbox("Ajouter au sommaire")

    if st.button("Transmettre", disabled=disabled):

        upload_files(analyse_images, file_type, generate_summary, image_only, pdfs, restart_image_analysis, topic_name)


def upload_files(analyse_images, file_type, generate_summary, image_only, pdfs, restart_image_analysis, topic_name):
    file_paths = []
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    for pdf in pdfs:
        file_path = os.path.join(upload_directory, pdf.name)
        with open(file_path, 'wb') as f:
            f.write(pdf.read())
        file_paths.append(file_path)
    metadata = {Metadata.DOCUMENT_TYPE.value: file_type, Metadata.TOPIC.value: topic_name}
    docs = []
    if not image_only:
        docs += load_doc(pdfs, metadata)
    if analyse_images:
        image_docs = load_image(pdfs, metadata, restart_image_analysis)
        docs += image_docs
    load_store(docs, collection_name=collection_name)
    if generate_summary:
        docs_li = docs_prepare(file_paths)
        summary_index = build_llama_index(docs_li)


def docs_prepare(file_paths: list[str]) -> list[LIDocument]:
    documents = SimpleDirectoryReader(
        input_files=file_paths,
        #input_dir=directory,
        required_exts=["."+e.value for e in SupportedFileType]
    ).load_data()
    return documents


def build_llama_index(docs: list[LIDocument]) -> BaseIndex:
    llm = load_llamaindex_model()
    # embeddings = load_llamaindex_embeddings()
    # splitter = SentenceSplitter(chunk_size=1024)
    # nodes = splitter.get_nodes_from_documents(docs_prepare)
    # summary_index = SummaryIndex(nodes)
    #
    #
    # summary_query_engine = summary_index.as_query_engine(
    #     response_mode="tree_summarize",
    #     use_async=True,
    #     llm=llm_prepare
    # )

    splitter = SentenceSplitter(chunk_size=get_child_chunk_size())
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        llm=llm,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    doc_summary_index.storage_context.persist(summary_index_folder)
    storage_context = StorageContext.from_defaults(persist_dir=summary_index_folder)
    #doc_summary_index = load_index_from_storage(storage_context)
    return doc_summary_index
    # summary_tool = QueryEngineTool.from_defaults(
    #     query_engine=summary_query_engine,
    #     description=(
    #         f"Use ONLY IF you want to get a holistic summary of {topic}."
    #         f"Do NOT use if you have specific questions on {topic}."
    #     ),
    # )

    # vector_index = VectorStoreIndex(nodes, embed_model=embeddings_prepare)
    # vector_query_engine = vector_index.as_query_engine(llm=llm_prepare)
    #
    # vector_tool = QueryEngineTool.from_defaults(
    #     query_engine=vector_query_engine,
    #     description=(
    #         f"Useful for retrieving specific questions over {topic}."
    #     ),
    # )

    # query_engine = RouterQueryEngine(
    #     selector=LLMSingleSelector.from_defaults(),
    #     query_engine_tools=[
    #         summary_tool,
    #         vector_tool,
    #     ],
    #     verbose=True
    # )
    # query_engine = create_sentence_window_engine(
    #     docs_prepare,
    # )


if __name__ == "__main__":
    if not check_password():
        # Do not continue if check_password is not True.
        st.stop()
    main()
