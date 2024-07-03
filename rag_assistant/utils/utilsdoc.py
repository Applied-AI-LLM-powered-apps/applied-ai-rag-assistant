import io
import os
import re
import shutil
import uuid

from pathlib import Path
from typing import Union, Optional


import boto3
import chromadb

from PyPDF2 import PdfReader

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS, Chroma, OpenSearchVectorSearch

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import re

import chromadb
from langchain_community.vectorstores import Chroma
import uuid


from .utilsllm import load_embeddings
from .config_loader import load_config
from .constants import Metadata, ChunkType
from requests_aws4auth import AWS4Auth
from botocore.session import Session
from opensearchpy import RequestsHttpConnection, exceptions


config = load_config()
child_chunk_size = int(config['KNOWLEDGE']['CHILD_CHUNK_SIZE'])
child_chunk_overlap = int(config['KNOWLEDGE']['CHILD_CHUNK_OVERLAP'])

parent_chunk_size = int(config['KNOWLEDGE']['PARENT_CHUNK_SIZE'])
parent_chunk_overlap = int(config['KNOWLEDGE']['PARENT_CHUNK_OVERLAP'])


def get_child_chunk_size() -> int:
    return child_chunk_size


def get_child_chunk_overlap() -> int:
    return child_chunk_overlap


def extract_unique_name(collection_name : str, key : str):
    metadatas = get_metadatas(collection_name=collection_name)

    unique_names = set()
    for item in metadatas:
        if item is not None and key in item:
            unique_names.add(item[key])
    return unique_names


def get_metadatas(collection_name : str):
    store = get_store(collection_name=collection_name)
    if isinstance(store, OpenSearchVectorSearch):
        client = store.client
        index_name = collection_name.lower()
        try:
            response = client.search(
                index=index_name,
                scroll='2m',  # Keep the search context alive for 2 minutes
                size=100,  # Adjust the size per page
                body={
                    "query": {
                        "match_all": {}
                    },
                    "_source": {
                        "includes": ["metadata"]
                    }
                }
            )

            # List to hold all results
            all_results = response['hits']['hits']

            # Use the scroll API to fetch all results
            while True:
                scroll_id = response['_scroll_id']
                response = client.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )
                
                # Break the loop if no more results
                if not response['hits']['hits']:
                    break
                
                all_results.extend(response['hits']['hits'])
            metadatas = [doc['_source']['metadata'] for doc in all_results]
        except exceptions.NotFoundError:
            metadatas = []
    elif isinstance(store, Chroma):
        collection = store._collection
        metadatas = collection.get()['metadatas']
    else:
        return {}

    return metadatas


def delete_documents_by_type_and_name(collection_name: str, type: str, name: str):
    if type not in [Metadata.FILENAME.value, Metadata.TOPIC.value, Metadata.CHUNK_TYPE.value]:
        raise ValueError(f"Type {type} not supported for deletion")

    store = get_store(collection_name=collection_name)
    if isinstance(store, OpenSearchVectorSearch):
        client = store.client
        index_name = collection_name.lower()

        response = client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"match": {f"metadata.{type}": name}},
                        ]
                    }
                }
            }
        )
    else:
        collection = store._collection
        collection.delete(where={f"{type}": {"$eq": f"{name}"}})


def split_documents(documents: list[Document]) -> list[Document]:
    # Initialize text splitter
    text_splitter = TokenTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    return chunks


def _chunk_texts(texts: str) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )
    character_split_texts = character_splitter.split_text(texts)

    # token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    # token_split_texts = []
    # for text in character_split_texts:
    #     token_split_texts += token_splitter.split_text(text)

    return character_split_texts


# def process_txt_folder(txt_input_folder_name, txt_folder_name):
#     # Initialize tokenizer
#     # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#
#     data = []
#     sources = []
#
#     text_splitter = TokenTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap)
#
#     # Iterate over all files in the folder
#     for filename in os.listdir(txt_input_folder_name):
#         # Only process PDF files
#         if filename.endswith(".txt"):
#             # Full path to the file
#             filepath = os.path.join(txt_input_folder_name, filename)
#
#             # Write the extracted text to a .txt file
#             txt_filename = filename
#             txt_filepath = os.path.join(txt_folder_name, txt_filename)
#             path = Path(txt_filepath)
#             if not path.is_file():
#                 print("Storing text:", txt_filename)
#                 shutil.copy(filepath, txt_filepath)
#
#             # Read the .txt file
#             with open(txt_filepath, 'r') as f:
#                 data.append(f.read())
#             sources.append(filename)
#
#         # Here we split the documents, as needed, into smaller chunks.
#         # We do this due to the context limits of the LLMs.
#         docs = []
#         metadatas = []
#         for i, d in enumerate(data):
#             splits = text_splitter.create_documents(d)
#             docs.extend(splits)
#             metadatas.extend([{"source": sources[i]}] * len(splits))
#
#     # Return the array of chunks
#     return docs, metadatas


def empty_store(collection_name="Default") -> None:

    vectordb = config['VECTORDB']['vectordb']

    if vectordb == "faiss":
        raise NotImplementedError(f"Sorry, empty_store for {vectordb} not implemented yet!")

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']
        persistent_client = chromadb.PersistentClient(path=persist_directory)

        persistent_client.delete_collection(name=collection_name)

    elif vectordb == "opensearch":
        credentials = Session().get_credentials()
        aws_region = config.get('VECTORDB', 'opensearch_aws_region')

        awsauth = AWS4Auth(region=aws_region, service='es',
                    refreshable_credentials=credentials)

        opensearch_url = config.get('VECTORDB', 'opensearch_url')
        # Index name should be lowercase
        index_name = collection_name.lower()

        db = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            embedding_function=load_embeddings(),
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=index_name,
        )
        db.delete_index(index_name)
    else:
        raise NotImplementedError(f"{vectordb} empty_store not implemented yet")

    pass


def load_store(documents: list[Document], embeddings: Embeddings = None, collection_name=None, split=True) -> VectorStoreIndexWrapper:

    vectordb = config['VECTORDB']['vectordb']

    # Store embeddings to vector db
    if split:
        documents = split_documents(documents)

    if not embeddings:
        embeddings = load_embeddings()

    if not collection_name:
        collection_name = config['VECTORDB']['collection_name']

    db = None

    if vectordb == "faiss":

        index_dir = config['VECTORDB']['faiss_persist_directory']

        db = FAISS.from_documents(documents, embeddings)

        ## This is generating this error:
        ## TypeError: cannot pickle '_thread.RLock' object
        ## so moving to FAISS.load_local and
        # index_file = f"{index_dir}/docs.index"
        # pkl_file = f"{index_dir}/faiss_store.pkl"
        # faiss.write_index(db.index, index_file)
        # db.index = None
        # with open(pkl_file, "wb") as f:
        #    pickle.dump(db, f)
        db.save_local(index_dir)

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']

        persistent_client = chromadb.PersistentClient(path=persist_directory)
        collection = persistent_client.get_or_create_collection(name=collection_name)

        # # Create a list of unique ids for each document based on the content
        # ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
        # unique_ids = list(set(ids))
        # # print(f"""Unique Ids : {unique_ids}""")
        #
        # # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        # seen_ids = set()
        # unique_docs = [doc for doc, id in zip(documents, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

        # Create a bag (dict) for unique ids for each document based on the content
        id_bag = {}
        unique_docs = []
        for doc in documents:
            id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
            if id not in id_bag:
                id_bag[id] = True
                unique_docs.append(doc)

        # Get unique ids
        unique_ids = list(id_bag.keys())

        db = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            ids=unique_ids,
            client=persistent_client,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        # Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
        # db.persist()
    elif vectordb == "opensearch":
        credentials = Session().get_credentials()
        aws_region = config.get('VECTORDB', 'opensearch_aws_region')

        awsauth = AWS4Auth(region=aws_region, service='es',
                    refreshable_credentials=credentials)

        opensearch_url = config.get('VECTORDB', 'opensearch_url')
        # Index name should be lowercase
        index_name = config.get('VECTORDB', 'collection_name').lower()

        bulk_size = config.getint('VECTORDB', 'opensearch_bulk_size', fallback=500)

        db = OpenSearchVectorSearch.from_documents(
            documents[:bulk_size],
            embedding=embeddings,
            opensearch_url=opensearch_url,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=index_name,
            bulk_size=bulk_size
        )
        remaining_documents = documents[bulk_size:]

        while remaining_documents:
            db.add_documents(remaining_documents[:bulk_size])
            remaining_documents = remaining_documents[bulk_size:]
    else:
        raise NotImplementedError(f"{vectordb} load_store not implemented yet")

    return db


def get_store(embeddings: Embeddings = None, collection_name=None) -> VectorStore:

    vectordb = config['VECTORDB']['vectordb']

    if not embeddings:
        embeddings = load_embeddings()

    if not collection_name:
        # "Default"
        collection_name = config['VECTORDB']['collection_name']

    db = None

    if vectordb == "faiss":
        index_dir = config['VECTORDB']['faiss_persist_directory']

        ## add allow_dangerous_deserialisation=True to solve this error
        ## "ValueError: The de-serialization relies loading a pickle file.
        ## Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on
        ## your machine.You will need to set `allow_dangerous_deserialization` to `True`
        ## to enable deserialization. If you do this, make sure that you trust the source of the data.
        ## For example, if you are loading a file that you created, and no that no one else has modified the file,
        ## then this is safe to do. Do not set this to `True`
        ## if you are loading a file from an untrusted source (e.g., some random site on the internet.)."
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']
        persistent_client = chromadb.PersistentClient(path=persist_directory)

        db = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=embeddings)

    elif vectordb == "opensearch":
        credentials = Session().get_credentials()
        aws_region = config.get('VECTORDB', 'opensearch_aws_region')

        awsauth = AWS4Auth(region=aws_region, service='es',
                    refreshable_credentials=credentials)
        
        opensearch_url = config.get('VECTORDB', 'opensearch_url')
        # Index name should be lowercase
        index_name = collection_name.lower()

        bulk_size = config.getint('VECTORDB', 'opensearch_bulk_size', fallback=500)

        db = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            embedding_function=embeddings,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            index_name=index_name,
            bulk_size=bulk_size
        )
    else:
        raise NotImplementedError(f"{vectordb} get_store not implemented yet")

    return db


def get_collection_count(collection_name:str = None) -> int:
    """Get the number of documents in a collection (chroma, FAISS) or an index (opensearch)"""
    if collection_name is None:
        collection_name = config.get('VECTORDB', 'collection_name')
    store = get_store(collection_name)
    if isinstance(store, OpenSearchVectorSearch):
        client = store.client
        index_name = collection_name.lower()
        try:
            count_response = client.count(index=index_name)
        except exceptions.NotFoundError:
            count_response = None
        if count_response:
            count = count_response['count']
        else:
            count = 0
    elif isinstance(store, Chroma):
        collection = store._collection
        count = collection.count()
    elif isinstance(store, FAISS):
        count = 0
    return count


def clean_text(s):
    regex_replacements = [
        (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
        (re.compile(r',(\s*])'), r'\1'),
    ]
    for regex, replacement in regex_replacements:
        s = regex.sub(replacement, s)
    return s


def load_doc(pdfs: Union[list[UploadedFile], None, UploadedFile], metadata = None) -> Optional[list[Document]]:
    if pdfs is not None:
        docs = []
        if metadata is None:
            metadata = {}
        for pdf in pdfs:
            if pdf.type == "application/pdf":
                reader = PdfReader(pdf)
                for i, page in enumerate(reader.pages, start=1):
                    page_content = page.extract_text().strip()
                    if page_content:
                        chunks = _chunk_texts(page_content)
                        for chunk in chunks:
                            page_metadata = {Metadata.PAGE.value: i, Metadata.FILENAME.value: pdf.name,
                                             Metadata.CHUNK_TYPE.value: ChunkType.TEXT.value}
                            page_metadata.update(metadata)
                            docs.append(Document(page_content=clean_text(chunk), metadata=page_metadata))
            elif pdf.type == "text/plain":
                page_content = pdf.read().decode().strip()  # read file content and decode it
                chunks = _chunk_texts(page_content)
                for chunk in chunks:
                    page_metadata = {Metadata.FILENAME.value: pdf.name, Metadata.CHUNK_TYPE.value: ChunkType.TEXT.value}
                    page_metadata.update(metadata)
                    docs.append(Document(page_content=clean_text(chunk), metadata=page_metadata))
        return docs
    else:
        return None


def load_text(texts: Union[list[UploadedFile], None, UploadedFile], metadata: dict={}):
    if texts is not None:
        docs = []
        for txt in texts:
            page_metadata = {Metadata.FILENAME: txt.name}
            file_content = txt.read().decode()  # read file content and decode it
            docs.append(Document(page_content=file_content, metadata=metadata))
        return docs
    else:
        return None


def persist_uploaded_documents(documents: list[UploadedFile]) -> None:
    """Persist documents uploaded from Streamlit"""
    for document in documents:
        persist_file(document, document.name)


def persist_images(images) -> None:
    """Persist images from pypdf"""
    for image in images:
        persist_file(image, image.name, 'image')


def persist_chat(chat: str, chat_name: str) -> None:
    """Persist chat history"""
    raise NotImplementedError('Function not implemented yet.')


def persist_file(file, filename:str, file_type: str = '') -> None:
    """Persist file to selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == 'LOCAL':
        persist_file_locally(file, filename=filename, file_type=file_type)
    elif storage_interface == 'S3':
        persist_file_to_s3(file, filename=filename, file_type=file_type)
    elif storage_interface == 'NONE':
        pass
    else:
        raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def persist_file_to_s3(file, filename: str, file_type: str) -> None:
    """Persist file to S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    folder_name = "images" if file_type == "image" else "docs"
    file_key = f"{folder_name}/{filename}"

    s3_client = boto3.client('s3')

    if file_type == 'image':
        file_object =  io.BytesIO(file.data)
    else:
        file_object = io.BytesIO(file.getvalue())

    s3_client.upload_fileobj(file_object, s3_bucket, file_key)


def persist_file_locally(file, filename:str, file_type: str) -> None:
    """Persist file to local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')
    folder_name = "images" if file_type == "image" else "docs"

    file_path = os.path.join(documents_path, folder_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = os.path.join(file_path, filename)

    with open(file_path, 'wb') as f:
        if file_type == 'image':
            f.write(file.data)
        else:
            f.write(file.getvalue())


def get_file(filename: str, file_type: str=''):
    """Get file from selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == 'LOCAL':
        return get_file_locally(filename=filename, file_type=file_type)

    if storage_interface == 'S3':
        return get_file_from_s3(filename=filename, file_type=file_type)

    if storage_interface == 'NONE':
        return None

    raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def get_file_locally(filename: str, file_type: str):
    """Get file from local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')
    folder_name = "images" if file_type == "image" else "docs"

    file_path = os.path.join(documents_path, folder_name, filename)

    if os.path.exists(file_path):
        return open(file_path, 'rb').read()

    return None


def get_file_from_s3(filename: str, file_type: str):
    """Get file from S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    folder_name = "images" if file_type == "image" else "docs"
    file_key = f"{folder_name}/{filename}"

    s3_client = boto3.client('s3')

    response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    return response['Body'].read()
