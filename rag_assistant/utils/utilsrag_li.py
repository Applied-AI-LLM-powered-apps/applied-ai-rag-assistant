import os
from typing import Sequence, Optional

import chromadb
from langchain_core.documents import Document
# from langchain_core.language_models import LLM
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.agent.function_calling.base import FunctionCallingAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter, HierarchicalNodeParser, \
    get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config_loader import load_config

config = load_config()

llama_index_root_dir = config['LLAMA_INDEX']['LLAMA_INDEX_ROOT_DIR']
sentence_index_dir = config['LLAMA_INDEX']['SENTENCE_INDEX_DIR']
merging_index_dir = config['LLAMA_INDEX']['MERGING_INDEX_DIR']
subquery_index_dir = config['LLAMA_INDEX']['SUBQUERY_INDEX_DIR']


def build_sentence_window_index(
    documents,
    save_dir=f"{llama_index_root_dir}/{sentence_index_dir}"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    text_splitter = SentenceSplitter()
    Settings.text_splitter = text_splitter
    if not os.path.exists(save_dir):

        nodes = node_parser.get_nodes_from_documents(documents)
        # base_nodes = text_splitter.get_nodes_from_documents(documents)

        sentence_index = VectorStoreIndex(nodes)
        sentence_index.storage_context.persist(persist_dir=save_dir)

        # base_index = VectorStoreIndex(base_nodes)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        # we can use another model just for rerank ???
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def build_automerging_index(
    documents,
    save_dir=f"{llama_index_root_dir}/{merging_index_dir}",
    chunk_sizes=None
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    # merging_context = ServiceContext.from_defaults(
    #     llm=llm,
    #     embed_model=embed_model,
    # )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context,
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )

    rerank = SentenceTransformerRerank(
        # we can use another model just for rerank ???
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )

    return auto_merging_engine


def create_automerging_engine(
        documents: Sequence[Document]
    ):

    automerging_index = build_automerging_index(
        documents
    )

    automerging_query_engine = get_automerging_query_engine(
        automerging_index,
    )

    return automerging_query_engine


def create_automerging_agent(
        llm,
        documents: Sequence[Document],
        name: str,
        description: str,
        query_engine: BaseQueryEngine = None
        ):

    if not query_engine:
        query_engine = create_automerging_engine(
            documents
        )

    agent_li = create_li_agent(name, description, query_engine, llm=llm)
    return agent_li


def create_sentence_window_engine(
        documents: Sequence[Document]
):
    sentence_index = build_sentence_window_index(
        documents
    )
    sentence_window_engine = get_sentence_window_query_engine(sentence_index)
    return sentence_window_engine


def create_sentence_window_agent(
        llm,
        documents: Sequence[Document],
        name:str,
        description: str,
        query_engine: BaseQueryEngine = None,
        storage_context: Optional[StorageContext] = None):

    if query_engine is None:
        query_engine = create_sentence_window_engine(
            documents
        )

    agent_li = create_li_agent(name, description, query_engine, llm=llm)

    return agent_li


def infer_topic_from_list(doc_name, topics):
    # Normalize the document name to lower case for case-insensitive matching
    doc_name_lower = doc_name.lower()
    for topic in topics:
        # Check if the topic is in the document name
        if topic.lower() in doc_name_lower:
            return topic
    return "Other"  # Default topic if no matches found


def create_subquery_engine(
        topics: list[str],
        documents: Sequence[Document]
):

    doc_set = {topic: [] for topic in topics}
    all_docs = []
    for doc in documents:
        topic = infer_topic_from_list(doc.metadata['file_path'], topics)
        doc_set[topic].append(doc)
        all_docs.append(doc)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    index_set = {}
    for topic in topics:
        # chroma_collection = db.get_or_create_collection(f"RAG_{topic}")
        # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[topic],
            storage_context=storage_context,
        )
        index_set[topic] = cur_index
        storage_context.persist(persist_dir=f"{llama_index_root_dir}/{subquery_index_dir}/{topic}")

    index_set = {}
    for topic in topics:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{llama_index_root_dir}/{subquery_index_dir}/{topic}"
        )
        cur_index = load_index_from_storage(
            storage_context,
        )
        index_set[topic] = cur_index

    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[topic].as_query_engine(),
            metadata=ToolMetadata(
                name=f"vector_index_{topic}",
                description=f"useful for when you want to answer queries about {topic}",
            ),
        )
        for topic in topics
    ]

    # now I want to do the same with a list of BaseTool
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
    )
    return query_engine


def create_subquery_agent(
        llm,
        topics: list[str],
        documents: Sequence[Document],
        name: str,
        description: str,
        query_engine: BaseQueryEngine = None
):

    if query_engine is None:
        query_engine = create_subquery_engine(
            topics,
            documents
        )

    agent_li = create_li_agent(name, description, query_engine, llm)
    return agent_li


def create_direct_query_engine(
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None):
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()
    return query_engine


def agent_li_factory(advanced_rag: str, llm: FunctionCallingLLM, documents, topics, vector_store=None):

    agent_lli = None
    if advanced_rag == "sentence_window":

        name = "sentence_window_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_lli = create_sentence_window_agent(llm=llm,
                                                 documents=documents,
                                                 name=name,
                                                 description=description)

    elif advanced_rag == "automerging":

        name = "automerging_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_lli = create_automerging_agent(llm=llm,
                                             documents=documents,
                                             name=name,
                                             description=description)

    elif advanced_rag == "subquery":

        name = "sub_question_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_lli = create_subquery_agent(
            llm=llm,
            topics=topics,
            documents=documents,
            name=name,
            description=description)

    elif advanced_rag == "direct_query":
        name = "direct_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        agent_lli = create_direct_query_agent(llm=llm,
                                              documents=documents,
                                              name=name,
                                              description=description,
                                              storage_context=storage_context)

    return agent_lli


def create_direct_query_agent(llm, documents: Sequence[Document],
                              name:str, description: str,
                              query_engine: BaseQueryEngine = None,
                              storage_context: Optional[StorageContext] = None
                              ):

    if query_engine is None:
        query_engine = create_direct_query_engine(
            documents,
            storage_context
        )

    agent_li = create_li_agent(name, description, query_engine, llm)

    return agent_li


def create_li_agent(name: str, description: str, query_engine: BaseQueryEngine, llm: Optional[LLM] = None):

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=name,
            description=description,
        ),
    )
    ## TODO NEW GENERIC VERSION TO CALL TOOL WITH LLAMAINDEX
    # agent_li = OpenAIAgent.from_tools(tools=[query_engine_tool], verbose=True)
    # MistralAIAgent.from_tools()
    agent_li = FunctionCallingAgent.from_llm(tools=[query_engine_tool], llm=llm, verbose=True)
    return agent_li
