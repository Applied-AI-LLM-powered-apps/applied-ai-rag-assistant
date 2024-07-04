import shutil
from abc import ABC, abstractmethod

import pytest
import os
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, DocumentSummaryIndex, \
    get_response_synthesizer
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from trulens_eval import TruChain, TruLlama, OpenAI

import numpy as np
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings


import nest_asyncio

import boto3

from trulens_eval import (
    Feedback,
)
from trulens_eval.feedback.provider.bedrock import Bedrock as TruBedrock

from trulens_eval.app import App

from rag_assistant.utils.utilsrag_lc import agent_lc_factory

_ = load_dotenv(find_dotenv())

nest_asyncio.apply()

aws_profile_name = os.getenv("profile_name")
feedback_model_id = "amazon.titan-text-express-v1"
aws_region_name = "eu-west-3"
bedrock_endpoint_url = "https://bedrock-runtime.eu-west-3.amazonaws.com"

embedding_model_id = "amazon.titan-embed-text-v1" # "amazon.titan-embed-image-v1"  # "amazon.titan-embed-text-v1"
embedding_region_name = "eu-central-1"
embedding_endpoint_url = "https://bedrock-runtime.eu-central-1.amazonaws.com"


class TestCaseHelper(ABC):

    @property
    def test_name(self):
        return "amazon.titan-text-express-v1"

    @property
    def model_id(self):
        return "default_model_name"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v1"

    @property
    def chunk_size(self):
        return 1024

    @property
    def chunk_overlap(self):
        return 20

    # qa_chain = None
    # db = None
    # eval_questions = None
    # embed_model = None
    # llm = None
    # documents = None
    # bedrock = None
    #
    # def __init__(self):
    #     self.qa_chain = None
    #     self.db = None
    #     self.eval_questions = None
    #     self.embed_model = None
    #     self.llm = None
    #     self.documents = None
    #     self.bedrock = None

    @pytest.fixture(scope="class", autouse=True)
    def bedrock_prepare(self, request):
        boto3.setup_default_session(profile_name=aws_profile_name)
        request.cls.bedrock = boto3.client('bedrock-runtime',
                                           region_name=aws_region_name,
                                           endpoint_url=bedrock_endpoint_url)

    @pytest.fixture(scope="class", autouse=True)
    def llm_prepare(self, request, bedrock_prepare):
        pass

    @pytest.fixture(scope="class", autouse=True)
    def embeddings_prepare(self, request, bedrock_prepare):
        pass

    @pytest.fixture(scope="class", autouse=True)
    def eval_questions_prepare(self, request):
        request.cls.eval_questions = []
        with open('tests/rag_bedrock/eval_questions.txt', 'r') as file:
            for line in file:
                request.cls.eval_questions.append(line.strip())

    @pytest.fixture(scope="class", autouse=True)
    def provider_prepare(self, request, bedrock_prepare):
        request.cls.provider = TruBedrock(
             model_id=feedback_model_id,
             client=request.cls.bedrock)


    @pytest.fixture(scope="class", autouse=True)
    def rag_prepare(self, request, documents_prepare, embeddings_prepare,
                    llm_prepare):
        pass

    @pytest.fixture(scope="class", autouse=True)
    def trulens_context_prepare(self, request, rag_prepare):
        request.cls.context = App.select_context(request.cls.rag)

    @pytest.fixture(scope="class", autouse=True)
    def trulens_recorder_prepare(self, request, rag_prepare, feedbacks_prepare):
        pass

    @pytest.fixture(scope="class", autouse=True)
    def feedbacks_prepare(self, request, trulens_context_prepare, provider_prepare):

        qa_relevance = (
            Feedback(request.cls.provider.relevance_with_cot_reasons, name="Answer Relevance")
            .on_input()
            .on_output()
        )

        qs_relevance = (
            Feedback(request.cls.provider.context_relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(request.cls.context)
            .aggregate(np.mean)
        )

        # not working on bedrock
        # groundedness = (
        #     Feedback(request.cls.provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        #     .on(request.cls.context.collect())
        #     .on_output()
        # )

        request.cls.feedbacks = [qa_relevance,
                                 qs_relevance,
                                 #groundedness
                                 ]


class LangchainTestRAGHelper(TestCaseHelper):

    @pytest.fixture(scope="class", autouse=True)
    def llm_prepare(self, request, bedrock_prepare):
        request.cls.llm = ChatBedrock(
            client=request.cls.bedrock,
            model_id=self.model_id,
            streaming=False,
        )

    @pytest.fixture(scope="class", autouse=True)
    def embeddings_prepare(self, request, bedrock_prepare):
        request.cls.embed_model = BedrockEmbeddings(
            #client=request.cls.bedrock,
            region_name=embedding_region_name,
            endpoint_url=embedding_endpoint_url,
            model_id=self.embedding_model_id
        )

    @pytest.fixture(scope="class", autouse=True)
    def documents_prepare(self, request):
        _loader = PyPDFLoader("tests/rag_bedrock/eval_document.pdf")
        _documents = _loader.load()
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        request.cls.documents = character_splitter.split_documents(_documents)

    @pytest.fixture(scope="class", autouse=True)
    def rag_prepare(self, request, documents_prepare, embeddings_prepare,
                    llm_prepare):

        request.cls.db = Chroma.from_documents(
            documents=request.cls.documents,
            embedding=request.cls.embed_model,
            collection_name=f"Test_RAG_bedrock_{self.test_name}",
        )

        request.cls.rag = agent_lc_factory(chain_type="stuff",
                                           llm=request.cls.llm,
                                           search_kwargs={"k": 4},
                                           search_type="similarity", vectorstore=request.cls.db)

    @pytest.fixture(scope="class", autouse=True)
    def trulens_recorder_prepare(self, request, rag_prepare, feedbacks_prepare):
        request.cls.tru_recorder = TruChain(
            request.cls.rag,
            app_id=self.test_name,
            feedbacks=request.cls.feedbacks
        )

    def test_qa_chain(self, request, trulens_recorder_prepare, rag_prepare):
        tru_recorder = request.cls.tru_recorder

        with tru_recorder as recording:
            for question in request.cls.eval_questions:
                print(f"question: {str(question)}")
                response = request.cls.rag.invoke(question)
                assert response is not None, "L'interprétation n'a pas retourné de résultat."
                print(f"response: {str(response)}")


class LlamaIndexTestRAGHelper(TestCaseHelper):

    @property
    def topic(self):
        return "default_topic"

    @pytest.fixture(scope="class", autouse=True)
    def documents_prepare(self, request):
        documents = SimpleDirectoryReader(
            input_files=["tests/rag_bedrock/eval_document.pdf",
                         ]
        ).load_data()
        splitter = SentenceSplitter(chunk_size=self.chunk_size,
                                    chunk_overlap=self.chunk_overlap)
        request.cls.documents = splitter.get_nodes_from_documents(documents)

    @pytest.fixture(scope="class", autouse=True)
    def llm_prepare(self, request, bedrock_prepare):
        request.cls.llm = Bedrock(client=request.cls.bedrock,
                                  model=self.model_id,
                                  )

    @pytest.fixture(scope="class", autouse=True)
    def embeddings_prepare(self, request, bedrock_prepare):
        request.cls.embed_model = BedrockEmbedding(
            region_name=embedding_region_name,
            endpoint_url=embedding_endpoint_url,
            #client=request.cls.bedrock,
            model_name=self.embedding_model_id
        )

    @pytest.fixture(scope="class", autouse=True)
    def temp_dir(self, request):
        # TODO don't really not were it goes ?
        dir_name = "./tmp_llama_index/"
        os.makedirs(dir_name, exist_ok=True)
        shutil.rmtree(dir_name)
        # Yield the directory name to the tests
        yield dir_name

        # Teardown: Remove the temporary directory after tests are done
        shutil.rmtree(dir_name, ignore_errors=True)
        pass

    @pytest.fixture(scope="class", autouse=True)
    def tru_recorder_prepare(self, request, rag_prepare, feedbacks_prepare):
        request.cls.tru_recorder = TruLlama(
            request.cls.rag,
            app_id=self.test_name,
            feedbacks=request.cls.feedbacks
        )


    @pytest.fixture(scope="class", autouse=True)
    def rag_prepare(self, request, llm_prepare, documents_prepare, embeddings_prepare):

        nodes = request.cls.documents

        vector_index = VectorStoreIndex(nodes, embed_model=request.cls.embed_model)
        vector_query_engine = vector_index.as_query_engine(llm=request.cls.llm)

        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True
        )

        summary_index = DocumentSummaryIndex(nodes=nodes,
                                             response_synthesizer=response_synthesizer,
                                             llm=request.cls.llm,
                                             embed_model=request.cls.embed_model)
        summary_query_engine = summary_index.as_query_engine(
            response_mode=ResponseMode.TREE_SUMMARIZE,
            use_async=True,
            llm=request.cls.llm
        )

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=(
                f"Use ONLY IF you want to get a holistic summary of {self.topic}."
                f"Do NOT use if you have specific questions on {self.topic}."
            ),
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                f"Useful for retrieving specific questions over {self.topic}."
            ),
        )
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                summary_tool,
                vector_tool,
            ],
            verbose=True
        )
        request.cls.rag = query_engine

        #query_engine_tool = QueryEngineTool.from_defaults(
        #    name=f"{self.test_name}",
        #    query_engine=query_engine,
        #    description=f"Usefully is you have any question on {self.topic}"
        #)
        #agent_worker = FunctionCallingAgentWorker.from_tools(
        #    tools = [query_engine_tool],
        #    llm=request.cls.llm,
        #    verbose=True
        #)
        #request.cls.rag = AgentRunner(agent_worker)



    def test_query_engine(self, request,
                          tru_recorder_prepare, rag_prepare, eval_questions_prepare):

        tru_recorder = request.cls.tru_recorder

        with tru_recorder as recording:
            for question in request.cls.eval_questions:
                print(f"question: {str(question)}")
                response = request.cls.rag.query(question)
                print(f"response: {str(response)}")
                assert response is not None, "L'interprétation n'a pas retourné de résultat."
