import pytest
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

import rag_assistant.utils.utilsrag_li
from rag_assistant.utils.utilsrag_li import create_automerging_engine, create_sentence_window_engine, create_subquery_engine, \
    create_direct_query_engine, create_li_agent

import shutil


import numpy as np

import nest_asyncio

from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI,
    Tru, Select
)
from trulens_eval.app import App

load_dotenv(find_dotenv())

# Set OpenAI API key from Streamlit secrets
openai_api_key = os.getenv('OPENAI_API_KEY')

nest_asyncio.apply()

provider = OpenAI()

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


def get_trulens_feedbacks(query_engine):

    context = App.select_context(query_engine)

    qa_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
        .on(context.collect())
        .on_output()
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    return feedbacks


def get_prebuilt_trulens_recorder(query_engine, app_id, feedbacks):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


@pytest.fixture(scope="module")
def temp_dir(request):
    # TODO define a test specific dir...
    # Setup: Create a temporary directory for the test module
    dir_name = rag_assistant.utils.utilsrag_li.llama_index_root_dir
    os.makedirs(dir_name, exist_ok=True)
    shutil.rmtree(dir_name)
    # Yield the directory name to the tests
    yield dir_name

    # Teardown: Remove the temporary directory after tests are done
    if os.path.isdir(dir_name):  # Check if the directory exists before removing it
        #shutil.rmtree(dir_name) # TODO commenting this while fix above is not done
        pass


def llm_prepare():
    llm = MistralAI(model="mistral-large-latest")

    Settings.llm = llm

    return llm


def embeddings_prepare():
    embed_model = MistralAIEmbedding()

    Settings.embed_model = embed_model

    return embed_model


@pytest.fixture
def docs_prepare():
    documents = SimpleDirectoryReader(
        input_files=["tests/utils/eBook-How-to-Build-a-Career-in-AI.pdf"]
    ).load_data()
    return documents


@pytest.fixture
def eval_questions_prepare():
    eval_questions = []
    with open('tests/utils/eval_questions.txt', 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            print(item)
            eval_questions.append(item)
    return eval_questions


def test_automerging_agent(temp_dir,
                           docs_prepare, eval_questions_prepare, trulens_prepare):

    llm = llm_prepare()

    query_engine = create_automerging_engine(docs_prepare)

    feedbacks = get_trulens_feedbacks(query_engine)

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Automerging Query Engine",
                                                 feedbacks=feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)
            assert response is not None, "L'interprétation n'a pas retourné de résultat."

    agent = create_li_agent(name="test_automerging_agent", description="Test Automerging Agent",
                            query_engine=query_engine, llm=llm)

    response = agent.query("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_sentence_window_agent(temp_dir, docs_prepare, eval_questions_prepare, trulens_prepare):

    llm = llm_prepare()

    query_engine = create_sentence_window_engine(
        docs_prepare,
    )

    feedbacks = get_trulens_feedbacks(query_engine)

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sentence Window Query Engine",
                                                 feedbacks=feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    agent = create_li_agent(name="test_sentence_window_agent", description="Test Sentence Window Agent",
                            query_engine=query_engine, llm=llm)

    response = agent.query("How do I get started on a personal project in AI?")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_llamaindex_agent(temp_dir, docs_prepare, eval_questions_prepare, trulens_prepare):

    llm = llm_prepare()

    query_engine = create_direct_query_engine(
        docs_prepare,
    )

    feedbacks = get_trulens_feedbacks(query_engine)

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Direct Query Engine",
                                                 feedbacks=feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    agent = create_li_agent(name="test_direct_query_agent", description="Test Direct Query Agent",
                            query_engine=query_engine,
                            llm=llm)

    response = agent.query("How do I get started on a personal project in AI?")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_subquery_agent(temp_dir, docs_prepare, eval_questions_prepare, trulens_prepare):

    llm = llm_prepare()

    topics = ["AI", "Other"]
    query_engine = create_subquery_engine(
        topics,
        docs_prepare,
    )

    feedbacks = get_trulens_feedbacks(query_engine)

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sub Query Engine",
                                                 feedbacks=feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    agent = create_li_agent(name="test_subquery_agent", description="Test Subquery Agent",
                            query_engine=query_engine,
                            llm=llm)

    response = agent.query("How do I get started on a personal project in AI?")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."

