from json import JSONDecodeError
from typing import Union

import chromadb
import streamlit as st
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import ToolException, Tool
from llama_index.embeddings.langchain import LangchainEmbedding

from llama_index.llms.openai import OpenAI

from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.config_loader import load_config
from utils.utilsdoc import load_doc

from utils.utilsrag_li import agent_li_factory

from utils.utilsllm import load_model, load_embeddings

from dotenv import load_dotenv, find_dotenv

from langchain.agents import AgentExecutor, create_structured_chat_agent

from llama_index.core import Settings
from llama_index.core.schema import Document

from langchain_core.tracers.context import tracing_v2_enabled


# EXTERNALISATION OF PROMPTS TO HAVE THEIR OWN VERSIONING
from shared.rag_prompts import human, __structured_chat_agent__

load_dotenv(find_dotenv())

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['LLM']['LLM_MODEL']

topics = ["Cloud", "Security", "GenAI", "Application", "Architecture", "AWS", "Other"]

model_to_index = {
    "OPENAI": 0,
    "MISTRAL": 1
}


def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.subheader("LangChain model provider")
        st.sidebar.checkbox("OpenAI", LLM_MODEL == "OPENAI", disabled=True)
        st.sidebar.checkbox("Mistral", LLM_MODEL == "MISTRAL", disabled=True)


def _load_doc(pdfs: Union[list[UploadedFile], None, UploadedFile]) -> list[Document]:
    # loader = SimpleDirectoryReader(input_dir=f"data/sources/pdf/", recursive=True, required_exts=[".pdf"])
    # all_docs = loader.load_data()
    lc_all_docs = load_doc(pdfs)
    all_docs = []
    for doc in lc_all_docs:
        all_docs.append(Document.from_langchain_format(doc))
    return all_docs


def configure_agent(all_docs: list[Document], model_name, advanced_rag):
    # all_docs = load_doc()

    ## START LLAMAINDEX
    lc_embeddings = load_embeddings(model_name)
    embeddings_rag = LangchainEmbedding(lc_embeddings)
    llm_rag = None
    if model_name.startswith("gpt"):
        llm_rag = OpenAI(model=model_name, temperature=0.1)
        # embeddings_rag = OpenAIEmbedding()
    elif model_name.startswith("mistral"):
        llm_rag = MistralAI(model=model_name, temperature=0.1)
        # embeddings_rag = MistralAIEmbedding()

    #
    # Settings seems to be the preferred version to setup llm and embeddings with latest LI API
    Settings.llm = llm_rag
    Settings.embed_model = embeddings_rag

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("RAG_LI_Agent")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    agent_li = agent_li_factory(advanced_rag=advanced_rag, llm=llm_rag,
                                documents=all_docs,
                                topics=topics,
                                vector_store=vector_store)

    def _handle_error(error: ToolException) -> str:
        if error == JSONDecodeError:
            return "Reformat in JSON and try again"
        elif error.args[0].startswith("Too many arguments to single-input tool"):
            return "Format in a SINGLE STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
        return (
                "The following errors occurred during tool execution:"
                + error.args[0]
                + "Please try another tool.")

    lc_tools = [
        Tool(
            name=f"Knowledge Agent (LI)",
            func=agent_li.chat,
            description=f"""Useful when you need to answer questions on {topics}. "
                        "DO NOT USE MULTI-ARGUMENTS INPUT.""",
            handle_tool_error=_handle_error,
        ),
    ]
    #
    # END LLAMAINDEX

    #
    # START LANGCHAIN
    # MODEL FOR LANGCHAIN IS DEFINE GLOBALLY IN CONF/CONFIG.INI
    llm_agent = load_model()

    # prompt = PromptTemplate.from_template(__template__)
    prompt = (ChatPromptTemplate.from_messages(
        [
            ("system", __structured_chat_agent__),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    ))

    agent = create_structured_chat_agent(
        llm=llm_agent,
        tools=lc_tools,
        prompt=prompt
    )

    #
    # TODO
    # sometimes received "Parsing LLM output produced both a final answer and a parse-able action" with mistral
    # add a handle_parsing_errors, reduce the case but still appears time to time.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        handle_parsing_errors="Check your output and make sure it conforms to format!"
                              " Do not output an action and a final answer at the same time.")
    # END LANGCHAIN
    #
    return agent_executor


def main():

    st.title("Question Answering Assistant (RAG)")

    load_sidebar()

    model_index = model_to_index[LLM_MODEL]
    agent_model = st.sidebar.radio("RAG Agent LLM Provider", ["OPENAI", "MISTRAL"], index=model_index)

    st.sidebar.subheader("RAG Agent Model")
    model_name_gpt = st.sidebar.radio("OpenAI Model", ["gpt-3.5-turbo", "gpt-4-turbo"],
                                      captions=["GPT 3.5 Turbo", "GPT 4 Turbo"],
                                      index=1, disabled=agent_model != "OPENAI")

    model_name_mistral = st.sidebar.radio("Mistral Model", ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
                                          captions=["Mistral 7b", "Mixtral", "Mistral Large"],
                                          index=2, disabled=agent_model != "MISTRAL")

    model_name = None
    if agent_model == "MISTRAL":
        model_name = model_name_mistral
    elif agent_model == "OPENAI":
        model_name = model_name_gpt

    st.sidebar.subheader("RAG Agent params")
    advanced_rag = st.sidebar.radio("Advanced RAG (Llamaindex)", ["direct_query", "subquery", "automerging",
                                                                  "sentence_window"])

    pdfs = st.file_uploader("Document(s) à transmettre", type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    disabled = True
    # calling an internal function for adapting LC or LI Document
    docs = _load_doc(pdfs)

    if (docs is not None) and (len(docs)):
        disabled = False

    if not disabled:

        st.header("RAG agent with LlamaIndex")
        agent = configure_agent(docs, model_name, advanced_rag)

        history = StreamlitChatMessageHistory(key="chat_history")
        if len(history.messages) == 0:
            history.add_ai_message("What do you want to know?")

        view_messages = st.expander("View the message contents in session state")

        chain_with_history = RunnableWithMessageHistory(
            agent,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Display chat messages from history on app rerun
        for message in history.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

        # Accept user input
        if prompt := st.chat_input():
            # Add user message to chat history
            # Note: new messages are saved to history automatically by Langchain during run
            # st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with tracing_v2_enabled(project_name="Applied AI RAG Assistant",
                                        tags=["LlamaIndex", "Agent"]):
                    config = {"configurable": {"session_id": "any"}}
                    response = chain_with_history.invoke(input={"input": prompt}, config=config)

                    answer = f"🦙: {response['output']}"
                    st.write(answer)

        # Draw the messages at the end, so newly generated ones show up immediately
        with view_messages:
            """
            Message History initialized with:
            ```python
            msgs = StreamlitChatMessageHistory(key="chat_history")
            ```
    
            Contents of `st.session_state.chat_history`:
            """
            view_messages.json(st.session_state.chat_history)


if __name__ == "__main__":
    main()
