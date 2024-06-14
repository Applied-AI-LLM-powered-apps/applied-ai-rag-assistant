import os
import random
import threading
from json import JSONDecodeError

import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import ToolException, Tool, tool
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.tools import QueryEngineTool
from streamlit_pdf_viewer import pdf_viewer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langsmith import traceable

import utils.constants
from utils.constants import Metadata
from utils.utilsdoc import get_store, extract_unique_name
from utils.config_loader import load_config
from streamlit_feedback import streamlit_feedback
import logging

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from shared.rag_prompts import __structured_chat_agent__

from utils.utilsllm import load_model

load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')

# set logging
logger = logging.getLogger('AI_assistant_feedback')
logger.setLevel(logging.INFO)

# Check if the directory exists, if not create it
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a file handler for the logger
handler = logging.FileHandler(os.path.join(log_dir, 'feedback.log'))
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s -  %(message)s')
handler.setFormatter(formatter)  # Add the formatter to the handler

# Add the handler to the logger
logger.addHandler(handler)

config = load_config()
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']

LLAMA_INDEX_ROOT_DIR = config["LLAMA_INDEX"]["LLAMA_INDEX_ROOT_DIR"]
SUMMARY_INDEX_DIR = config["LLAMA_INDEX"]["SUMMARY_INDEX_DIR"]
summary_index_folder = f"{LLAMA_INDEX_ROOT_DIR}/{SUMMARY_INDEX_DIR}"

sessionid = "abc123"

__template2__ = """You are an assistant designed to guide software application architect and tech lead to go through a risk assessment questionnaire for application cloud deployment. 
    The questionnaire is designed to cover various pillars essential for cloud architecture,
     including security, compliance, availability, access methods, data storage, processing, performance efficiency,
      cost optimization, and operational excellence.
      
    You will assist user to answer to the questionnaire solely based on the information that will be provided to you.

    For each question, you are to follow the "Chain of Thought" process. This means that for each user's response, you will:

    - Acknowledge the response,
    - Reflect on the implications of the choice,
    - Identify any risks associated with the selected option,
    - Suggest best practices and architecture patterns that align with the user’s selection,
    - Guide them to the next relevant question based on their previous answers.

    Your objective is to ensure that by the end of the questionnaire, the user has a clear understanding of the appropriate architecture and services needed for a secure, efficient, and compliant cloud deployment. Remember to provide answers in a simple, interactive, and concise manner.

    Process:

    1. Begin by introducing the purpose of the assessment and ask the first question regarding data security and compliance.
    2. Based on the response, discuss the chosen level of data security, note any specific risks or requirements, and recommend corresponding cloud services or architectural patterns.
    3. Proceed to the next question on application availability. Once the user responds, reflect on the suitability of their choice for their application's criticality and suggest availability configurations.
    4. For questions on access methods and data storage, provide insights on securing application access points or optimizing data storage solutions.
    5. When discussing performance efficiency, highlight the trade-offs between performance and cost, and advise on scaling strategies.
    6. In the cost optimization section, engage in a brief discussion on budgetary constraints and recommend cost-effective cloud resource management.
    7. Conclude with operational excellence, focusing on automation and monitoring, and propose solutions for continuous integration and deployment.
    8. After the final question, summarize the user's choices and their implications for cloud architecture.
    9. Offer a brief closing statement that reassures the user of the assistance provided and the readiness of their cloud deployment strategy.

    Keep the interactions focused on architectural decisions without diverting to other unrelated topics.
    Be concise in your answer with a professional tone. 
    You are not to perform tasks outside the scope of the questionnaire, 
    such as executing code or accessing external databases. 
    Your guidance should be solely based on the information provided by the user in the context of the questionnaire.
    
    To start the conversation, introduce yourself and give 3 domains in which you can assist user."""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, ctx, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.ctx = ctx

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        # THIS TRICKS DOES NOT WORK IF THERE IS A SYSTEM PROMPT
        # SO QUESTION IS SHOWED WHEN STREAMING AND THEN CLEARED with final result
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
        # adding current thread to streamlit context to be able to display streaming
        add_script_run_ctx(threading.current_thread(), self.ctx)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# Define the callback handler for printing retrieval information
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container, ctx):
        self.status = container.status("**Context Retrieval**")
        self.ctx = ctx

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        # adding current thread to streamlit context to be able to display streaming
        add_script_run_ctx(threading.current_thread(), self.ctx)
        if 'retrievals' not in st.session_state:
            st.session_state['retrievals'] = []
        self.status.write(f"**Question reformulée:** {query}")
        # self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata[Metadata.FILENAME.value]
            page = doc.metadata[Metadata.PAGE.value]
            self.status.write(f"**Chunk {idx} from {source} - page {page}**")
            self.status.markdown(doc.page_content)
        st.session_state['retrievals'] = documents
        self.status.update(state="complete")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = StreamlitChatMessageHistory(key="chat_history")
    return st.session_state.store[session_id]


def configure_retriever():
    vectordb = get_store()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # , "fetch_k": 4

    return retriever


def _submit_feedback(user_response, emoji=None):
    if user_response['score'] == '👍':
        feedback_score = '+1'
    else:
        feedback_score = '-1'
    logger.info(f"Feedback_Score: {feedback_score}, Feedback_text: {user_response['text']}")
    return user_response


st.set_page_config(page_title="Chat with Documents", page_icon="🦜")

# Configure the retriever with PDF files
retriever = configure_retriever()

unique_topic_names = extract_unique_name(collection_name, Metadata.TOPIC.value)
topics = ', '.join(sorted(['\''+topic+'\'' for topic in unique_topic_names]))
if not unique_topic_names:
    topics = "'None'"

unique_file_names = extract_unique_name(collection_name, Metadata.FILENAME.value)
filenames = ', '.join(sorted(['\''+filename+'\'' for filename in unique_file_names]))
if not unique_file_names:
    filenames = "'None'"

chunk_type_desc = [e.value for e in utils.constants.ChunkType]
chunk_type_desc = ", ".join(chunk_type_desc) if len(chunk_type_desc) > 1 else chunk_type_desc[0]


metadata_field_info = [
    AttributeInfo(
        name=f"{Metadata.TOPIC.value}",
        description=f"The topic covered by the knowledge. One of [{topics}]",
        type="string",
    ),
    AttributeInfo(
        name=f"{Metadata.FILENAME.value}",
        description=f"The filename of the document containing the knowledge. One of [{filenames}]",
        type="string",
    ),
    AttributeInfo(
        name=f"{Metadata.CHUNK_TYPE.value}",
        description=f"The element type of knowledge. One of [{chunk_type_desc}]",
        type="string",
    ),
    AttributeInfo(
        name=f"{Metadata.PAGE.value}",
        description="The page of the knowledge within original document.",
        type="integer"
    ),
]

vectorstore = get_store()
self_query_llm = load_model(streaming=False, temperature=0)
document_content_description = f"Knowledge base on {topics}"
self_query_retriever = SelfQueryRetriever.from_llm(
    self_query_llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)
#
# prompt = get_query_constructor_prompt(
#     document_content_description,
#     metadata_field_info,
# )
# output_parser = StructuredQueryOutputParser.from_components()
# query_constructor = prompt | self_query_llm | output_parser
#
# self_query_retriever = SelfQueryRetriever(
#     query_constructor=query_constructor,
#     vectorstore=vectorstore,
#     structured_query_translator=ChromaTranslator(),
# )

st.session_state.store = {}

#
# L     L     MM     MM
# L     L     M M   M M
# L     L     M  M M  M
# LLLL  LLLL  M   M   M
#

storage_context = StorageContext.from_defaults(persist_dir=summary_index_folder)
doc_summary_index = load_index_from_storage(storage_context)
summary_query_engine = doc_summary_index.as_query_engine(
    response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True
)


def _handle_error(error: ToolException) -> str:
    if error == JSONDecodeError:
        return "Reformat in JSON and try again"
    elif error.args[0].startswith("Too many arguments to single-input tool"):
        return "Format in a SINGLE STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
    return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool.")


# Setup LLM and QA chain
llm_rag = load_model(temperature=0.1, streaming=False)
llm = load_model(streaming=True)

# llm.bind_tools(lc_tools)

msgs = get_session_history(sessionid)
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. \
If the question has specific rules on the content of knowledge like \
page number, file name or element type like Image, ensure to keep \
all these informations on the reformulated question.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
Maintain the same language as the user question.
Do not explain your logic, just output the reformulated question.

Reformulated Question:"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm_rag,
    #retriever,
    self_query_retriever,
    contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks on {topics}. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
If the question is not on {topics}, don't answer it. \
Use three sentences maximum and keep the answer concise.\
Maintain the same writing style as used in the context.\
Keep the same language as the follow up question.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_rag, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

memory_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# @tool
# def self_query(query: str) -> str:
#     """Useful IF you have a query that needs to have access to a specific content of
#      knowledge like page, file or element type.
#     DO NOT use if you have specific question.
#     DO NOT USE MULTI-ARGUMENTS INPUT."""
#     return self_query_retriever.invoke(query)


@tool
def knowledge_summary(question: str) -> str:
    """Useful IF you need to answer a general question or need to have a holistic summary on knowledge.
    DO NOT use if you have specific question.
    DO NOT USE MULTI-ARGUMENTS INPUT."""
    return summary_query_engine.query(question)


@tool
def question_answerer(question: str) -> str:
    """Useful when you need to answer a specif question on knowledge.
    or IF you have a query that requires to have access to a specific part of
     knowledge like page, file or element type.
    DO NOT USE MULTI-ARGUMENTS INPUT."""
    # Retrieving the streamlit context to bind it to call back
    # in order to write in another thread context
    ctx = get_script_run_ctx()
    retrieval_handler = PrintRetrievalHandler(st.container(), ctx)
    # RETRIEVE THE CONTAINER TO CLEAR IT LATER to not show question twice
    e = st.empty()
    stream_handler = StreamHandler(e, ctx)
    return memory_rag_chain.invoke(input={"input": question, "topics": topics}, config={
        "configurable": {"session_id": sessionid},
        "callbacks": [
            retrieval_handler,
            stream_handler
        ]
    })['answer']


lc_tools = [knowledge_summary, question_answerer]

human = '''{input}

    {agent_scratchpad}'''

prompt = (ChatPromptTemplate.from_messages(
    [
        ("system", __structured_chat_agent__),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
    ]
))

llm.bind_tools(lc_tools)
agent = create_structured_chat_agent(llm, lc_tools, prompt)


def _handle_error2(error) -> str:
    return str(error)  # [:250]


agent_executor = AgentExecutor(
    agent=agent,
    tools=lc_tools,
    handle_parsing_errors=
    "Check your output and make sure it includes an action that conforms to the expected format (json blob)!"
    " Do not output an action and a final answer at the same time."
)

conversational_rag_chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)

main_chain = conversational_rag_chain

suggested_questions_examples = [
    "Comment sécuriser les données sensibles ?",
    "Quelles stratégies pour une haute disponibilité ?",
    "Quels sont les mécanismes d'authentification API ?",
    "Comment assurez l'efficacité des performances ?",
    "Quelles informations doivent être fournies lors du lancement de l'IHM?",
    "Quel est le rôle de la fonction d'acheminement pour acheminer l'appel de service?",
    'Quelles sont les principales fonctionnalités du portail fournisseur dans la gestion des API?',
    "Que comprend la fonction d'exposition dans la gestion des API?"
]


def conversation_starters():
    llm = load_model(streaming=True)

    context = summary_query_engine.query("Make a complete summary of knowledge available"
                                         " on following topics {topics}.")

    ### Answer question ###
    cs_system_prompt = """You are a helpful solution architect and software engineer assistant.
        Your users are asking questions on specific topics.\
        Suggest exactly 6 questions related to the provided context to help them find the information they need. \
        Suggest only short questions without compound sentences. \
        Question must be self-explanatory and topic related.
        Suggest a variety of questions that cover different aspects of the context. \
        Use the summary of knowledge to generate the question on topics. \
        Make sure they are complete questions, and that they are related to the topics.
        Output one question per line. Do not number the questions. Do not group question by topics. 
        DO NOT make a summary or an introduction of your result. Output ONLY the generated questions.
        DO NOT output chapter per topic. Avoid blank line.
        Avoid duplicate question. Generate question in French.
        Questions: """

    #         Examples:
    #         What information needs to be provided during IHM launch?
    #         How is the data transferred to the service call?
    #         What functions are involved in API Management?
    #         What does the Exposure function in API Management entail?

    cs_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", cs_system_prompt),
            ("human", "{topics}"
                      "{summary}"),
        ]
    )
    output_parser = StrOutputParser()
    model = load_model(streaming=False)

    chain = cs_prompt | model | output_parser
    response = chain.invoke({"topics": topics, "summary": context})
    print(response)

    response_list = [line for line in response.split("\n") if line.strip() != '']
    if len(response_list) > 4:
        response_list = random.sample(response_list, 4)
    elif len(response_list) < 4:
        diff = 4 - len(response_list)
        additional_questions = random.sample(suggested_questions_examples, diff)
        response_list.extend(additional_questions)

    return response_list


if 'conversation_starters' not in st.session_state:
    st.session_state['conversation_starters'] = conversation_starters()

suggested_questions = st.session_state['conversation_starters']

@traceable(run_type="chain", project_name="RAG Assistant", tags=["LangChain", "RAG", "Chat_with_Docs"])
def handle_assistant_response(user_query):
    st.chat_message("user").write(user_query)
    with ((st.chat_message("assistant"))):
        if 'retrievals' in st.session_state:
            del st.session_state['retrievals']
        response = main_chain.invoke(
            input={"input": user_query, "topics": topics},
            config={"configurable": {"session_id": sessionid}}
        )
        ai_response = response["output"]  # "answer"
        # emptying container to remove initial question that is render by llm
        # e.empty()
        e = st.empty()
        with e.container():
            st.markdown(ai_response)
            # context = response["context"]
            if 'retrievals' in st.session_state:
                context = st.session_state['retrievals']
                display_context_in_pdf_viewer(context)
        logger.info(f"User Query: {user_query}, AI Response: {ai_response}")


def display_context_in_pdf_viewer(context):
    metadata = [(doc.metadata['filename'], doc.metadata['page']) for doc in context]
    metadata_dict = {}
    for filename, page in metadata:
        if filename not in metadata_dict:
            metadata_dict[filename] = []  # Initialize a new list for new filename
        metadata_dict[filename].append(page)
    # Create a new sorted list
    sorted_metadata = sorted(metadata_dict.items(), key=lambda x: len(x[1]), reverse=True)
    # Format the sorted list
    formatted_metadata = []
    for item in sorted_metadata:
        formatted_metadata.append([item[0], item[1]])
    if formatted_metadata:  # check if list is empty
        first_metadata = formatted_metadata[0]
        filename = first_metadata[0]
        pages = first_metadata[1]
        # show_retrievals = st.checkbox("Show PDFs")
        with st.expander(f"Source: {filename}", expanded=True):
            pdf_viewer(f"{upload_directory}/{filename}",
                       height=400,
                       pages_to_render=pages)


def suggestion_clicked(question):
    st.session_state.user_suggested_question = question


def main():
    st.title("Chat with Documents")

    msgs = get_session_history(sessionid)

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()

    # Display suggested questions in a 2x2 table
    with st.container():
        st.subheader("Amorces de conversation", divider="rainbow")
        col1, col2 = st.columns(2)
        for i, question in enumerate(suggested_questions, start=1):
            # if not st.session_state.get(f"suggested_question_{i}_hidden", False):
            col = col1 if i % 2 != 0 else col2
            col.button(question, on_click=suggestion_clicked, args=[question])

    # Chat interface
    avatars = {"human": "user", "ai": "assistant"}
    msgs = get_session_history(sessionid)
    for i, msg in enumerate(msgs.messages):
        st.chat_message(avatars[msg.type]).write(msg.content)
        if (msg.type == "ai") and (i > 0):
            streamlit_feedback(feedback_type="thumbs",
                               optional_text_label="Cette réponse vous convient-elle?",
                               key=f"feedback_{i}",
                               on_submit=lambda x: _submit_feedback(x, emoji="👍"))

    # Handle suggested questions
    if "user_suggested_question" in st.session_state:
        user_query = st.session_state.user_suggested_question
        st.session_state.pop("user_suggested_question")  # Clear the session state
        handle_assistant_response(user_query)

    # Handle user queries
    if user_query := st.chat_input(placeholder="Ask me anything!"):
        handle_assistant_response(user_query)


if __name__ == "__main__":
    main()
