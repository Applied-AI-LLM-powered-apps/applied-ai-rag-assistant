import os
import threading

import opensearchpy
import streamlit as st
from opensearchpy import NotFoundError
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


from shared.llm_facade import get_conversation_starters
from utils.auth import check_password
from utils.constants import Metadata, CollectionType
from utils.utilsdoc import get_store, extract_unique_name
from utils.config_loader import load_config
from utils.utilsfile import get_file
from streamlit_feedback import streamlit_feedback
import logging

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from utils.utilsllm import load_model

load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')

# set logging
feedback_logger = logging.getLogger('AI_assistant_feedback')
feedback_logger.setLevel(logging.INFO)

# Check if the directory exists, if not create it
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a file handler for the logger
feedback_handler = logging.FileHandler(os.path.join(log_dir, 'feedback.log'))
feedback_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s -  %(message)s')
feedback_handler.setFormatter(formatter) # Add the formatter to the handler

# Add the handler to the logger
feedback_logger.addHandler(feedback_handler)


config = load_config()
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']
top_k = int(config['LANGCHAIN']['SEARCH_TOP_K'])
search_type = config['LANGCHAIN']['SEARCH_TYPE']

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
        self.status = container.status("**Récupération du Contexte**")
        self.ctx = ctx

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        # adding current thread to streamlit context to be able to display streaming
        add_script_run_ctx(threading.current_thread(), self.ctx)
        self.status.write(f"**Question Reformulée:** {query}")
        # self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata[Metadata.FILENAME.value]
            page = doc.metadata[Metadata.PAGE.value]
            self.status.write(f"**Morceau {idx} de {source} - page {page}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = StreamlitChatMessageHistory(key="chat_history")
    return st.session_state.store[session_id]


def configure_retriever():
    vectordb = get_store()

    retriever = vectordb.as_retriever(search_type=search_type,
                                      search_kwargs={"k": top_k})

    return retriever


def _submit_feedback(user_response, emoji=None):
    if user_response['score'] == '👍':
        feedback_score = '+1'
    else:
        feedback_score = '-1'
    feedback_logger.info(f"Feedback_Score: {feedback_score}, Feedback_text: {user_response['text']}")
    return user_response


st.set_page_config(page_title="Chat with Documents", page_icon="🦜")


# Configure the retriever with PDF files
retriever = configure_retriever()

# Setup memory for contextual conversation

st.session_state.store = {}

unique_topic_names = extract_unique_name(collection_name, Metadata.TOPIC.value)
topics = ', '.join(sorted(list(unique_topic_names)))
if not unique_topic_names:
    topics = None

#
# L     L     MM     MM
# L     L     M M   M M
# L     L     M  M M  M
# LLLL  LLLL  M   M   M
#

# Setup LLM and QA chain
llm = load_model(streaming=True)

# msgs = StreamlitChatMessageHistory()
msgs = get_session_history(sessionid)
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
User question should be on {topics}.
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
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks on {topics}. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
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

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# llm_stream = load_model(streaming=True)
#
#
# general_system_template = r"""
# Given a specific context, please give a short answer to the question,
#  covering the required advices in general
#
# Context:
# ----
# {context}
# ----
# Please respond while maintaining the same writing style as used in this excerpt.
# Maintain the same language as the follow up input message.
# """
#
# # Was in the previous prompt
# #  and then provide the names all of relevant (even if it relates a bit) products.
#
# general_user_template = "Question:```{question}```"
# messages = [
#             SystemMessagePromptTemplate.from_template(general_system_template),
#             HumanMessagePromptTemplate.from_template(general_user_template)
# ]
# qa_prompt = ChatPromptTemplate.from_messages(messages)
#
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm_stream, retriever=retriever, memory=memory, verbose=True,
#     combine_docs_chain_kwargs={'prompt': qa_prompt}
# )


if 'conversation_starters' not in st.session_state:
    st.session_state['conversation_starters'] = get_conversation_starters(unique_topic_names)


@traceable(run_type="chain", project_name="RAG Assistant", tags=["LangChain", "RAG", "Chat_with_Docs"])
def handle_assistant_response(user_query):
    try:
        st.chat_message("user").write(user_query)
        with ((st.chat_message("assistant"))):
            # Retrieving the streamlit context to bind it to call back
            # in order to write in another threadcontext
            ctx = get_script_run_ctx()
            retrieval_handler = PrintRetrievalHandler(st.container(), ctx)
            # RETRIEVE THE CONTAINER TO CLEAR IT LATER to not show question twice
            e = st.empty()
            stream_handler = StreamHandler(e, ctx)
            # CODE WORKING BUT ALL LC API
            # THE CODE BELOW IS WORKING WITH RETRIEVER PRINT AND STREAMING
            # BUT ADDING A SYSTEM PROMPT SEEMS VERY TRICKY
            # OK SYSTEM PROMPT ADDED ABOVE ON QA_CHAIN WITH combine_docs_chain_kwargs
            # ai_response = qa_chain.invoke({"question": user_query},
            #                               {"configurable": {"session_id": sessionid},
            #                                   "callbacks": [
            #                                   retrieval_handler,
            #                                   stream_handler
            #                                 ]
            #                               },
            # )["answer"]
            # END CODE WORKING WITH ALL LC API

            # NEW API OF LANGCHAIN
            # PROMPT NEED TO BE CHANGED
            response = conversational_rag_chain.invoke(
                input={"input": user_query, "topics": topics},
                config={
                    "configurable": {"session_id": sessionid},
                    "callbacks": [
                        retrieval_handler,
                        stream_handler
                    ]
                },
            )
            ai_response = response["answer"]
            # emptying container to remove initial question that is render by llm
            e.empty()
            with e.container():
                st.markdown(ai_response)
                context = response["context"]
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
                    #show_retrievals = st.checkbox("Show PDFs")
                    try:
                        with st.expander(f"Source: {filename}", expanded=True):
                          file_object = get_file(filename, CollectionType.DOCUMENTS.value)
                          pdf_viewer(file_object,
                                      height=400,
                                      pages_to_render=pages)
                    except FileNotFoundError as fnfe:
                        full_path = fnfe.filename
                        filename = os.path.basename(full_path)
                        print(f"Erreur sur l'affichage du PDF: {fnfe}")
                        st.error(f"Le fichier PDF '{filename}' est manquant.")

            feedback_logger.info(f"User Query: {user_query}, AI Response: {ai_response}")
    except NotFoundError as e:
        print(f"Erreur sur  opensearch: {e}")
        st.error(f"La base de connaissance n'a pas été initialisée.\n\n"
                 f"Source: {e}")


def suggestion_clicked(question):
    st.session_state.user_suggested_question = question


def main():
    st.title("Dialogue avec les Connaissances")

    msgs = get_session_history(sessionid)

    suggested_questions = st.session_state['conversation_starters']

    if len(msgs.messages) == 0 or st.sidebar.button("Effacer la conversation"):
        msgs.clear()
        #msgs.add_ai_message("Comment puis-je vous aider?")


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
            streamlit_feedback(feedback_type = "thumbs",
                               optional_text_label="Cette réponse te convient?",
                               key=f"feedback_{i}",
                               on_submit=lambda x: _submit_feedback(x, emoji="👍"))


    # Handle suggested questions
    if "user_suggested_question" in st.session_state:
        user_query = st.session_state.user_suggested_question
        st.session_state.pop("user_suggested_question")  # Clear the session state
        handle_assistant_response(user_query)

    #Handle user queries
    if user_query := st.chat_input(placeholder="Pose-moi toutes tes questions!"):
        handle_assistant_response(user_query)


if __name__ == "__main__":
    if not check_password():
        # Do not continue if check_password is not True.
        st.stop()
    main()
