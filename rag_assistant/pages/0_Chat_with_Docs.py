import os
import threading
import streamlit as st
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
from utils.constants import Metadata
from utils.utilsdoc import get_store, extract_unique_name
from utils.config_loader import load_config
from streamlit_feedback import streamlit_feedback
import logging
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from utils.utilsllm import load_model
import time

load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')

# set logging
logger = logging.getLogger('')
logger.setLevel(logging.INFO)

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger.handlers.clear()
handler = logging.FileHandler(os.path.join(log_dir, 'feedback.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


config = load_config()
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']
top_k = int(config['LANGCHAIN']['SEARCH_TOP_K'])
search_type = config['LANGCHAIN']['SEARCH_TYPE']


# Generate session ID
def get_session_id():
    if "session_id" not in st.session_state:
        sesssion_id = "Nouvelle conversation 1"
        st.session_state.session_id = sesssion_id
    return st.session_state.session_id

# Retrieve chat history for a given session
def get_chat_history(session_id):
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = StreamlitChatMessageHistory(key=f"chat_history_{session_id}")
    return st.session_state.chat_histories[session_id]

#Generer les titres en utilisant le LLM
def generate_session_title(query):
    prompt = f"Créez une phrase concise de 3 à 5 mots comme en-tête de la requête suivante, en respectant strictement la limite de 3 à 5 mots Pas besoin de reformuler la requete en disant 'Voici une phrase comncise de 3 mots comme en-tete pour votre requete:' avant de l'afficher , affiche juste les 3 ou 5 mots: {query}"
    llm = load_model()
    response = llm.invoke(prompt)

    # Se rassurer que la reponse est une chaine de caractere 
    if isinstance(response, str):
        return response.strip()
    elif hasattr(response, 'content'):  
        return response.content.strip()
    else:
        raise ValueError("Unexpected response type from LLM")



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
#msgs = get_session_history(sessionid)
session_id = get_session_id()
msgs = get_chat_history(session_id)

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
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


if 'conversation_starters' not in st.session_state:
    st.session_state['conversation_starters'] = get_conversation_starters(unique_topic_names)

def _submit_feedback(user_response):
    feedback_score = '+1' if user_response['score'] == '👍' else '-1'
    feedback_text = user_response.get('text', '')
    logger.info(f"Feedback_Score: {feedback_score}, Feedback_text: {feedback_text}")
    st.toast(f"Feedback submitted: {user_response['score']} {feedback_text}", icon='✅')
    return user_response

@traceable(run_type="chain", project_name="RAG Assistant", tags=["LangChain", "RAG", "Chat_with_Docs"])
def handle_assistant_response(user_query):
    session_id = get_session_id()
    msgs = get_chat_history(session_id)
    
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        ctx = get_script_run_ctx()
        retrieval_handler = PrintRetrievalHandler(st.container(), ctx)
        e = st.empty()
        stream_handler = StreamHandler(e, ctx)
        
        response = conversational_rag_chain.invoke(
            input={"input": user_query, "topics": topics},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [
                    retrieval_handler,
                    stream_handler
                ]
            },
        )

        ai_response = response["answer"]
        #print(ai_response)
        e.empty()
        with e.container():
            st.markdown(ai_response)
            context = response["context"]
            metadata = [(doc.metadata['filename'], doc.metadata['page']) for doc in context]
            metadata_dict = {}
            for filename, page in metadata:
                if filename not in metadata_dict:
                    metadata_dict[filename] = []
                metadata_dict[filename].append(page)

            sorted_metadata = sorted(metadata_dict.items(), key=lambda x: len(x[1]), reverse=True)

            formatted_metadata = []
            for item in sorted_metadata:
                formatted_metadata.append([item[0], item[1]])
            if formatted_metadata:
                first_metadata = formatted_metadata[0]
                filename = first_metadata[0]
                pages = first_metadata[1]
                with st.expander(f"Source: {filename}", expanded=True):
                    pdf_viewer(f"{upload_directory}/{filename}",
                               height=400,
                               pages_to_render=pages)
        logger.info(f"User Query: {user_query}, AI Response: {ai_response}")

def suggestion_clicked(question):
    session_id = get_session_id()
    if session_id not in st.session_state.chat_titles or st.session_state.chat_titles[session_id] == session_id:
        title = generate_session_title(question)
        st.session_state.chat_titles[session_id] = title
    st.session_state.user_suggested_question = question



def main():
    st.title("Dialogue avec les connaissances")
    st.sidebar.title("Sessions de conversation")

    suggested_questions = st.session_state['conversation_starters']

    if "session_id" not in st.session_state:
        session_id = get_session_id()
    else:
        session_id = st.session_state.session_id

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if "chat_titles" not in st.session_state:
        st.session_state.chat_titles = {}

    if "new_chat_counter" not in st.session_state:
        st.session_state.new_chat_counter = 1

    chat_sessions = list(st.session_state.chat_histories.keys())

    if st.sidebar.button("Nouvelle Conversation"):
        st.session_state.new_chat_counter += 1
        session_id = f"Nouvelle Conversation {st.session_state.new_chat_counter}"
        st.session_state.session_id = session_id
        st.session_state.chat_histories[session_id] = StreamlitChatMessageHistory(key=f"chat_history_{session_id}")
        st.session_state.chat_titles[session_id] = session_id  # Use session ID as a temporary title
        st.rerun()

    selected_session = session_id
    session_deleted = False

    # Reverse the chat sessions to display the newest first
    for chat_session in reversed(chat_sessions):
        title = st.session_state.chat_titles.get(chat_session, chat_session)
        with st.sidebar:
            with st.expander(title):
                if st.button(f"🚮 Supprimer", key=f"delete_{chat_session}"):
                    if chat_session in st.session_state.chat_histories:
                        del st.session_state.chat_histories[chat_session]
                    if chat_session in st.session_state.chat_titles:
                        del st.session_state.chat_titles[chat_session]
                    session_deleted = True
                    if selected_session == chat_session:
                        selected_session = None
                    break
                else:
                    if st.button(f"{title}", key=f"select_{chat_session}"):
                        selected_session = chat_session
                        st.session_state.session_id = selected_session
                        st.rerun()

    if session_deleted:
        chat_sessions = list(st.session_state.chat_histories.keys())
        if chat_sessions:
            selected_session = chat_sessions[0]
        else:
            selected_session = None
            st.session_state.new_chat_counter = 1
            session_id = f"Nouvelle Conversation {st.session_state.new_chat_counter}"
            st.session_state.chat_histories[session_id] = StreamlitChatMessageHistory(key=f"chat_history_{session_id}")
            st.session_state.chat_titles[session_id] = session_id
            selected_session = session_id
        st.session_state.session_id = selected_session
        st.rerun()

    if selected_session != session_id:
        session_id = selected_session
        st.session_state.session_id = session_id

    msgs = get_chat_history(session_id) if session_id else None

    st.subheader("Amorces de conversation", divider="rainbow")
    col1, col2 = st.columns(2)
    for i, question in enumerate(suggested_questions, start=1):
        col = col1 if i % 2 != 0 else col2
        col.button(question, on_click=suggestion_clicked, args=[question])

    if msgs and len(msgs.messages) > 0:
        avatars = {"human": "user", "ai": "assistant"}
        for i, msg in enumerate(msgs.messages):
            st.chat_message(avatars[msg.type]).write(msg.content)
            if msg.type == "ai":
                feedback = streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="[Optional] Est-ce que cette réponse vous convient ?",
                    key=f"feedback_{session_id}_{i+1}",
                    on_submit= _submit_feedback
                )
                if feedback:
                    st.success("Merci pour votre feedback!")

    if "user_suggested_question" in st.session_state:
        user_query = st.session_state.user_suggested_question
        st.session_state.pop("user_suggested_question")
        handle_assistant_response(user_query)
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Est-ce que cette réponse vous convient ?",
            key=f"feedback_{session_id}_{len(msgs.messages)}",
        )

    if user_query := st.chat_input(placeholder="Pose moi tes questions!"):
        if session_id not in st.session_state.chat_titles or st.session_state.chat_titles[session_id] == session_id:
            title = generate_session_title(user_query)
            st.session_state.chat_titles[session_id] = title
        handle_assistant_response(user_query)
        streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Est-ce que cette réponse vous convient ?",
            key=f"feedback_{session_id}_{len(msgs.messages)}",
        )

    

if __name__ == "__main__":
    main()