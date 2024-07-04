from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from langchain.docstore.document import Document

from langchain.schema.prompt_template import format_document

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from functools import partial
from operator import itemgetter

from langchain.callbacks.manager import trace_as_chain_group

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

from .config_loader import load_config

config = load_config()

# "local:BAAI/bge-small-en-v1.5"


# https://python.langchain.com/docs/use_cases/question_answering/
# https://python.langchain.com/docs/modules/chains/document/stuff
# https://python.langchain.com/docs/modules/chains/document/map_reduce
# https://python.langchain.com/docs/modules/chains/document/refine
# https://python.langchain.com/docs/modules/chains/document/map_rerank
def invoke(question: str, template: str, llm: BaseChatModel, chain_type: str, vectorstore: VectorStore,
           search_type: str, k: int, verbose: bool):

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': k})
    docs = retriever.invoke(question)
    output = None

    if verbose:
        print(docs)

    document_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_document = partial(format_document, prompt=document_prompt)
    # temporary to replace with incoming question

    map_prompt = PromptTemplate.from_template(
        "Answer the user question using the context."
        "\n\nContext:\n\n{context}\n\nQuestion: {question}"
    )

    rag_prompt_custom = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    if chain_type == "stuff":
        print("stuff chain")

        rag_chain = (
                {
                    "context": lambda x: "\n\n".join(
                        format_document(doc, document_prompt) for doc in x["docs"]
                    ),
                    "question": itemgetter("question"),
                }
                | map_prompt
                | llm
                | StrOutputParser()
        )
        output = rag_chain.invoke({"docs": docs, "question": question})

    elif chain_type == "map_reduce":

        print("map_reduce chain")

        # PromptTemplate.from_template("Summarize this content:\n\n{context}")
        first_prompt = PromptTemplate.from_template(
            "Answer the user question using the context."
            "\n\nContext:\n\n{context}\n\nQuestion: " + question
        )
        # first_prompt = first_prompt.format_prompt(question=question)

        # The chain we'll apply to each individual document.
        map_chain = (
                {"context": partial_format_document}
                | first_prompt
                | llm
                | StrOutputParser()
        )

        # A wrapper chain to keep the original Document metadata
        map_as_doc_chain = (
                RunnableParallel({"doc": RunnablePassthrough(), "context": map_chain})
                | (
                    lambda x: Document(page_content=x["context"], metadata=x["doc"].metadata)
                )
        ).with_config(run_name="Summarize (return doc)")

        # The chain we'll repeatedly apply to collapse subsets of the documents
        # into a consolidate document until the total token size of our
        # documents is below some max size.
        collapse_chain = (
                {"context": format_docs}
                | PromptTemplate.from_template("Collapse this content:\n\n{context}")
                | llm
                | StrOutputParser()
        )

        def get_num_tokens(docs):
            return llm.get_num_tokens(format_docs(docs))

        def collapse(
                docs,
                config,
                token_max=4000,
        ):
            collapse_ct = 1
            while get_num_tokens(docs) > token_max:
                config["run_name"] = f"Collapse {collapse_ct}"
                invoke = partial(collapse_chain.invoke, config=config)
                split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
                docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
                collapse_ct += 1
            return docs

        # The chain we'll use to combine our individual document summaries
        # (or summaries over subset of documents if we had to collapse the map results)
        # into a final summary.
        reduce_chain = (
                {"context": format_docs}
                | PromptTemplate.from_template("Combine these answers:\n\n{context}")
                | llm
                | StrOutputParser()
        ).with_config(run_name="Reduce")

        # The final full chain
        rag_chain = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(
            run_name="Map reduce"
        )

        output = rag_chain.invoke(docs, config={"max_concurrency": 5})

    elif chain_type == "map_rerank":
        print("map_reduce chain")

        # Chain to apply to each individual document. Chain
        # provides an answer to the question based on the document
        # and scores it's confidence in the answer.
        class AnswerAndScore(BaseModel):
            """Return the answer to the question and a relevance score."""

            answer: str = Field(
                description="The answer to the question, which is based ONLY on the provided context."
            )
            score: float = Field(
                description="A 0.0-1.0 relevance score, where 1.0 indicates the provided context answers the question completely and 0.0 indicates the provided context does not answer the question at all."
            )

        function = convert_pydantic_to_openai_function(AnswerAndScore)
        map_chain = (
                map_prompt
                | ChatOpenAI().bind(
            temperature=0, functions=[function], function_call={"name": "AnswerAndScore"}
        )
                | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
        ).with_config(run_name="Map")

        # Final chain, which after answer and scoring based on
        # each doc return the answer with the highest score.

        def top_answer(scored_answers):
            return max(scored_answers, key=lambda x: x.score).answer

        # document_prompt = PromptTemplate.from_template("{page_content}")
        rag_chain = (
                (
                    lambda x: [
                        {
                            "context": format_document(doc, document_prompt),
                            "question": question,  # x["question"]
                        }
                        for doc in x["docs"]
                    ]
                )
                | map_chain.map()
                | top_answer
        ).with_config(run_name="Map rerank")

        output = rag_chain.invoke({"docs": docs, "question": question})

    elif chain_type == "refine":
        # first_prompt = PromptTemplate.from_template("Summarize this content:\n\n{context}")
        first_prompt = PromptTemplate.from_template(
            "Answer the user question using the context."
            "\n\nContext:\n\n{context}\n\nQuestion: " + question
        )
        document_prompt = PromptTemplate.from_template("{page_content}")
        partial_format_doc = partial(format_document, prompt=document_prompt)
        summary_chain = {"context": partial_format_doc} | first_prompt | llm | StrOutputParser()
        refine_prompt = PromptTemplate.from_template(
            "Answer the user question."
            "\n\nHere's your first summary: {prev_response}. "
            "\n\nNow add to it based on the following context: {context}\n\nQuestion: " + question
        )
        refine_chain = (
                {
                    "prev_response": itemgetter("prev_response"),
                    "context": lambda x: partial_format_doc(x["doc"]),
                }
                | refine_prompt
                | llm
                | StrOutputParser()
        )

        def refine_loop(docs):
            with trace_as_chain_group("refine loop", inputs={"input": docs}) as manager:
                summary = summary_chain.invoke(
                    docs[0], config={"callbacks": manager, "run_name": "initial summary"}
                )
                for i, doc in enumerate(docs[1:]):
                    summary = refine_chain.invoke(
                        {"prev_response": summary, "doc": doc},
                        config={"callbacks": manager, "run_name": f"refine {i}"},
                    )
                manager.on_chain_end({"output": summary})
            return summary

        output = refine_loop(docs)
    return output


def agent_lc_factory(chain_type, llm, search_kwargs, search_type, vectorstore):

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type=chain_type, retriever=retriever
    )
    return retrieval_qa_chain
