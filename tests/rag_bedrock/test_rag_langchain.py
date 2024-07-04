
import numpy as np
import pytest


from rag_bedrock.base import LangchainTestRAGHelper


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "eval_questions_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainClaude3SonnetTitanEmbedV1(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Claude_3_Sonnet_Titan_Embed_V1"

    @property
    def model_id(self):
        return "anthropic.claude-3-sonnet-20240229-v1:0"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v1"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "eval_questions_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainClaude3SonnetTitanEmbedV2(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Claude_3_Sonnet_Titan_Embed_V2"

    @property
    def model_id(self):
        return "anthropic.claude-3-sonnet-20240229-v1:0"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v2:0"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainMistralLargeTitanEmbedV1(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Mistral_Large_Titan_Embed_V1"

    @property
    def model_id(self):
        return "mistral.mistral-large-2402-v1:0"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v1"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainMistralLargeTitanEmbedV2(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Mistral_Large_Titan_Embed_V2"

    @property
    def model_id(self):
        return "mistral.mistral-large-2402-v1:0"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v2:0"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainMistralLargeTitanEmbedMultiModal(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Mistral_Large_Titan_Multimodal"

    @property
    def model_id(self):
        return "mistral.mistral-large-2402-v1:0"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-image-v1"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLangChainMistralLargeCohereEmbedMultiLingual(LangchainTestRAGHelper):

    @property
    def test_name(self):
        return "Langchain_Mistral_Large_Cohere_Embed"

    @property
    def model_id(self):
        return "mistral.mistral-large-2402-v1:0"

    @property
    def embedding_model_id(self):
        return "cohere.embed-multilingual-v3"
