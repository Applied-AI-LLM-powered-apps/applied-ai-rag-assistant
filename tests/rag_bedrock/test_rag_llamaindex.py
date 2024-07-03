import pytest

from rag_bedrock.base import LlamaIndexTestRAGHelper


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "temp_dir",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLlamaIndexClaude3(LlamaIndexTestRAGHelper):

    @property
    def test_name(self):
        return "LlamaIndex_Claude_3_Sonnet_Titan_Embed_V1"

    @property
    def model_id(self):
        return "anthropic.claude-3-sonnet-20240229-v1:0"

    @property
    def topic(self):
        return "How to Build a Career in AI"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v1"


@pytest.mark.usefixtures("trulens_prepare",
                         "bedrock_prepare",
                         "temp_dir",
                         "documents_prepare",
                         "llm_prepare",
                         "embeddings_prepare",
                         "trulens_context_prepare",
                         "provider_prepare",
                         "eval_questions_prepare",
                         "rag_prepare",
                         "feedbacks_prepare")
class TestRAGLlamaIndexMistral(LlamaIndexTestRAGHelper):

    @property
    def test_name(self):
        return "LlamaIndex_Mistral_Large_Titan_Embed_V1"

    @property
    def model_id(self):
        return "mistral.mistral-large-2402-v1:0"

    @property
    def topic(self):
        return "How to Build a Career in AI"

    @property
    def embedding_model_id(self):
        return "amazon.titan-embed-text-v1"
