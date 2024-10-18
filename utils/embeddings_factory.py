from embeddings_huggingface import HuggingFaceEmbeddings
from embeddings_openai import OpenAIEmbeddings
from embeddings_bedrock import BedrockEmbeddings

class EmbeddingsFactory:
    
    @staticmethod
    def get_embeddings(provider: str):
        if provider == 'huggingface':
            return HuggingFaceEmbeddings()
        elif provider == 'openai':
            return OpenAIEmbeddings()
        elif provider == 'bedrock':
            return BedrockEmbeddings()
        else:
            raise ValueError(f"Provider {provider} não é suportado.")