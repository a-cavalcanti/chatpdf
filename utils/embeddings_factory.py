from utils.embeddings_huggingface import EmbeddingsHuggingFace
from utils.embeddings_openai import EmbeddingsOpenAI
from utils.embeddings_bedrock import EmbeddingsBedrock

class EmbeddingsFactory:
    
    @staticmethod
    def get_embeddings(provider: str):
        if provider == 'huggingface':
            return EmbeddingsHuggingFace()
        elif provider == 'openai':
            return EmbeddingsOpenAI()
        elif provider == 'bedrock':
            return EmbeddingsBedrock()
        else:
            raise ValueError(f"Provider {provider} não é suportado.")