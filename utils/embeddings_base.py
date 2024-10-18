from abc import ABC, abstractmethod

class BaseEmbeddings(ABC):
    
    @abstractmethod
    def get_vector_store(self):
        """
        Método abstrato para obter o vetor de embeddings.
        """
        pass

    @abstractmethod
    def create_conversation_chain(self):
        """
        Método abstrato para criar uma cadeia de conversação.
        """
        pass
