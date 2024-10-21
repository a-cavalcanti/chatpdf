from utils.embeddings_base import BaseEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


class EmbeddingsOpenAI(BaseEmbeddings):

    def load_vector_store(self):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local('faiss_openai', embeddings)
        return vectorstore
    
    def get_vector_store(self, chuncks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
        vectorstore.save_local('faiss_openai')
        return vectorstore

    def create_conversation_chain(self, vectorstore=None):

        if not vectorstore:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local('faiss_openai', embeddings)
        
        llm = ChatOpenAI(temperature=0.0)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        return conversation
