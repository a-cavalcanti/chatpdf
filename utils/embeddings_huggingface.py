from utils.embeddings_base import BaseEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


class EmbeddingsHuggingFace(BaseEmbeddings):
    def get_vector_store(self, chuncks):
        embeddings = HuggingFaceInstructEmbeddings(model_name="whereIsAI/UAE-Large-Vl")
        vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
        return vectorstore

    def create_conversation_chain(self, vectorstore):
        llm = HuggingFaceHub(repo_id="google/flan-t5-large", 
                            model_kwargs={"max_length":512, "temperature": 0.0})
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        return conversation