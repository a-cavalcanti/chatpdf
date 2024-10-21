from utils.embeddings_base import BaseEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings


class EmbeddingsBedrock(BaseEmbeddings):

    def get_vector_store(self, chuncks):    
        embeddings = EmbeddingsBedrock(
            credentials_profile_name="bedrock", region_name="us-east-1"
        )
        vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
        return vectorstore

    def create_conversation_chain(self, vectorstore):
        
        llm = ChatBedrock(
            region_name = "us-east-1",
            model_kwargs={
                "temperature":1,
                "top_k":250,
                "top_p":0.999,
                "anthropic_version":"bedrock-2023-05-31"
                },
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            credentials_profile_name="bedrock"
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        return conversation
