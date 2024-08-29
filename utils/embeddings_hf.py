from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain.llms import HuggingFaceHub


def get_vector_store(chuncks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
    return vectorstore

def get_vector_store_aws(chuncks):    
    embeddings = BedrockEmbeddings(
        credentials_profile_name="bedrock", region_name="us-east-1"
    )
    vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", 
                         model_kwargs={
                             "temperature": 0.1, 
                             "max_length": 512, 
                             "truncate": True
                            }
                        )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation

def create_conversation_chain_aws(vectorstore):
    
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