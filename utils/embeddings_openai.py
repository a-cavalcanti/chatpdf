from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_vector_store(chuncks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chuncks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.0)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation
