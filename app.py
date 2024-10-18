import streamlit as st
from utils import text, embeddings_factory
from utils.embeddings_factory import EmbeddingsFactory
from streamlit_chat import message
from dotenv import load_dotenv


def main():
    
    load_dotenv()
    
    provider = "openai"    
    embeddings = EmbeddingsFactory.get_embeddings(provider)

    st.set_page_config(page_title="ChatPDF", page_icon=":books:", layout="wide")
    
    st.header("ChatPDF")
    user_question = st.text_input("Faça sua pergunta.")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if user_question:
        response = st.session_state.conversation(user_question)["chat_history"]
        
        for i, text_message in enumerate(response):
            if i % 2 == 0:
                message(text_message.content, is_user=True, key=str(i) + "_user")
            else:
                message(text_message.content, is_user=False, key=str(i) + "_ai")
    
    with st.sidebar:
        st.subheader("Seus arquivos")
        pdf_docs = st.file_uploader("Carregue seus PDFs aqui...", accept_multiple_files=True)

        if st.button("Processar"):
            with st.spinner("Processando"):
                # Processar os PDFs
                raw_text = text.get_text_files(pdf_docs)
                text_chunks = text.get_text_chunks(raw_text)
                vectorstore = embeddings.get_vector_store(text_chunks)
                st.session_state.conversation = embeddings.create_conversation_chain(vectorstore)
                
                st.success("Done")
        
if __name__ == '__main__':
    main()
