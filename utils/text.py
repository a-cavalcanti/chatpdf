from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_text_files(files):
    text = ""
    
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks