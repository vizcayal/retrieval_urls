import streamlit as st 
from langchain.document_loaders import UnstructuredURLLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI    
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from dotenv import load_dotenv
import os
import certifi
print(f'{certifi.where() =}')

load_dotenv()
api_key = os.getenv("OPENAI_KEY")
print(api_key)

NUM_URLS = 3
file_path = 'vectorstore.pkl'

st.title('Summary P&R Research')
st.sidebar.title('Articles')


urls = []
for i in range(NUM_URLS):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_urls_click = st.sidebar.button('Process')

if process_urls_click:
    loader = UnstructuredURLLoader(urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',',','.'],
        chunk_size = 1000
    )

    chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore_openai = FAISS.from_documents(chunks, embeddings)

    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore_openai, f)





    
