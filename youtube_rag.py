import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def transcript(link):
    url = link
    save_dir = "C:/Users/mpooj/OneDrive/Desktop/gen_ai/"

    
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser(api_key=os.getenv("OPENAI_API_KEY"))
    )
    docs = loader.load()
    return docs

def text_chunks(transcripts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False
    )
    splited_docs = splitter.split_documents(transcripts)
    return splited_docs

def embedding(chunks):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = Chroma.from_documents(
        chunks, embeddings  
    )
    return vectordb

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def generate_response(user_input, embeddings, temperature, max_tokens):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature, max_tokens=max_tokens)
    retriever = embeddings.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )
    result = chain.invoke(user_input)
    return result['answer']

st.title("YouTube Video Q&A")

link = st.sidebar.text_input("YouTube Video Link")
button = st.sidebar.button("Enter")

if button and link:
    transcripts = transcript(link)
    chunks = text_chunks(transcripts)
    embeddings = embedding(chunks)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

user_input = st.text_input("Go ahead and ask any question")

if user_input:
    response = generate_response(user_input, embeddings, temperature, max_tokens)
    st.write(response)
