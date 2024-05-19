from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pathlib import Path
from dotenv import load_dotenv
from helper import (
  iter_files_pdf,
  iter_files_pkl,
  parsed_doc,
  set_custom_prompt,
  custom_prompt_template
)

import joblib
import os
import time
import atexit
import shutil

import streamlit as st

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.header("PDF Chatbot using Groq and Chroma")


# PDF upload here
parsing_instruction = st.sidebar.text_input(
  label="Enter parsing instructions for the files",
  placeholder="This is a research paper"
)

uploaded_files = st.sidebar.file_uploader(
  label="Upload your PDF files here",
  accept_multiple_files=True
)

# check if PDF exists and update session accordingly
if len(uploaded_files) != 0:
  st.session_state.is_uploaded = True
  # check if output.md exists
  st.session_state.output_exists = os.path.exists('TEMP_output.md')
  
  # if output.md exists, check for qa chain
  if st.session_state.output_exists:
    if "qa" in st.session_state:
      # if qa chain exists, answer using that
      print("QA Chain already exists!")
    else:
      # if qa chain does not exist, create it
      with st.spinner(text="Loading output.md..."):
        loader = UnstructuredMarkdownLoader('TEMP_output.md')
      
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=2000,
          chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents=documents)
        
        embed_model = FastEmbedEmbeddings()
        vectorstore = Chroma.from_documents(
          documents=docs,
          collection_name='pdf_collection',
          embedding=embed_model
        )
        
      with st.spinner(text="Creating chat model..."):
        # create chain
        chat_model = ChatGroq(
          model_name='llama3-8b-8192',
          api_key=GROQ_API_KEY
        )
        
        retriever = vectorstore.as_retriever(
          search_kwargs={'k':3}
        )
        
        prompt = set_custom_prompt(custom_prompt_template=custom_prompt_template)
        
        # create chat chain
        combine_docs_chain = create_stuff_documents_chain(
          llm=chat_model, prompt=prompt
        )
        
        qa = create_retrieval_chain(
          retriever=retriever,
          combine_docs_chain=combine_docs_chain
        )
        
      st.session_state.qa = qa
  else:
    # output.md does not exist but PDFs are available
    # parse them and create the model
    
    # if files are uploaded, save them
    if st.session_state.is_uploaded:
      for uploaded_file in uploaded_files:
        if not uploaded_file.name.endswith('.pdf'):
          filename = 'TEMP_' + uploaded_file.name + '.pdf'
        else:
          filename = 'TEMP_' + uploaded_file.name
        save_path = os.path.join(os.getcwd(), filename)
        # if file is already saved, don't save it again
        if os.path.exists(save_path):
          print(f"{uploaded_file.name} already exists!")
          continue
        else:
          print(f"{uploaded_file.name} does not exist! saving...")
          with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    with st.spinner(text="Parsing the PDFs..."):
      directory = os.getcwd()
      for filename in iter_files_pdf(directory=directory):
        document = parsed_doc(
          directory=filename,
          parsing_instruction=parsing_instruction,
          LLAMA_CLOUD_API_KEY=LLAMA_CLOUD_API_KEY
        )
        joblib.dump(document, filename[:-4] + ".pkl")
        time.sleep(5)

      Path('TEMP_output.md').touch()
      
      # add the saved pickles to output.md
      for filename in iter_files_pkl(directory=directory):
        docs = joblib.load(filename)
        with open('TEMP_output.md', 'a', encoding='utf-8') as f:
          for doc in docs:
            f.write(doc.text + '\n')
            
    with st.spinner(text="Loading output.md..."):
      loader = UnstructuredMarkdownLoader('TEMP_output.md')

      documents = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100
      )
      docs = text_splitter.split_documents(documents)
            
      embed_model = FastEmbedEmbeddings()
      vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name='pdf_collection',
        embedding=embed_model
      )
      
    with st.spinner(text="Creating chat model..."):
      # create chain
      chat_model = ChatGroq(
        model_name='llama3-8b-8192',
        api_key=GROQ_API_KEY
      )
      
      retriever = vectorstore.as_retriever(
        search_kwargs={'k':3}
      )
      
      prompt = set_custom_prompt(custom_prompt_template=custom_prompt_template)
      
      # create chat chain
      combine_docs_chain = create_stuff_documents_chain(
        llm=chat_model, prompt=prompt
      )
      
      qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
      )
    
    st.session_state.qa = qa
else:
  # no files have been uploaded
  st.session_state.is_uploaded = False
        

# create a text input box
query_input = st.text_input(
  label='Ask a question about the PDFs',
  placeholder='What is the meaning of life?',
  autocomplete='What is the meaning of life?'
)

if query_input and "qa" not in st.session_state:
  # user has a query but there is no existing qa
  
  # create a normal chat instance
  chat_model = ChatGroq(
    model_name='llama3-8b-8192',
    api_key=GROQ_API_KEY
  )
  response = chat_model.invoke(
    input=query_input
  )
  st.markdown("*No PDFs uploaded!*")
  st.write(response.content)
  
if query_input and "qa" in st.session_state:
  # user has a query and there is an existing qa
  response = st.session_state.qa.invoke(
    {
      'input': query_input
    }
  )
  st.write(response['answer'])
  print(response)

# delete the files on session end
def cleanup_dir():
  for filename in os.listdir(os.getcwd()):
    if filename.startswith('TEMP_'):
      os.remove(filename)
  chroma_folder = ".chroma"
  shutil.rmtree(chroma_folder)

atexit.register(cleanup_dir)

