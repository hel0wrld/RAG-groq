from llama_parse import LlamaParse
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path

import streamlit as st

import joblib
import time
# from dotenv import load_dotenv

import os

# load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {input}

Make sure to explain the answer in great detail, covering as much information as possible
"""


def iter_files_pdf(directory: str):
  """Generator to iterate over all pdf files in a directory."""
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.pdf'):
      yield filename


def iter_files_pkl(directory: str):
  """Generator to iterate over all pickle files in a directory."""
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path) and filename.endswith(".pkl"):
      yield filename


def parsed_doc(directory: str, parsing_instruction: str, LLAMA_CLOUD_API_KEY: str):
  """Returns parsed documents for each file"""
  parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY, 
    parsing_instruction=parsing_instruction,
    result_type="markdown" 
  )
  document = parser.load_data(directory)
  return document


def set_custom_prompt(custom_prompt_template: str):
  """
  Prompt template for QA retrieval for each vectorstore
  """
  prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
  )
  return prompt

def parse_pdfs(uploaded_files, parsing_instruction: str):
  """
  Parse all pdfs in the current directory
  
  Args:
  uploaded_files: list of uploaded files
  parsing_instruction: instruction for parsing the PDFs
  """

  for uploaded_file in uploaded_files:
    if not uploaded_file.name.endswith('.pdf'):
      filename = 'TEMP_' + uploaded_file.name + '.pdf'
    else:
      filename = 'TEMP_' + uploaded_file.name
    st.session_state.files_list.append(filename)
    save_path = os.path.join(os.getcwd(), filename)
    # if file is already saved, don't save it again
    if os.path.exists(save_path):
      print(f"\"{uploaded_file.name}\" already exists!")
      continue
    else:
      print(f"\"{uploaded_file.name}\" does not exist! saving...")
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
        
    
def load_output():
  """
  Load output.md
  """
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
    
  return vectorstore


def create_chat_chain(vectorstore, GROQ_API_KEY: str):
  """
  Create a chat chain
  """
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
    
  return qa
