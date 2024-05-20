from langchain_groq import ChatGroq
from dotenv import load_dotenv
from helper import (
  parse_pdfs,
  load_output,
  create_chat_chain
)

import joblib
import os
import time
import atexit
# import shutil

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
  # if True, files are uploaded
  # if False, files are not uploaded
  st.session_state.is_uploaded = True
  
  # check if output.md exists
  if "output_exists" in st.session_state:
    # if True, output.md exists
    # if False, output.md does not exist
    
    # if output.md exists, check the files list
    # collect all the uploaded PDF files and add "TEMP_" for comparsion
    temp_files_list = []
    for uploaded_file in uploaded_files:
      if not uploaded_file.name.endswith('.pdf'):
        filename = 'TEMP_' + uploaded_file.name + '.pdf'
      else:
        filename = 'TEMP_' + uploaded_file.name
      temp_files_list.append(filename)
    
    temp_files_list.sort()
    st.session_state.files_list.sort()
    # check if this matches with list already present
    if temp_files_list == st.session_state.files_list:
      # if True, no need to parse again, we are ready to answer questions
      # if False, we need to parse the files again
      print("No PDF files have been modified...")
      print("Existing PDFs in vector database: ", uploaded_files)
      st.write("Ready for questions!")
    else:
      # parse all the PDFs
      # create output.md
      # create chat chain
      
      # we should ideally only add the new PDFs, but it is tricky to implement
      # i will freshly parse all the new PDFs
      
      # in this case, an old output.md files should already exist
      # we want to remove it first
      print("New PDF files have been uploaded!")
      print("Existing PDFs in vector database: ", st.session_state.files_list)
      print("New PDFs in vector database: ", uploaded_files)
      
      print("Removing old output.md file...")
      os.remove('TEMP_output.md')
      
      # parse the PDFs
      parse_pdfs(
        uploaded_files=uploaded_files,
        parsing_instruction=parsing_instruction
      )
      # create output.md
      vectorstore = load_output()
      # mark that output.md has been created
      st.session_state.output_exists = True
      # create chat chain
      qa_model = create_chat_chain(
        vectorstore=vectorstore,
        GROQ_API_KEY=GROQ_API_KEY
      )
      # set qa on session_state
      st.session_state.qa = qa_model
  else:
    # output.md does not exist
    # we need to parse all the PDFs 
    
    # we don't need to remove any output.md files here
    parse_pdfs(
      uploaded_files=uploaded_files,
      parsing_instruction=parsing_instruction
    )
    # create output.md
    vectorstore = load_output()
    # mark that output.md has been created
    st.session_state.output_exists = True
    # create chat chain
    qa_model = create_chat_chain(
      vectorstore=vectorstore,
      GROQ_API_KEY=GROQ_API_KEY
    )
    # set qa on session_state
    st.session_state.qa = qa_model
else:
  # no files have been uploaded yet
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
  # chroma_folder = ".chroma"
  # shutil.rmtree(chroma_folder)

atexit.register(cleanup_dir)

