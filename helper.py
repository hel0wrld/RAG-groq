from llama_parse import LlamaParse
from langchain.prompts import PromptTemplate
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
