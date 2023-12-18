# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:48:51 2023

@author: msamwelmollel
"""

import logging
import sys
from sentence_transformers import SentenceTransformer

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext, set_global_service_context
from llama_index.embeddings.langchain import LangchainEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_cpp import Llama


documents = SimpleDirectoryReader("data/").load_data()


llm = LlamaCPP(
    # Optionally, you can pass in the URL to a GGML model to download it automatically
    # model_url=None,
    # Set the path to a pre-downloaded model instead of model_url
    model_path='mixtral-8x7b-instruct-v0.1.Q2_K.gguf',
    temperature=0.1,
    max_new_tokens=1024, # Increasing to support longer responses
    context_window=8192, # Mistral7B has an 8K context-window
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)




embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    ## you can select sentence transfomer embedding also
  # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)



service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)


index = VectorStoreIndex.from_documents(documents, service_context=service_context)


query_engine = index.as_query_engine()
response = query_engine.query("Who is Michael Mollel")

print(response)

while True:
  query=input()
  response = query_engine.query(query)
  print(response)

