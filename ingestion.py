import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from uuid import uuid4

# Load variables from .env file
load_dotenv()

# Create csv. Makes single row chunking easy
file_path = 'data/data_file.xlsx'
df = pd.read_excel(file_path)
df.to_csv('data/data_file.csv', index=False)

# Data loading
loader = CSVLoader(file_path="data/data_file.csv")
data = loader.load()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="ecommerce_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_ecommerce_db",
)

# Embed data to vector store
uuids = [str(uuid4()) for _ in range(len(data))]
vector_store.add_documents(documents=data, ids=uuids)