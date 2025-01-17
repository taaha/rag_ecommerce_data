# rag_ecommerce_data


This code is to set up a fastapi server for RAG powered chatbot for ecommerce. `gpt4o` is used as LLM and `openai-text-embedding-3-large` is used as embedding model.
## Machine

Ubuntu-22.04, Python-3.10, Fastapi


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`


## Setting up
Following commands need to be run when deploying to a new machine for the first time.

To install dependencies run

```bash
  pip install -r requirements.txt
```

To ingest documents run the following command. 
```bash
  python ingestion.py
```
Above command will create a chroma database and embed documents in it. Above command should only be run once to avoid duplicates.
## Starting server

to start server (in dev mode)

```bash
  fastapi dev
```

now you can make queries from http://127.0.0.1:8000/docs
