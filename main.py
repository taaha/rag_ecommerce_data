from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="ecommerce_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_ecommerce_db",
)
retriever = vector_store.as_retriever()
system_prompt = """You are an assistant for ecommerce website. 
Use the following pieces of retrieved context to answer the question. 
Do not make up anything by yourself even if specifications slightly differ.
If you don't know the answer, just say that you don't know.
If product record does not match with question, say the item is not yet available.  
Use three sentences maximum and keep the answer concise.
Context: {context}:"""
model = ChatOpenAI(model="gpt-4o", temperature=0)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query_rag/")
async def query_rag(question: str = Body(...)):    
    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Combine the documents into a single string
    docs_text = "".join(d.page_content for d in docs)

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # Generate a response
    response = model.invoke([SystemMessage(content=system_prompt_fmt),
                            HumanMessage(content=question)])
    
    # Get top matched record
    top_matched_record = None
    if len(vector_store.similarity_search(question)) > 0:
        top_matched_record = vector_store.similarity_search(question)[0].page_content

    return {
        "response": response.content,
        "top_matched_record": top_matched_record
    }