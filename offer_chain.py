from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

# embedding model (must match ingest.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3", temperature=0.3)

prompt = PromptTemplate(
    input_variables=["context", "username"],
    template="""
You are a retail marketing assistant.

Customer name: {username}

Customer purchase history:
{context}

Based on their buying behavior generate:
- 3 personalized offers
- discount suggestions
- cross sell items
- loyalty recommendations
Keep response short.
"""
)

def get_offers(username):
    docs = retriever.invoke(username)
    context = "\n".join([d.page_content for d in docs])

    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "username": username
    })

    return result.content
