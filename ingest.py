
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load PDF
loader = PyPDFLoader("data/customers.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store in vector DB
db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./db"
)

db.persist()

print("✅ Data stored in vector DB")
