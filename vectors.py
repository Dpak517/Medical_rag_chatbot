import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "medical-chatbot-v2"

# 1. Loading PDF
print("Loading PDFs...")
loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split pages into small overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Embedding model: turns text into a 384-number vector
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create the Pinecone index 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
existing = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,          # must match the embedding model
        metric="cosine",        # cosine-similarity search
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# 5. Embed all chunks and upload them to Pinecone
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
)
print("Done, Knowledge base is ready.")
