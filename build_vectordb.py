from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


loader = DirectoryLoader(
    "data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
	chunk_size=500,
	chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_index")

print("Vector database created.")
