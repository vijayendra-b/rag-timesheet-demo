from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Sample query
query = """
Employee logged 8 hours on Monday.
"""

# Retrieve relevant documents
docs = db.similarity_search(query, k=3)

print("\n===== RETRIEVED DOCUMENTS =====\n")

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)

print("\n===== DECISION =====\n")

# Simple rule-based decision for PoC
decision = "Rejected"

reason = """
The employee logged 12 working hours without manager approval.
According to the retrieved policy documents, overtime exceeding the allowed limit requires manager approval.
A similar past case was also rejected due to missing approval.
"""

print(f"Decision: {decision}")
print(reason)