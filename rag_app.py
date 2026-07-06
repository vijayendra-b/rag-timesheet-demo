from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# User Input
query1 = """
Employee logged 12 hours on Monday.
No manager approval attached.
"""

query = """
Employee logged 9 hours on Monday.
Manager approval is attached.
"""

# Retrieve similar docs
docs = db.similarity_search(query, k=3)

print("\n===== RETRIEVED DOCUMENTS =====\n")

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)

# Build Context
context = "\n\n".join([doc.page_content for doc in docs])

# Prompt
prompt = f"""
You are an intelligent Timesheet Approval Assistant.

Use ONLY the retrieved context below to make a decision.

Retrieved Context:
{context}

New Timesheet Request:
{query}

Tasks:
1. Decide Approve or Reject
2. Explain why
3. Mention relevant policy
4. Mention similar historical case if available
5. Provide confidence level

Format:

Decision:
Explanation:
Policy Reference:
Similar Case:
Confidence:
"""

# Call Llama3
response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("\n")
print("-" * 80)
print("YOUR QUERY")
print("-" * 80)
print(query)

print("=" * 80)
print("LLM DECISION")
print("=" * 80)

print(response["message"]["content"])
print("\n")