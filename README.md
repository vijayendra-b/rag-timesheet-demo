# RAG Timesheet Approval System

## Objective

This project demonstrates Retrieval-Augmented Generation (RAG) for context-aware timesheet validation.

## Features

- Policy-based decision making
- Historical case retrieval
- Semantic search using FAISS
- HuggingFace embeddings
- LLM reasoning using Ollama

## Technology Stack

- Python 3.11
- LangChain
- FAISS
- Sentence Transformers
- Ollama (Llama 3.2)

## Project Structure

```
rag-timesheet-demo/
│
├── data/
│   ├── policy_1.txt
│   ├── policy_2.txt
│   ├── past_decision_1.txt
│   └── past_decision_2.txt
│
├── build_vectordb.py
├── rag_app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── faiss_index/   (excluded)
```

## Sample Output

Decision: Reject

Reason:
Employee submitted 12 hours without manager approval.

Policy:
Timesheets exceeding 10 hours require manager approval.

Historical Case:
Previous similar case was rejected.

### Screenshot

<img width="2914" height="1252" alt="image" src="https://github.com/user-attachments/assets/98236fa1-fc4f-4be6-b925-03629e448404" />

