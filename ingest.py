import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


DOCUMENTS_DIR = "documents"
VECTOR_STORE_DIR = "vector_store"
OLLAMA_MODEL = "mistral:7b-instruct-q4_K_M"


def load_documents():
    docs = []

    for file in os.listdir(DOCUMENTS_DIR):
        path = os.path.join(DOCUMENTS_DIR, file)

        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

        elif file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

        elif file.endswith(".csv"):
            docs.extend(CSVLoader(path).load())

    return docs


def main():
    print("📄 Loading documents...")
    docs = load_documents()

    if not docs:
        print("❌ No documents found")
        return

    print(f"✅ Loaded {len(docs)} document pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} text chunks")

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    print("📦 Storing vectors in FAISS...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_STORE_DIR)

    print("🎉 Vector store created successfully!")


if __name__ == "__main__":
    main()
