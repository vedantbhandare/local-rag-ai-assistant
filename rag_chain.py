from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate


VECTOR_STORE_DIR = "vector_store"
OLLAMA_MODEL = "mistral:7b-instruct-q4_K_M"


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.

Use the context below to answer the question.
You may summarize or infer ONLY from the context.
Do NOT add external information.

If the context does not contain the answer, say "I don't know".

Context:
{context}

Question:
{question}

Answer (clear and concise):
"""
)


def main():
    print("🔄 Loading vector store...")
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    db = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("🤖 Loading LLM...")
    llm = OllamaLLM(model=OLLAMA_MODEL)

    print("\n✅ RAG Assistant Ready!")
    print("Type your question or type 'exit' to quit.\n")

    while True:
        question = input("❓ Your question: ")

        if question.lower() in ["exit", "quit"]:
            break

        # 🔍 Retrieve documents
        docs = db.similarity_search(question, k=3)

        if not docs:
            print("\n💡 Answer:\nI don't know\n")
            continue

        context = "\n\n".join(doc.page_content for doc in docs)

        # DEBUG (shows RAG is working)
        print("\n🔍 Retrieved Context (preview):")
        print(context[:400], "\n")

        prompt_text = PROMPT.format(
            context=context,
            question=question
        )

        answer = llm.invoke(prompt_text)

        print(f"\n💡 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
