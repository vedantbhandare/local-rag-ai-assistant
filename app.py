import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate


# -----------------------------
# Configuration
# -----------------------------
VECTOR_STORE_DIR = "vector_store"
OLLAMA_MODEL = "mistral:7b-instruct-q4_K_M"


# -----------------------------
# Prompt
# -----------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.

Answer the question using ONLY the information from the context below.
You may summarize or infer, but DO NOT add outside information.

If the context does not contain the answer, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)


# -----------------------------
# Load resources (cached)
# -----------------------------
@st.cache_resource
def load_vector_store():
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    return FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    return OllamaLLM(model=OLLAMA_MODEL)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Local RAG Assistant", layout="wide")

st.title("🧠 Local RAG AI Assistant")
st.caption("Runs 100% locally using Ollama + FAISS")

db = load_vector_store()
llm = load_llm()

question = st.text_input("❓ Ask a question based on your documents:")

if question:
    with st.spinner("Thinking..."):
        # 1️⃣ Retrieve documents
        docs = db.similarity_search(question, k=3)

        if not docs:
            st.warning("I don't know. No relevant context found.")
        else:
            # 2️⃣ Build context
            context = "\n\n".join(doc.page_content for doc in docs)

            # 3️⃣ Build prompt
            prompt_text = PROMPT.format(
                context=context,
                question=question
            )

            # 4️⃣ Get answer
            answer = llm.invoke(prompt_text)

            # -----------------------------
            # Display
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💡 Answer")
                st.write(answer)

            with col2:
                st.subheader("📄 Retrieved Context")
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(doc.page_content[:800])
