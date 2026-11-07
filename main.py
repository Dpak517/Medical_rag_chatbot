import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_NAME = "medical-chatbot-v2"
MODEL_NAME = "openai/gpt-oss-120b"

st.set_page_config(page_title="Medical Chatbot 🩺")
st.title("Medical Chatbot 🩺")
st.caption("Answers come from a medical encyclopedia. Not a substitute for a doctor.")

# ---- Your Pinecone key: from Streamlit Secrets (online) or .env (local) ----

try:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
except Exception:
    pass  # running locally: the key comes from the .env file instead
# ---- Sidebar: the user pastes their own Groq key ----
groq_key = st.sidebar.text_input("Groq API key", type="password")
st.sidebar.markdown("[Get a free Groq key](https://console.groq.com/keys)")

if st.sidebar.button("🗑️ New chat"):
    st.session_state.messages = []
    st.rerun()

if not groq_key:
    st.info("👈 Enter your free Groq API key in the sidebar to start.")
    st.stop()


# ---- Load the embedding model and Pinecone once ----
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 8})


retriever = load_retriever()

prompt = ChatPromptTemplate.from_template(
    """You are a careful medical information assistant based on the book "Where There Is No Doctor".

Rules:
1. Use ONLY the information in the Context below. Do not use outside knowledge.
2. Include ALL relevant points, steps and warnings from the Context that answer the question.
3. Do not add any fact, cause, treatment or dose that is not stated in the Context.
4. If the person describes symptoms in everyday words, match them to the closest condition in the Context. If several conditions fit, say so.
5. If the Context does not contain the answer, or the question is not about health, reply exactly: "I don't know based on the available information. I can only answer health questions covered in the book."
6. If the Context mentions danger signs or says to seek a health worker, include that advice.
7. Write in simple, clear language.

Context:
{context}

Question: {question}

Answer:""")

# The LLM uses the key the user typed
llm = ChatGroq(model=MODEL_NAME, api_key=groq_key, temperature=0)
chain = prompt | llm | StrOutputParser()

# ---- Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a medical question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer = chain.invoke({"context": context, "question": question})
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})