import os
import time
import tempfile
from dotenv import load_dotenv

import streamlit as st

from pinecone import Pinecone, ServerlessSpec

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI

############################################################
# Load Environment Variables
############################################################

load_dotenv()


def get_env_setting(name, default=""):
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


OPENROUTER_API_KEY = get_env_setting("OPENROUTER_API_KEY") or get_env_setting("OPENAI_API_KEY")
PINECONE_API_KEY = get_env_setting("PINECONE_API_KEY")

INDEX_NAME = get_env_setting("INDEX_NAME")

EMBED_MODEL = get_env_setting("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

MODEL_NAME = get_env_setting("MODEL_NAME", "openai/gpt-4o-mini")

############################################################
# Streamlit Config
############################################################

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

st.caption("OpenRouter + Pinecone + LangChain + Streamlit")

############################################################
# Sidebar
############################################################

with st.sidebar:

    st.header("Settings")

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2
    )

    top_k = st.slider(
        "Top K Documents",
        1,
        10,
        4
    )

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Process PDFs",
        use_container_width=True
    )

    clear_chat = st.button(
        "Clear Chat",
        use_container_width=True
    )

############################################################
# Session State
############################################################

if "messages" not in st.session_state:
    st.session_state.messages = []

if clear_chat:
    st.session_state.messages = []

############################################################
# Validate API Keys
############################################################

if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

if not PINECONE_API_KEY:
    st.error("Missing PINECONE_API_KEY")
    st.stop()

############################################################
# Initialize Embedding Model
############################################################

@st.cache_resource
def load_embedding():

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

embedding_model = load_embedding()

############################################################
# Initialize Pinecone
############################################################

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

############################################################
# Create Pinecone Index
############################################################

existing_indexes = [index["name"] for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

############################################################
# Create Vector Store
############################################################

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)

############################################################
# Text Splitter
############################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ".",
        " ",
        ""
    ]
)

############################################################
# Process Uploaded PDFs
############################################################

if process_button:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        with st.spinner("Processing PDFs..."):

            documents = []

            start = time.time()

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(uploaded_file.read())

                    pdf_path = tmp.name

                loader = PyPDFLoader(pdf_path)

                pdf_docs = loader.load()

                documents.extend(pdf_docs)

                os.remove(pdf_path)

            chunks = text_splitter.split_documents(documents)

            for i, chunk in enumerate(chunks):

                chunk.metadata["chunk_id"] = i

            vectorstore.add_documents(chunks)

            elapsed = round(time.time() - start, 2)

        st.success(
            f"✅ {len(chunks)} chunks uploaded in {elapsed} sec."
        )

############################################################
# Retriever
############################################################

retriever = vectorstore.as_retriever(

    search_type="similarity",

    search_kwargs={
        "k": top_k
    }

)

############################################################
# Prompt Template
############################################################

prompt = ChatPromptTemplate.from_template(
"""
You are an expert AI assistant.

Answer ONLY from the provided context.

If the answer is unavailable,
say:

'I don't know based on the provided documents.'

Context:
{context}

Question:
{question}

Requirements:

• Be concise.

• Use bullet points if needed.

• Mention page numbers when available.

Answer:
"""
)

############################################################
# Initialize OpenRouter LLM
############################################################

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    model=MODEL_NAME,
    temperature=temperature,
    base_url="https://openrouter.ai/api/v1",
)

############################################################
# Helper Function
############################################################

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

############################################################
# RAG Chain (LCEL)
############################################################

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

############################################################
# Display Previous Chat
############################################################

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

############################################################
# Chat Input
############################################################

question = st.chat_input("Ask anything from your PDFs...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            start = time.time()

            docs = retriever.invoke(question)

            try:
                answer = rag_chain.invoke(question)
            except Exception as e:
                error_message = str(e).lower()
                if "401" in str(e) or "invalid_api_key" in error_message or "api key" in error_message:
                    st.error("The OpenRouter API key is invalid or expired. Please update OPENROUTER_API_KEY in your .env file or environment variables and restart the app.")
                elif "404" in str(e) or "not_found" in error_message or "not supported" in error_message:
                    st.error(f"The OpenRouter model '{MODEL_NAME}' is not available for your account. Please set MODEL_NAME in your .env file to a supported model such as 'openai/gpt-4o-mini' or 'google/gemini-2.0-flash-001'.")
                else:
                    st.error("The OpenRouter model request failed.")
                    st.exception(e)
                st.stop()

            elapsed = round(time.time() - start, 2)

            st.markdown(answer)

            st.caption(f"⏱ Response Time: {elapsed} sec")

            if docs:

                with st.expander("📄 Source Chunks"):

                    for i, doc in enumerate(docs, start=1):

                        page = doc.metadata.get("page", "Unknown")

                        source = doc.metadata.get(
                            "source",
                            "Unknown File"
                        )

                        st.markdown(
                            f"### Source {i}"
                        )

                        st.write(
                            f"**File:** {os.path.basename(source)}"
                        )

                        st.write(
                            f"**Page:** {page + 1 if isinstance(page, int) else page}"
                        )

                        st.write(
                            doc.page_content[:800]
                        )

                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )