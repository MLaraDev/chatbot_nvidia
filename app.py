import os
import streamlit as st
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="NVIDIA RAG PDF Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Assistente Virtual de Manuais em PDF")
st.write("Qual sua dúvida? Como posso ajudar?")

# =====================================================
# CHAVE DA API NVIDIA
# =====================================================

nvidia_api_key = os.getenv("NVIDIA_API_KEY")

try:
    if "NVIDIA_API_KEY" in st.secrets:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass

if not nvidia_api_key:
    st.error(
        "Por favor, adicione sua NVIDIA_API_KEY ao arquivo .env ou ao Streamlit Secrets."
    )
    st.stop()

# =====================================================
# FUNÇÃO AUXILIAR
# =====================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =====================================================
# CARREGAMENTO E VETORIZAÇÃO DO PDF
# =====================================================

@st.cache_resource(show_spinner="Processando o PDF...")
def inicializar_rag():

    nome_arquivo = "manual.pdf"

    if not os.path.exists(nome_arquivo):
        st.error(f"Arquivo '{nome_arquivo}' não foi encontrado.")
        st.stop()

    # Carrega PDF
    loader = PyPDFLoader(nome_arquivo)
    paginas = loader.load()

    # Divide em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(paginas)

    # Embeddings NVIDIA
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )

    # Vetoriza documentos
    vectorstore = FAISS.from_documents(
        docs,
        embedding=embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

# =====================================================
# RETRIEVER
# =====================================================

retriever = inicializar_rag()

# =====================================================
# LLM NVIDIA
# =====================================================

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    nvidia_api_key=nvidia_api_key,
    temperature=0.2
)

# =====================================================
# PROMPT
# =====================================================

template_prompt = """
Você é um assistente técnico especializado e prestativo.

Os textos fragmentados de contexto inseridos abaixo foram extraídos de um manual de produtos ou sistemas e podem estar em inglês.

Sua tarefa é analisar o contexto, mesmo que esteja em inglês, mas responder SEMPRE em Português do Brasil.

Use estritamente as informações fornecidas no contexto para responder.

Se a resposta não puder ser encontrada no texto, diga exatamente:

"Desculpe, mas essa informação não consta no manual."

Contexto:
{context}

Pergunta:
{question}

Resposta em português:
"""

prompt = ChatPromptTemplate.from_template(template_prompt)

# =====================================================
# PIPELINE RAG
# =====================================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# HISTÓRICO DO CHAT
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Processsei o manual com sucesso. "
                "Faça qualquer pergunta sobre ele."
            )
        }
    ]

# Exibe histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================================================
# ENTRADA DO USUÁRIO
# =====================================================

if prompt_usuario := st.chat_input(
    "Ex: Qual o significado do código 4?"
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt_usuario
        }
    )

    with st.chat_message("user"):
        st.write(prompt_usuario)

    with st.chat_message("assistant"):

        with st.spinner("Consultando o manual técnico..."):

            try:

                resposta = rag_chain.invoke(prompt_usuario)

                st.write(resposta)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": resposta
                    }
                )

            except Exception as e:

                st.error(
                    f"Erro ao processar a requisição da API: {e}"
                )