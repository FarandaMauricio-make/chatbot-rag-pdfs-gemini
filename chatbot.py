import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# 1) CONFIGURAÇÃO (eu deixo as chaves fora do código)
# ============================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_PATH = "data/faiss_index.pkl"

# Modelo Gemini (bom e barato): "gemini-1.5-flash"
# Se quiser mais “caprichado”: "gemini-1.5-pro"
GEMINI_MODEL = "models/gemini-2.5-flash"

# ============================================================
# 2) FUNÇÕES: PDF -> TEXTO -> CHUNKS
# ============================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Eu extraio texto do PDF com pypdf.
    Observação: se o PDF for escaneado (imagem), pode vir vazio.
    """
    try:
        reader = PdfReader(uploaded_file)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def chunk_text(text: str, chunk_size=1000, chunk_overlap=150) -> list[str]:
    """
    Eu quebro o texto em pedaços menores (chunks) pra melhorar a busca vetorial.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ============================================================
# 3) FUNÇÕES: FAISS (índice vetorial) + embeddings locais (offline)
# ============================================================
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def save_vectorstore(vectordb: FAISS):
    """
    Eu salvo o índice FAISS em disco pra não reindexar toda vez.
    """
    ensure_data_dir()
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(vectordb, f)


def load_vectorstore() -> FAISS | None:
    """
    Eu carrego o índice FAISS salvo (se existir).
    """
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return None


def build_vectorstore(texts: list[str], metadatas: list[dict]) -> FAISS:
    """
    Eu gero embeddings localmente com um modelo leve e bom (MiniLM),
    e crio o índice FAISS com os chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vectordb


def retrieve_docs(vectordb: FAISS, question: str, k: int = 4):
    """
    Eu busco os top-k trechos mais relevantes (similaridade vetorial).
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(question)


def format_context(docs) -> str:
    """
    Eu monto o contexto com fonte + trecho.
    Isso ajuda o Gemini a responder 'grounded' (sem inventar).
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "desconhecido")
        parts.append(f"[Trecho {i} | Fonte: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ============================================================
# 4) FUNÇÃO: responder com Gemini usando o contexto recuperado
# ============================================================
def gemini_answer(question: str, context: str) -> str:
    """
    Eu passo para o Gemini:
    - instrução pra usar apenas o contexto
    - contexto recuperado do FAISS
    - pergunta do usuário
    """
    system_rules = (
        "Você é um assistente que responde SOMENTE com base no CONTEXTO fornecido.\n"
        "Se a resposta não estiver no contexto, responda exatamente: "
        "\"Não encontrei essa informação nos documentos enviados.\" \n"
        "Responda em português e cite a Fonte (nome do PDF) quando usar trechos."
    )

    prompt = (
        f"{system_rules}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA:\n{question}\n\n"
        f"RESPOSTA:"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# ============================================================
# 5) UI Streamlit
# ============================================================
st.set_page_config(page_title="Chatbot de PDFs (Gemini + RAG)", page_icon="📄", layout="wide")
st.title("📄🤖 Chatbot baseado em PDFs (RAG) — Gemini + FAISS")

if not GOOGLE_API_KEY:
    st.error("Não encontrei GOOGLE_API_KEY no .env. Crie um .env com sua chave do Gemini.")
    st.stop()

# Configura Gemini
genai.configure(api_key=GOOGLE_API_KEY)

with st.sidebar:
    st.header("📥 Upload e Indexação")

    uploaded_files = st.file_uploader(
        "Envie 1 ou mais PDFs (com texto selecionável)",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:
        do_index = st.button("🧠 Indexar PDFs")

    with col2:
        clear_index = st.button("🗑️ Apagar índice")

    st.divider()
    st.caption("O índice fica salvo em /data/faiss_index.pkl para reaproveitar nas próximas execuções.")

if clear_index:
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        st.success("Índice apagado. Indexe novamente.")
    else:
        st.info("Não existe índice salvo ainda.")

# Carrego índice (se existir)
vectordb = load_vectorstore()

# Indexação
if do_index:
    if not uploaded_files:
        st.warning("Envie pelo menos 1 PDF antes de indexar.")
    else:
        all_texts = []
        all_metadatas = []

        for f in uploaded_files:
            text = extract_text_from_pdf(f)

            if not text:
                st.warning(f"Não consegui extrair texto do PDF: {f.name} (talvez seja escaneado).")
                continue

            chunks = chunk_text(text)
            all_texts.extend(chunks)
            all_metadatas.extend([{"source": f.name} for _ in chunks])

        if not all_texts:
            st.error("Nenhum texto foi extraído. Use PDFs que tenham texto selecionável (não escaneado).")
        else:
            vectordb = build_vectorstore(all_texts, all_metadatas)
            save_vectorstore(vectordb)
            st.success(f"✅ Indexação concluída! Total de chunks indexados: {len(all_texts)}")

st.divider()
st.subheader("💬 Chat")

if vectordb is None:
    st.info("Ainda não há PDFs indexados. Faça upload e clique em **Indexar PDFs** na barra lateral.")
    st.stop()

# Histórico de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Pergunte algo sobre os PDFs indexados...")

if question:
    # Mostro pergunta
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Busco contexto e respondo
    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos relevantes e gerando resposta com Gemini..."):
            docs = retrieve_docs(vectordb, question, k=4)
            context = format_context(docs)
            answer = gemini_answer(question, context)

        st.markdown(answer)

        # Transparência: mostro trechos usados
        with st.expander("📌 Ver trechos usados como base (grounding)"):
            for i, d in enumerate(docs, start=1):
                source = d.metadata.get("source", "desconhecido")
                st.markdown(f"**Trecho {i} — Fonte: {source}**")
                st.write(d.page_content)

    st.session_state["messages"].append({"role": "assistant", "content": answer})