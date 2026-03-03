# 🤖📄 Chatbot Inteligente Baseado em PDFs (RAG) com Gemini

Projeto desenvolvido como desafio prático para aplicar conceitos de **IA Generativa, Embeddings e Busca Vetorial**, implementando um sistema de **RAG (Retrieval-Augmented Generation)** capaz de responder perguntas com base no conteúdo de documentos PDF.

---

## 🚀 Visão Geral

Este projeto permite:

- 📥 Upload de arquivos PDF
- ✂️ Divisão inteligente do texto em chunks
- 🧠 Geração de embeddings locais
- 🔎 Busca vetorial com FAISS
- 🤖 Geração de respostas contextualizadas utilizando **Google Gemini**

O sistema responde exclusivamente com base nos trechos recuperados dos documentos, evitando respostas inventadas (hallucinations).

---

## 🧠 Arquitetura do Projeto

Fluxo do pipeline:

### 🔹 Tecnologias Utilizadas

- Python
- Streamlit (interface interativa)
- FAISS (busca vetorial)
- SentenceTransformers (embeddings locais)
- Google Gemini API (LLM)
- LangChain (estruturação do pipeline)

---

## 📂 Estrutura do Projeto

> ⚠️ Arquivos como `.env`, `venv/`, `data/` e `__pycache__/` não são versionados.

---

## ⚙️ Como Executar o Projeto

### 1️⃣ Clone o repositório

```bash

git clone https://github.com/seu-usuario/chatbot-rag-pdfs-gemini.git
cd chatbot-rag-pdfs-gemini

```
### 2️⃣ Crie o ambiente virtual

python -m venv venv
venv\Scripts\activate  # Windows

``
### 3️⃣ Instale as dependências

pip install -r requirements.txt

``
### 4️⃣ Configure sua API Key do Gemini

GOOGLE_API_KEY=sua_chave_aqui

``
5️⃣ Execute o app

streamlit run app.py
