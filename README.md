# Intelligent Agentic RAG Assistant

> An enterprise-grade Document Intelligence system that utilizes Agentic routing to dynamically classify user intent and execute specialized AI tools.

## 🎥 Project Demonstration

[Insert Video Here] 
*(Note: Replace this line with `![Demo Video](./path_to_your_video/video.mp4)` or a YouTube link once you record your screen).*

---

## 🛑 The Problem Statement
Standard Retrieval-Augmented Generation (RAG) systems suffer from a critical flaw: they treat every user query as a semantic search problem. If a user asks for a high-level summary or a list of tasks, a standard RAG system will blindly retrieve a few random chunks of text and fail to answer the prompt correctly. 

The goal of this project is to build an **Intelligent Document Q&A Assistant** that acts as a traffic controller. It must ingest a large document, establish a local knowledge base, and use a central Agent logic component to "read the user's mind" (Intent Classification) before routing the query to the correct specialized tool.

---

## 🏗️ My Approach & Architecture
To solve this, I designed a modular, multi-tool Agentic architecture powered by an open-source LLM (Nemotron-3 via OpenRouter) and a local Vector Database. 

### 1. Document Ingestion & Knowledge Base
* **Processing:** The system accepts multi-page PDF uploads, extracting the raw text and intelligently cleaning it.
* **Chunking:** The text is split into overlapping chunks (800 characters) to preserve context boundaries.
* **Embedding:** Utilized `all-MiniLM-L6-v2` via HuggingFace to convert chunks into dense vector representations.
* **Storage:** Embedded data is stored in a local **ChromaDB** instance, which automatically refreshes its memory pool upon new document uploads to prevent cross-contamination of data.

### 2. The Agent Dispatcher (Intent Classification)
Instead of immediately querying the database, user inputs are first sent to the Agent Dispatcher. The LLM acts as a strict router, analyzing the prompt and classifying the intent into one of three distinct paths.

### 3. Specialized Execution Tools
Based on the Agent's classification, the system triggers one of three distinct tools:
* 📋 **The Summarizer Engine:** Bypasses the Vector DB entirely. It ingests the document text directly to generate a comprehensive, high-level executive abstract.
* 🔍 **The General Q&A Engine:** Executes a standard RAG pipeline. It converts the query into an embedding, retrieves the top 4 most semantically similar chunks from ChromaDB, and generates a highly specific answer based *only* on the retrieved context.
* ✅ **The Action Item Extractor:** A custom business-logic tool. It scans the retrieved context specifically for tasks, future research directions, and required next steps, formatting them into a clear deliverable list.

### 4. Enterprise User Interface
The backend logic is connected to a custom **Gradio** web interface. It features a modern, professional CSS theme, a dedicated document ingestion panel, and an interactive chat workspace. 

---

## ⭐ Advanced Features Achieved
This project successfully fulfills the core requirements and implements the following advanced features:
1. **Accurate Context Citations:** The Q&A tool extracts metadata during retrieval and dynamically appends the exact source page numbers to the bottom of the AI's response.
2. **Real-Time Tool Visibility:** The UI features a dynamic "Agent Dispatcher Activity" badge that visually updates to show the user exactly which routing decision the Agent made.
3. **Custom Specific Task Tool:** Implemented the Action Item Extractor as a third, highly practical capability beyond standard summaries and Q&A.

---

*Developed by Adarsh*
