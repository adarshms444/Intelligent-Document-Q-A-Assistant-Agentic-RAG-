!pip install chromadb langchain-community langchain-text-splitters langchain-huggingface

import os
import json
import requests
import gradio as gr
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import requests
OPENROUTER_API_KEY = ""

LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

def call_llm(prompt, system_prompt="You are a helpful AI assistant."):
    """Helper function to call the OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error from LLM API: {response.text}"

"""Cell 3: Knowledge Base & Document Ingestion"""

global_full_text = ""
chroma_client = chromadb.Client()
collection_name = "assignment_collection"

try:
    chroma_client.delete_collection(name=collection_name)
except:
    pass
collection = chroma_client.create_collection(name=collection_name)

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_pdf(file_path):
    global global_full_text

    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Save full text for the summarize tool
    global_full_text = "\n".join([doc.page_content for doc in documents])

    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in chunks]

    metadatas = [{"page": str(chunk.metadata.get('page', 'Unknown'))} for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Creating embeddings and storing in ChromaDB...")
    embedded_texts = embeddings_model.embed_documents(texts)
    collection.add(
        embeddings=embedded_texts,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Success! Processed {len(documents)} pages into {len(chunks)} searchable chunks.")

def summarize_tool(query_ignored):
    print(">> Executing Summarize Tool...")
    truncated_text = global_full_text[:15000]
    prompt = f"Provide a comprehensive, well-structured executive summary of the following document:\n\n{truncated_text}"
    system = "You are an expert summarizer. Abstract the main points clearly."
    return call_llm(prompt, system)

def qa_tool(query):
    print(">> Executing General Q&A Tool...")
    query_embedding = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    context = ""
    citations = []

    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        page_num = int(results['metadatas'][0][i]['page']) + 1
        context += f"--- Chunk {i+1} (Page {page_num}) ---\n{doc_text}\n\n"
        citations.append(f"Page {page_num}")

    unique_citations = list(set(citations))

    prompt = f"Answer the user's question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
    system = "You are a precise Q&A assistant. If the answer is not in the context, say so."

    answer = call_llm(prompt, system)

    return f"{answer}\n\n**Sources Consulted:** {', '.join(unique_citations)}"

def action_item_tool(query):
    print(">> Executing Action Item Extractor Tool...")
    query_embedding = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    context = "\n".join(results['documents'][0])
    prompt = f"Review the following context and extract any actionable tasks, future research directions, or recommended next steps. Format them as a clear bulleted list.\n\nContext:\n{context}\n\nUser Request: {query}"
    system = "You are an Action Item Extractor. Identify concrete tasks, goals, or required actions from the text."

    return call_llm(prompt, system)

def agent_dispatcher(query):
    if not global_full_text:
        return "System Error", "Please ingest a PDF document first."

    prompt = f"""Analyze the following user query and classify their intent.
    Return ONLY ONE of the following exact words:
    - SUMMARIZE : If they ask for an overview, summary, or abstract of the whole document.
    - ACTION : If they explicitly ask for tasks, action items, next steps, or future directions.
    - QA : If they are asking a specific question or looking for details in the document.

    Query: "{query}"
    """

    intent_response = call_llm(prompt, system_prompt="You are a strict routing agent. Output only one word.").strip().upper()

    if "SUMMARIZE" in intent_response:
        active_tool = "Summarize Tool"
        response = summarize_tool(query)
    elif "ACTION" in intent_response:
        active_tool = "Action Item Extractor"
        response = action_item_tool(query)
    else:
        active_tool = "General Q&A Tool"
        response = qa_tool(query)

    return active_tool, response

!pip install PyPDF

pdf_path = "/content/LLM.pdf"

# Ingest the document
ingest_pdf(pdf_path)

print("\n" + "="*50)

user_query = "Give the summary of this pdf"

# 3. Run the dispatcher
tool_used, final_answer = agent_dispatcher(user_query)

print(f"USER QUERY: {user_query}")
print(f"ROUTED TO TOOL: [{tool_used}]")
print("="*50 + "\n")
print(final_answer)

pdf_path = "/content/LLM.pdf"

ingest_pdf(pdf_path)

print("\n" + "="*50)


user_query = "What is LLM and What are pretrained models?"


# 3. Run the dispatcher
tool_used, final_answer = agent_dispatcher(user_query)

print(f"USER QUERY: {user_query}")
print(f"ROUTED TO TOOL: [{tool_used}]")
print("="*50 + "\n")
print(final_answer)

pdf_path = "/content/LLM.pdf"

# Ingest the document
ingest_pdf(pdf_path)

print("\n" + "="*50)

user_query = "can you explain me about BERT"

tool_used, final_answer = agent_dispatcher(user_query)

print(f"USER QUERY: {user_query}")
print(f"ROUTED TO TOOL: [{tool_used}]")
print("="*50 + "\n")
print(final_answer)


import gradio as gr

def ui_process_pdf(file_obj):
    if file_obj is None:
        return "⚠️ Please upload a PDF file first."
    try:
        ingest_pdf(file_obj.name)
        return "✅ Knowledge Base Initialized Successfully. Previous data cleared."
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def ui_chat(query, history):
    if history is None:
        history = []

    if not global_full_text:
        error_msg = """<div class="error-badge">⚠️ System Error</div>

Please upload and ingest a document into the Knowledge Base before submitting queries."""
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": error_msg})

        return history, "⚠️ SYSTEM ERROR: NO DOCUMENT"

    tool_used, answer = agent_dispatcher(query)
    formatted_answer = f"""<div class="tool-badge">⚙️ Agent Routed to: {tool_used.upper()}</div>\n\n{answer}"""

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": formatted_answer})

    status_box_update = f"🟢 ROUTED TO: {tool_used.upper()}"

    return history, status_box_update


# Professional Blue CSS & Theming

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f4f6f8 !important;
}
.header-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 2.5rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
    margin-bottom: 25px;
    border: 1px solid #bfdbfe;
}
.header-banner h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    letter-spacing: -0.025em !important;
    margin: 0 !important;
}
.header-banner p {
    color: #e0f2fe !important;
    font-size: 1.1rem !important;
    margin-top: 10px !important;
}
#chat-container {
    background: #ffffff !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
    border: 1px solid #e2e8f0 !important;
}
.tool-status input {
    color: #0369a1 !important;
    background-color: #f0f9ff !important;
    font-weight: 700 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 8px !important;
    text-align: center !important;
    letter-spacing: 0.05em !important;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
}
.tool-badge {
    display: inline-block;
    background-color: #eff6ff;
    color: #1e40af;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
    border: 1px solid #bfdbfe;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.error-badge {
    display: inline-block;
    background-color: #fef2f2;
    color: #991b1b;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 16px;
    border: 1px solid #fecaca;
}
"""

professional_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)


# 3. Gradio Interface Layout
with gr.Blocks(theme=professional_theme, css=custom_css) as demo:

    gr.HTML(
        """
        <div class="header-banner">
            <h1>Intelligent Agentic RAG</h1>
            <p>Enterprise Document Architecture with Automated Intent Routing</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📂 1. Knowledge Base Initialization")
            gr.Markdown("Upload a document to generate vector embeddings and populate the local ChromaDB instance.")

            pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            process_btn = gr.Button("⚙️ Generate Embeddings", variant="primary")
            upload_status = gr.Textbox(label="System Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                """
                ### 🛠️ Active Agent Capabilities
                The Dispatcher uses an LLM to classify intent and route to:
                - 📋 **Summarizer Engine**
                - 🔍 **Vector Q&A (with Citations)**
                - ✅ **Action Item Extractor**
                """
            )

        with gr.Column(scale=3):
            active_tool_display = gr.Textbox(
                label="Agent Dispatcher Activity",
                value="WAITING FOR QUERY...",
                interactive=False,
                elem_classes=["tool-status"]
            )

            chatbot = gr.Chatbot(
                label="Agent Workspace",
                elem_id="chat-container",
                height=500,
                type="messages",
                avatar_images=("👤", "🧠"),
                show_copy_button=True,
                render_markdown=True
            )

            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Enter a prompt, question, or task for the Agent...",
                    scale=5,
                    container=False
                )
                submit_btn = gr.Button("Submit Request", variant="primary", scale=1)

            clear_btn = gr.ClearButton([user_input, chatbot, active_tool_display], value="🗑️ Clear Workspace")


    # Event Listeners

    process_btn.click(
        fn=ui_process_pdf,
        inputs=[pdf_input],
        outputs=[upload_status]
    )

    user_input.submit(
        fn=ui_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, active_tool_display]
    ).then(lambda: "", None, user_input)

    submit_btn.click(
        fn=ui_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, active_tool_display]
    ).then(lambda: "", None, user_input)

demo.launch(debug=True, share=True)
