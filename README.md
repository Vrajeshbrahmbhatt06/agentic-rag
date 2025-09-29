# RAG Agent

[**Check out live demo**](https://agentic-rag-dbo79amkexnu6ejhfc476j.streamlit.app/)

A Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **Gemini LLM**, **ChromaDB**, and document ingestion utilities.
This project lets you chat with research papers, preview/manage PDFs, and retrieve documents from external sources like **ArXiv**.

---

## 📂 Project Structure

### 📄 Streamlit Pages (`pages/`)

This folder contains the **Streamlit app pages** that form the user interface.

* **`chat.py`**

  * Implements the **chat interface** for interacting with the RAG agent.
  * Uses `ConversationBufferMemory` to persist memory across the ongoing session.
  * Returns both the **user query** and the final answer along with **agent logs** for better traceability.

* **`pdf_viewer.py`**

  * Provides a page to **preview available documents** within the system.
  * Allows users to **download documents** directly from the UI.

---

### 📥 Extractors (`extractors/`)

This folder contains utilities for fetching external research documents.

* **`arxiv_extractor.py`**

  * Provides helper functions to download research papers directly from **ArXiv**.
  * Parameters:

    * **`path`** → Directory where papers will be stored.
    * **`subject`** → ArXiv subject area (defaults to **Astronomy** or **AI**).
    * **`num`** → Number of documents to retrieve.
  * Designed to make bulk collection of research papers simple and reproducible.

📌 Currently, all downloaded documents are stored in **`data/pdfs/`**.

---

### Agent Utilities (`agent_utils/`)

This folder contains helper modules for creating and managing **RAG agents** and their supporting tools.

* **`agents.py`**

  * Imports LLM clients (currently **Gemini**) and RAG tools.
  * Provides the `create_rag_agent` function to initialize and configure a RAG agent with the required retriever and tools.

* **`tools.py`**

  * **`RagRetriever`**: Implements core retrieval tools for the RAG agent, including:

    * `web_search` (via **Tavily**)
    * `search_all_docs`
    * `parse_doc_and_query` (internal helper)
    * `search_in_docs`
    * `list_documents`
  * **`RAGToolFactory`**: Wraps retriever objects into tools with names and descriptions, making them ready for agent use.
  * **`EnhanceResponseTools`**: Consisting tools designed to enhance response agent (not active yet).

* **`prompts_collection.py`**

  * Stores reusable **system messages** and prompt templates for different agent tasks.

* **`llm_clients.py`**

  * Returns an **LLM client instance**.
  * Currently supports **Gemini** (default: `gemini-2.5-flash`).

---

### 📑 Chunking Strategy (`vectorstore/utils.py`)

To prepare PDFs for embedding and retrieval, we split documents into **semantic chunks** using sentence boundaries.

* **Target size:** Each chunk is ~**500 words**.
* **Overlap:** The last ~**150 words** of each chunk are carried into the next one to preserve context.
* **Unit of split:** Sentences (via `nltk.sent_tokenize`) rather than raw text cuts, ensuring chunks remain readable and meaningful.
* **Metadata stored:**

  * Document ID (hash of file)
  * Source filename
  * Page range (`page_start`, `page_end`)
  * Chunk index & word length
  * Section heading (heuristic detection)
  * SHA256 hash of chunk (for deduplication)

This ensures chunks are **context-aware**, **deduplicated**, and **optimized for embeddings** when stored in ChromaDB.
