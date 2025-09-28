import json
import os
import requests
import google.generativeai as genai
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.llms.base import LLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from typing import List, Optional, Dict, Union
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from .llm_clients import get_gemini_client

class RagRetriever:
    """Handles RAG retrieval from a Chroma/Vector store and calls the LLM."""
    def __init__(self, embedder, collection, llm_client, pdf_dir: str = "./data/pdfs"):
        self.embedder = embedder
        self.collection = collection
        self.llm = llm_client
        self.pdf_dir = pdf_dir
        
    @staticmethod    
    def web_search(query: str, num_results: int = 5) -> str:
        """
        Perform a web search using Tavily API and return top results.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "num_results": num_results},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            results = resp.json()
            return "\n".join([f"{r['title']}: {r['url']}" for r in results.get("results", [])])
        except requests.RequestException as e:
            return f"Error during web search: {e}"

    def search_all_docs(self, query: str, k: int = 5) -> Dict:
        """
        Retrieve relevant chunks across ALL documents for a given query.
        Returns a structured dict with retrievals and synthesized LLM response.
        """
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas"]
        )

        if not results["documents"] or not results["documents"][0]:
            return {"answer": "No relevant results found.", "retrieved_chunks": []}

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            retrieved.append({
                "source": meta.get("source_filename", "unknown"),
                "pages": f"{meta.get('page_start', '?')}–{meta.get('page_end', '?')}",
                "content": doc
            })

        # Prepare messages for LLM
        rag_output = "\n".join(
            [f"Source: {r['source']}, Pages {r['pages']}\n{r['content']}" for r in retrieved]
        )
        messages = [
            SystemMessage(content="You are an expert in synthesizing RAG retrieval results."),
            HumanMessage(content=f"User Query: {query}\nRetrieved chunks:\n{rag_output}\n\n"
                                 f"Combine concisely, cite source pdf names.")
        ]

        response = self.llm.invoke(messages)

        return {
            "retrieved_chunks": retrieved,
            "answer": getattr(response, "content", str(response))
        }
    @staticmethod
    def parse_doc_and_query(input_str: str):
        """
        Parse input string that can be either:
        1. JSON string: '{"doc_name": "file.pdf", "query": ""}'
        2. Plain string: "doc_name='file.pdf', query=''"
        Returns: doc_name (str), query (str)
        """
        doc_name = ""
        query = ""

        input_str = input_str.strip()

        # Case 1: JSON string
        if input_str.startswith("{") and input_str.endswith("}"):
            try:
                data = json.loads(input_str)
                doc_name = data.get("doc_name", "")
                query = data.get("query", "")
                return doc_name, query
            except json.JSONDecodeError:
                # fallback to plain string parsing if JSON fails
                pass

        # Case 2: Plain string
        parts = input_str.split(",")
        for part in parts:
            part = part.strip()
            if part.startswith("doc_name="):
                doc_name = part[len("doc_name="):].strip().strip("'\"")
            elif part.startswith("query="):
                query = part[len("query="):].strip().strip("'\"")

        return doc_name, query

    def search_in_docs(self, doc_name: str, query: str = "", k: int = 5) -> Dict:
        """
        Retrieve relevant chunks from specific document. Query is optional.
        Returns a structured dict with retrievals and synthesized LLM response.
        """
        doc_name, query = RagRetriever.parse_doc_and_query(doc_name)
        print(f"\n\nDOC NAME:{doc_name}  TYPE: {type(doc_name)}")
        print(f"\n\nQUERY:{doc_name}  TYPE: {type(query)}")
        if query:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True).tolist()
        else:
            query_embedding = [[0.0] * self.embedder.get_sentence_embedding_dimension()]
        
        if isinstance(doc_name, str):
            doc_name = [doc_name]
        where_filter = {"source_filename": {"$in": doc_name}}
        
        # Handle filter for one vs many docs
        # where_filter = {"source_filename": {"$in": doc_name}} if isinstance(doc_name, list) else {
        #     "source_filename": doc_name
        # }

        results = self.collection.query(
            query_embeddings=query_embedding,
            where=where_filter,
            n_results=k,
            include=["documents", "metadatas"]
        )

        if not results["documents"] or not results["documents"][0]:
            return {"answer": f"No relevant results found in {doc_name}.", "retrieved_chunks": []}

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            retrieved.append({
                "source": meta.get("source_filename", "unknown"),
                "pages": f"{meta.get('page_start', '?')}–{meta.get('page_end', '?')}",
                "content": doc
            })

        rag_output = "\n".join(
            [f"Source: {r['source']}, Pages {r['pages']}\n{r['content']}" for r in retrieved]
        )
        messages = [
            SystemMessage(content="You are an expert in extracting insights from research papers."),
            HumanMessage(content=f"Query: {query}\nDocuments: {doc_name}\nRetrieved chunks:\n{rag_output}\n\n"
                                 f"Ignore references in chunks. Focus on actual relevant data. "
                                 f"Answer concisely and cite the correct source filename.")
        ]

        response = self.llm.invoke(messages)

        return {
            "retrieved_chunks": retrieved,
            "answer": getattr(response, "content", str(response))
        }

    def list_documents(self, *args, **kwargs) -> List[str]:
        """Return list of available PDFs."""
        return os.listdir(self.pdf_dir)


class RagToolsFactory:
    """Wraps RagRetriever methods into LangChain Tool objects."""

    @staticmethod
    def create_tools(retriever: RagRetriever) -> List[Tool]:
        return [
            Tool(
                name="SearchAllDocs",
                func=retriever.search_all_docs,
                description="Search across ALL documents for a query. Input: query string. and k value (default is 5)"
            ),
            Tool(
                name="SearchInDocs",
                func=retriever.search_in_docs,
                description="Search within a specific document"
                            "Input: doc_name and k value(default is 5)"
            ),
            Tool(
                name="ListDocuments",
                func=retriever.list_documents,
                description="List all available documents in the system. No input required."
            ), 
            Tool(
                name="WebSearch",
                func=retriever.web_search,
                description="Useful for searching the web for relevant citations. Input: query (str), optional num_results (int, default=5)."
                ),
            
        ]

class EnhanceResponseTools:
    @staticmethod
    def web_search(query: str, num_results: int = 5) -> str:
        """
        Perform a web search using Tavily API and return top results.
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables.")

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "num_results": num_results},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            results = resp.json()
            return "\n".join([f"{r['title']}: {r['url']}" for r in results.get("results", [])])
        except requests.RequestException as e:
            return f"Error during web search: {e}"

    @staticmethod
    def explain_term(term: str) -> str:
        """
        Explain a given term using Gemini AI client.
        """
        prompt = (
            f"You are an expert in explaining AI and Astronomy related terms. "
            f"Explain the following term in simple words:\n\n{term}"
        )
        try:
            response = get_gemini_client().invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error while explaining term: {e}"

    @staticmethod
    def write_code(description: str) -> str:
        """
        Generate code for a given requirement using Gemini AI client and explain it briefly.
        """
        prompt = f"Write code for the following requirement and explain what it does briefly:\n\n{description}"
        try:
            response = get_gemini_client().invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error while writing code: {e}"

    @classmethod
    def create_tools(cls) -> List[Tool]:
        """
        Create a list of tools for document retrieval.
        """
        return [ 
                Tool(
                    name="WebSearch",
                    func=cls.web_search,
                    description="Useful for searching the web for relevant citations. Input: query (str), optional num_results (int, default=5)."
                    ),
                # Tool(
                #     name="ExplainTerm",
                #     func=cls.explain_term,
                #     description="Useful for explaining technical terms in simple words. Input: term (str)."
                # ),
                # Tool(
                #     name="CodeWriter",
                #     func=cls.write_code,
                #     description="Useful for generating code in a specified language. Input: description (str)."
                # ),
            ]