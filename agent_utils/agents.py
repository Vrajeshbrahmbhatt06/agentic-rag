from .tools import RagRetriever, RagToolsFactory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from sentence_transformers import SentenceTransformer
import chromadb
from .llm_clients import get_gemini_client

def create_rag_agent():
    retriever = RagRetriever(
        embedder=SentenceTransformer("all-mpnet-base-v2"),
        collection=chromadb.PersistentClient(path="chroma_db").get_collection("papers_chunks"),
        llm_client=get_gemini_client()
    )

    rag_agent_tools = RagToolsFactory.create_tools(retriever)

    rag_agent = initialize_agent(
        tools=rag_agent_tools,
        llm=get_gemini_client(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True 

    )

    return rag_agent
