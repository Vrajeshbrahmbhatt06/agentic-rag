import streamlit as st
from agent_utils import create_rag_agent, prompts
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from io import StringIO # Not strictly needed anymore, but kept for clarity on original intent
import time
from contextlib import redirect_stdout # Not strictly needed anymore, but kept for clarity on original intent

# --- UI Configuration ---
# Set the page title and layout for better chat experience
st.set_page_config(page_title="RAG Chat UI", layout="wide")
st.title("Enhanced RAG Chatbot")

# --- Agent & Memory Initialization ---
# Build agent once per session
if "rag_agent" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.rag_agent = create_rag_agent()

# Ensure all new state variables are initialized even if 'rag_agent' existed from a prior session
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

if "current_query" not in st.session_state:
    st.session_state.current_query = None

if "trace_history" not in st.session_state:
    # Initialize a list to store the console trace (now the full streamed trace) for each AI response
    st.session_state.trace_history = [] 





# --- Clear Chat History Button ---
def clear_chat_history():
    """Clears the session memory and forces a rerun."""
    st.session_state.memory.clear()
    st.session_state.is_loading = False
    st.session_state.current_query = None
    st.session_state.trace_history = [] # Clear trace history too
    # Using st.rerun() ensures the chat history is immediately cleared on the UI
    st.rerun()

if st.button("🗑️ Clear Chat History", type="secondary"):
    clear_chat_history()


# --- Display Conversation History (All messages) ---
# Messages are displayed in chronological order (oldest at top, newest at bottom).
trace_index = 0
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            
        # Display the associated trace immediately after the AI message
        if trace_index < len(st.session_state.trace_history) and st.session_state.trace_history[trace_index]:
            with st.expander(f"See Agent Trace for Response #{trace_index + 1}"):
                # Display the raw captured text in a code block for clean formatting
                st.code(st.session_state.trace_history[trace_index], language='text')
            
        trace_index += 1


# --- STICKY Input Box and Query Handler ---
# st.chat_input is sticky at the bottom of the screen and automatically clears on submit.
user_query = st.chat_input(
    "Ask something about your documents...",
    key="chat_input",
    # Disable the input box while the AI is processing
    disabled=st.session_state.is_loading
)

# 1. CAPTURE QUERY AND INITIATE LOADING STATE
if user_query and not st.session_state.is_loading:
    # Store the query and set loading flag
    st.session_state.current_query = user_query
    st.session_state.is_loading = True
    # Immediately rerun to disable the input box
    st.rerun()




# 2. EXECUTE AGENT TASK (This runs in the second execution cycle, where input is disabled)
if st.session_state.is_loading and st.session_state.current_query:
    query_to_process = st.session_state.current_query
    
    # 1. Display User Message (right aligned)
    with st.chat_message("user"):
        st.write(query_to_process)

    # 2. Add user message to memory (required for history context)
    st.session_state.memory.chat_memory.add_message(HumanMessage(content=query_to_process))

    history_messages = st.session_state.memory.chat_memory.messages
    
    # Separate history and current input
    # chat_history: all messages *before* the current query
    raw_chat_history = history_messages[:-1] 
    # input: the text of the current query
    current_input_text = history_messages[-1].content
    
    # Inject the system message to the start of the chat_history for the agent payload
    # This overrides the default agent system prompt but ensures the instructions are passed.
    payload_chat_history = [SystemMessage(content=prompts["RAG_AGENT_SYS_MSG"])] + raw_chat_history

    payload = {
        "chat_history": payload_chat_history,
        "input": current_input_text
    }
    ai_message_container = st.chat_message("assistant")
    status_container = ai_message_container.empty()
    def update_status(message: str):
        status_container.markdown(
            f"""
            <div class="loader-text">{message}</div>
            <style>
            .loader-text {{
                display: inline-flex;
                align-items: center;
                margin: 0;
                padding: 0;
                line-height: 1.2;
                animation: pulse 3s infinite;
            }}
            @keyframes pulse {{
                0% {{ opacity: 0.3; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.3; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    
    
    # --- STREAMING LOGIC ---
    # 4. Create containers for live output
    
    # Placeholder for the final, streamed text response
    final_answer_placeholder = ai_message_container.empty()
    final_answer_text = ""
    
    # # Placeholder for the live trace (Thought/Action/Observation)
    # status_container = ai_message_container.empty()

    agent_trace_output = ""
    
    # Initialize variables for output storage
    output_text = ""
    
    # 5. Run the agent using .stream() (non-blocking yield)
    try:
        update_status("Checking your query with guardrails...")
        time.sleep(0.1)
        update_status("✅ Guardrails passed! Moving forward...")

        stream = st.session_state.rag_agent.stream({"input": payload})

        for chunk in stream:
            # 5a. Handle Intermediate Steps (Logs/Trace)
            if 'intermediate_steps' in chunk:
                # AgentExecutor returns list of (AgentAction, Observation) tuples
                for step in chunk['intermediate_steps']:
                    tool_call = step[0]
                    tool_result = step[1]
                    print(f"Print Tool call starts --------- \nType:{type(tool_call)}\nToolCall:{tool_call}-----ENDS------\n\n")
                    # Append new trace steps, formatting them like the verbose console output
                    agent_trace_output += f'> Entering Tool Call sequence\n'
                    # The 'log' property contains the Thought: and Action: text
                    agent_trace_output += tool_call.log
                    agent_trace_output += f'\nObservation: {tool_result}\n'
                    agent_trace_output += f'> Finished Tool Call sequence\n'
                    
                # # Update the trace placeholder IMMEDIATELY
                # trace_text_placeholder.code(agent_trace_output, language='text')
                # preparing RAG agent
                update_status("📖 Processing your query with knowledge base...")
                time.sleep(0.1)
                # when agent is thinking
                update_status("🤔 Thinking it through...")
                time.sleep(0.01)
                # while final output is being streamed
                update_status("💡 Generating final answer...")
                

            # 5b. Handle Final Answer
            if 'output' in chunk:
                # The 'output' contains the final answer text chunked
                final_answer_text += chunk['output']
                # Update the final answer placeholder IMMEDIATELY
                # final_answer_placeholder.markdown(final_answer_text)
                output_text = final_answer_text # Capture final text
                
    except Exception as e:
        output_text = f"An error occurred while running the agent, Please try again!!"
        st.error(output_text)
        print(f"Agent Error: {e}")
            
    # 6. Save AI turn in memory
    # We use the final captured output_text and the concatenated trace
    st.session_state.memory.chat_memory.add_message(AIMessage(content=output_text))
    st.session_state.trace_history.append(agent_trace_output)

    # 7. CLEAN UP: Reset loading state, clear temporary query, and force a rerun
    st.session_state.current_query = None
    st.session_state.is_loading = False
    st.rerun()