import streamlit as st

# Configure the app settings (Important: only run once)
st.set_page_config(
    page_title="RAG App",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.title("Hey, welcome!!")

st.markdown("""
### Get Started
Please use the **sidebar on the left** to navigate between the application pages:
1.  **Chat**: Engage in a conversational chat with the RAG Agent.
2.  **Pdf Viewer**: View the documents that feed the RAG system.
""")