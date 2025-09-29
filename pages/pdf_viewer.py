import streamlit as st
import os
import base64

try:
    from streamlit_pdf_viewer import pdf_viewer
    PDF_VIEWER_INSTALLED = True
except ImportError:
    # Fallback flag if the component isn't installed
    PDF_VIEWER_INSTALLED = False
    st.warning("For best performance with large PDFs, please install the viewer: `pip install streamlit-pdf-viewer`")


# --- Configuration and Initialization ---
st.set_page_config(page_title="PDF Viewer", layout="wide")
st.title("📂 PDF Explorer")

PDF_DIR = "./data/pdfs"

# Initialize session state for the selected PDF if it doesn't exist
if 'selected_pdf' not in st.session_state:
    st.session_state['selected_pdf'] = None

# --- Function to get PDF files ---
def get_pdf_files():
    """Lists all PDF files in the specified directory."""
    if os.path.exists(PDF_DIR):
        return [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    return []

# --- Button to list PDFs and Selector ---
pdf_files = get_pdf_files()

if not pdf_files:
    st.warning(f"No PDFs found in the folder '{PDF_DIR}' or the folder does not exist.")
else:
    if st.button("📑 Show/Refresh Available PDFs"):
        st.rerun() 
    
    st.subheader("Choose a PDF to preview:")
    
    # Get the index of the currently selected file for persistence
    try:
        default_index = pdf_files.index(st.session_state['selected_pdf']) if st.session_state['selected_pdf'] in pdf_files else 0
    except ValueError:
        default_index = 0
        
    selected_pdf = st.selectbox(
        "Available PDFs:",
        pdf_files,
        index=default_index,
        key='selectbox_pdf'
    )
    
    # Update session state whenever the selectbox value changes
    st.session_state['selected_pdf'] = selected_pdf

# --- pdf Preview Section ---
if st.session_state['selected_pdf']:
    selected_pdf = st.session_state['selected_pdf']
    pdf_path = os.path.join(PDF_DIR, selected_pdf)

    st.markdown("---")
    st.subheader(f"📄 Previewing: **{selected_pdf}**")

    try:
        # Read the file for the Download button (required outside the viewer)
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_data,
            file_name=selected_pdf,
            mime="application/pdf"
        )
        
        if PDF_VIEWER_INSTALLED:
            # The component takes the file path directly for efficient rendering
            pdf_viewer(pdf_path, width=800, height=700)
            
        else:
            # Fallback to the slow base64 method if the component is missing
            st.error("Using fallback method. Please install 'streamlit-pdf-viewer' for better performance.")
            base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
    except FileNotFoundError:
        st.error(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {e}")