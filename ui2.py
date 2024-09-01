import streamlit as st # type: ignore
import os
import tempfile
import pandas as pd # type: ignore
from typing import List, Tuple
from rag_pipeline import RAGPipeline  
from dotenv import load_dotenv

load_dotenv()

# Workaround for OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
PREDIBASE_API_TOKEN = os.getenv("PREDIBASE_TOKEN", "")
LLM_MODEL_NAME = "llama-3-1-8b-instruct"
EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"

# Define the fields as a global variable
ANALYSIS_FIELDS: List[Tuple[str, str]] = [
    ("Document Type", "What type of document is this, answer in one word"),
    ("Confidentiality Level", "What is the confidentiality level of this legal document, in one word?"),
    ("Parties", "Can you tell me the parties involved"),
    ("Severance", "How many pages are in there"),
    ("Summary", "Provide a summary of this legal document, in short"),
]

def process_document(file, rag_pipeline: RAGPipeline, fields: List[Tuple[str, str]]):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    try:
        # Extract text from file
        text = rag_pipeline.extract_text_from_file(temp_file_path)
        
        # Create vector store for this file
        vector_store = rag_pipeline.create_vector_store(text)
        
        # Create RAG chain for this file
        rag_chain = rag_pipeline.create_rag_chain(vector_store)
        
        file_results = {"File Name": file.name}
        for field_name, query in fields:
            prompt = rag_pipeline.generate_prompt(
                "You are an AI assistant specialized in analyzing legal and business documents.",
                query
            )
            result = rag_pipeline.run_rag_pipeline(rag_chain, prompt)
            file_results[field_name] = result['result']
        
        return file_results
    
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None
    
    finally:
        os.unlink(temp_file_path)

def display_document_card(doc, fields: List[Tuple[str, str]]):
    st.markdown(f"### {doc['File Name']}")
    
    # Display all fields except the last one in columns, two per row
    for i in range(0, len(fields) - 1, 2):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{fields[i][0]}:** {doc[fields[i][0]]}")
        if i + 1 < len(fields) - 1:
            with col2:
                st.markdown(f"**{fields[i+1][0]}:** {doc[fields[i+1][0]]}")
    
    # Add an expander for the last field
    last_field = fields[-1]
    with st.expander(f"View {last_field[0]}"):
        st.write(doc[last_field[0]])
    
    st.markdown("---")

def main():
    st.set_page_config(page_title="Better Call Saul", layout="wide")
    
    # Create a centered column with reduced width
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:  # This is our main content area
        st.title("Better Call Saul")
        st.markdown("Upload your legal documents for quick analysis and summary. Supports pdf, docx, doc, txt")

        # Initialize RAG pipeline
        @st.cache_resource
        def load_rag_pipeline():
            return RAGPipeline(EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, PREDIBASE_API_TOKEN)

        rag_pipeline = load_rag_pipeline()
        st.success("All systems online!")

        # File uploader
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])

        if uploaded_files:
            if st.button("Process Documents"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                st.subheader("Document Analysis Results")
                results_container = st.container()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    result = process_document(file, rag_pipeline, ANALYSIS_FIELDS)
                    if result:
                        results.append(result)
                        with results_container:
                            display_document_card(result, ANALYSIS_FIELDS)
                        
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)

                status_text.text("All documents processed!")
                progress_bar.empty()

                # Create a DataFrame for CSV download
                if results:
                    df = pd.DataFrame(results)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name="document_analysis_results.csv",
                        mime="text/csv",
                    )

if __name__ == "__main__":
    main()