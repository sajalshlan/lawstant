import streamlit as st
import os
import pandas as pd
from typing import List, Tuple
from rag_pipeline import RAGPipeline 
from dotenv import load_dotenv
import io
from pdf2image import convert_from_bytes
import docx
import pdfplumber

load_dotenv()

# Workaround for OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
PREDIBASE_API_TOKEN = os.getenv("PREDIBASE_TOKEN")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
LLM_MODEL_NAME = "llama-3-1-8b-instruct"
EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"

PROMPTS = {
    "Document-Type": """
        You are the highly experienced General Counsel of a Fortune 500 company, specializing in legal document analysis.
        Your task is to carefully analyze the content and structure of the documents, and classify it into one of the following categories:

        * Intellectual Property Agreement
        * Commercial Contract
        * Loan Agreement
        * Regulatory Filing
        * Non-Disclosure Agreement
        * Partnership Agreement
        * Legal Opinion
        * Authorization Document
        * Tax Compliance Document
        * Audit Report
        * Investment Agreement
        * Resolution Plan
        * Employee Loan Agreement
        * Employment Contract
    """,
    
    "Confidentiality-Level": """
        You are the highly experienced General Counsel of a Fortune 500 company, responsible for handling sensitive legal documents.
        I have provided you with a legal document. Your task is to carefully analyze the content and context of the document, and determine its appropriate confidentiality level, which can be one of the following:

        1. Public
        2. Confidential
        3. Highly Confidential
    """,

    "Summary": """
        As the General Counsel of a Fortune 500 company, your role involves distilling complex legal documents into accurate summaries.
    """
}

ANALYSIS_FIELDS: List[Tuple[str, str]] = [
    ("Document-Type", "Classify the provided document in one of the categories, without any additional explanation."),
    ("Confidentiality-Level", "Provide your assessment of the confidentiality level for the provided document as a single word answer, without any additional explanation."),
    ("Summary", "Given the document provided, generate a brief summary that highlights the key points and the main purpose that would be important for a quick understanding of the document's content and implications. Keep the summary to the point."),
]

def extract_text_from_file(file, rag_pipeline: RAGPipeline):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.pdf':
        try:
            # Check if the PDF is digitally native
            pdf_content = io.BytesIO(file.read())
            file.seek(0)  # Reset file pointer
            
            
            with pdfplumber.open(pdf_content) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text() or ""
                if len(text) > 100:
                    # PDF is digitally native, use pdfplumber
                    return " ".join([page.extract_text() or "" for page in pdf.pages])
                else:
                    # PDF needs OCR
                    images = convert_from_bytes(file.read())
                    image_files = []
                    for i, image in enumerate(images[:25]):  # Limit to 25 pages for OCR
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        image_files.append(('image', ('image.jpg', img_byte_arr, 'image/jpeg')))
                    
                    ocr_text = rag_pipeline.analyze_images(image_files)
                    return ocr_text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
    elif file_extension in ['.doc', '.docx']:
        doc = docx.Document(io.BytesIO(file.read()))
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension in ['.txt']:
        return file.read().decode('utf-8')
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def process_document(file, rag_pipeline: RAGPipeline, fields: List[Tuple[str, str]]):
    try:
        # Extract text from file (now includes OCR functionality)
        text = extract_text_from_file(file, rag_pipeline)
        
        # Create vector store for this file
        vector_store = rag_pipeline.create_vector_store(text)
        
        # Create RAG chain for this file
        rag_chain = rag_pipeline.create_rag_chain(vector_store)
        
        file_results = {"File Name": file.name}
        for field_name, query_key in fields:
            system_message = PROMPTS.get(field_name, "You are the highly experienced General Counsel of a Fortune 500 company.")
            prompt = rag_pipeline.generate_prompt(system_message, query_key)
            result = rag_pipeline.run_rag_pipeline(rag_chain, prompt)
            file_results[field_name] = result['result']
        
        return file_results
    
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def display_document_card(doc, fields: List[Tuple[str, str]]):
    st.markdown(f"### {doc['File Name']}")
    
    # Display all fields except the last one in columns, two per row
    for i in range(0, len(fields) - 1, 2):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"*{fields[i][0]}:* {doc[fields[i][0]]}")
        if i + 1 < len(fields) - 1:
            with col2:
                st.markdown(f"*{fields[i+1][0]}:* {doc[fields[i+1][0]]}")
    
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
        st.title("Order! Order!")
        st.markdown("Upload your legal documents for quick analysis and summary. Supports pdf, docx, doc, txt. Max Page Limit - Scanned PDFs: 25; Digital PDfs: None")

        # Initialize RAG pipeline
        @st.cache_resource
        def load_rag_pipeline():
            return RAGPipeline(EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, PREDIBASE_API_TOKEN, GOOGLE_VISION_API_KEY)

        rag_pipeline = load_rag_pipeline()
        st.success("All systems online!")

        # File uploader
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'docx', 'doc', 'txt'])

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
                    csv = df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name="document_analysis_results.csv",
                        mime="text/csv",
                    )

if __name__ == "__main__":
    main()
