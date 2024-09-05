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
import pandas as pd
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
    """,

    "Key Dates": """
        As the General Counsel of a Fortune 500 company,, review the document and identify the key dates present.
    """,

    "Next Action Items": """
        As the General Counsel, review the document and identify any next action items, such as deadlines, required actions, or agreement renewals. Summarize the key next steps in a concise manner.
    """,

    "Obligations Summary": """
        As the General Counsel of a Fortune 500 company, your expertise is crucial in summarizing obligations outlined in legal documents.

    """,
    
    "Term and Validity": """
        You are the highly experienced General Counsel of a Fortune 500 company, responsible for ensuring clarity in contractual terms.
    """,
    
    "Termination Rights": """
        Identify the parties who have the right to terminate the agreement, as well as the conditions under which termination can occur. Provide a concise summary.
    """,
    
    "Termination Consequences": """
        Analyze the potential consequences of terminating the agreement, such as penalties, liabilities, or ongoing obligations. Summarize the key points.
    """,
    
    "Personal Data Handling": """
        Assess whether the document contains any provisions related to the handling of personal information. Indicate yes or no, and provide a brief explanation if necessary.
    """,
    
    "Document Classification": """
        Determine the overall classification of the document, whether it is a constitutional, operational, or financial document. Provide a single-word answer.
    """,
    
    "Listed Company Relevance": """
        As the General Counsel of a Fortune 500 company, it's critical to identify whether a document pertains to a listed company.
    """,
    
    "Price Sensitivity": """
        Evaluate if the document contains any price-sensitive information.
    """,

    "Potential Breach": """
        Carefully review the provided document and identify any clauses that could potentially constitute a breach of the agreement. Specifically, look for the following red flag clauses:

        1. Unilateral Termination Clauses
        2. Indemnity Clauses with Unlimited Liability
        3. Excessive Penalty Clauses
        4. Non-Compete Clauses
        5. Broad Confidentiality Clauses
        6. Automatic Renewal Clauses
        7. Ambiguous or Vague Terms
        8. Governing Law and Jurisdiction Clauses Unfavorable to Your Company
        9. Force Majeure Clauses That Are Too Restrictive
        10. Overly Broad Assignment Clauses
        11. Clauses Requiring Waiver of Rights
        12. Intellectual Property Clauses That Grant Excessive Rights
        13. Non-Solicitation Clauses That Are Too Broad
        14. Payment Terms That Are Too Strict or Unfavorable
        15. Clauses Limiting Liability in an Unfair Manner
        16. Clauses Requiring Use of Specific Vendors or Suppliers
        17. Clauses with Hidden Costs or Obligations
    """

}

ANALYSIS_FIELDS: List[Tuple[str, str]] = [
    ("Document-Type", "Classify the provided document in one of the categories, without any additional explanation."),
    ("Confidentiality-Level", "Provide your assessment of the confidentiality level for the provided document as a single word answer, without any additional explanation."),
    ("Document Classification", "Determine the overall classification of the document (constitutional, operational, or financial)."),
    ("Listed Company Relevance", "Assess whether the document is specifically related to a listed company. IF present, provide a yes and companies name and if not present, provide no. "),
    ("Price Sensitivity", "Evaluate if the document contains any price-sensitive information. Provide yes or no answer."),
    ("Summary", "Given the document provided, generate a brief summary that highlights the key points and the main purpose that would be important for a quick understanding of the document's content and implications. Keep the summary to the point."),
    ("Key Dates", "Identify the key dates mentioned in the document (e.g., effective date, expiration date, important deadlines) and provide them in a numeric format (MM/DD/YYYY) and the event."),
    ("Next Action Items", "Summarize any next action items, deadlines, or agreement renewals identified in the document."),
    ("Obligations Summary", "Provide a concise summary of the key obligations outlined in the document."),
    ("Term and Validity", "Summarize the term or validity period of the agreement, including any start and end dates. if not present, just answer not present."),
    ("Termination Rights", "Identify the parties with the right to terminate the agreement and the conditions for termination."),
    ("Termination Consequences", "Summarize the key potential consequences of terminating the agreement."),
    ("Personal Data Handling", "Indicate whether the document contains provisions related to the handling of personal information. IF present, provide a yes and the personal details present and if not present, provide no."),
    ("Potential Breach", "Summarize any clauses in the document that could potentially constitute a breach of the agreement, based on the provided list of red flag clauses. Keep the summary concise and to the point."),
]


PROMPTS_GPT = {
    "Document-Type": """
        You are the highly experienced General Counsel of a Fortune 500 company, specializing in legal document analysis.
        Your task is to carefully analyze the content and structure of the document, and classify it into one of the following categories:

        Intellectual Property Agreement
        Commercial Contract
        Loan Agreement
        Regulatory Filing
        Non-Disclosure Agreement
        Partnership Agreement
        Legal Opinion
        Authorization Document
        Tax Compliance Document
        Audit Report
        Investment Agreement
        Resolution Plan
        Employee Loan Agreement
        Employment Contract
    """,

    "Confidentiality-Level": """
        You are the highly experienced General Counsel of a Fortune 500 company, responsible for handling sensitive legal documents.
        I have provided you with a legal document. Your task is to carefully analyze the content and context of the document, and determine its appropriate confidentiality level, which can be one of the following:

        Public
        Confidential
        Highly Confidential
    """,

    "Summary": """
        As the General Counsel of a Fortune 500 company, your role involves distilling complex legal documents into accurate summaries.
        Given the document provided, generate a brief summary that highlights the key points and the main purpose that would be important for a quick understanding of the document's content and implications. Keep the summary to the point.
    """,

    "Next-Action-Items": """
        You are the highly experienced General Counsel of a Fortune 500 company, with a keen eye for identifying key actions in legal documents.    
    """,

    "Obligations-Summary": """
        As the General Counsel of a Fortune 500 company, your expertise is crucial in summarizing obligations outlined in legal documents.
    """,

    "Term": """
        You are the highly experienced General Counsel of a Fortune 500 company, responsible for ensuring clarity in contractual terms.
    """,

    "Termination-Rights": """
        As the General Counsel of a Fortune 500 company, you need to be vigilant about termination clauses.
    """,

    "Consequences-of-Termination": """
        You are the highly experienced General Counsel of a Fortune 500 company, adept at understanding the implications of contract termination.
    """,

    "Personal-Information-Captured": """
        As the General Counsel of a Fortune 500 company, privacy and data protection are paramount in your legal evaluations.
    """,

    "Meta-Classification": """
        You are the highly experienced General Counsel of a Fortune 500 company, skilled in categorizing documents based on their nature.
    """,

    "Listed-Company-Relevance": """
        As the General Counsel of a Fortune 500 company, it's critical to identify whether a document pertains to a listed company.
    """,

    "Price-Sensitive-Information": """
        You are the highly experienced General Counsel of a Fortune 500 company, well-versed in securities regulations.
    """,

    "Potential Breach": """
        You are the highly experienced General Counsel of a Fortune 500 company, well-versed in identifying any clauses that could potentially constitute a breach of the agreement. Carefully review the provided document and identify any clauses that could potentially constitute a breach of the agreement. Specifically, look for the following red flag clauses:

        1. Unilateral Termination Clauses
        2. Indemnity Clauses with Unlimited Liability
        3. Excessive Penalty Clauses
        4. Non-Compete Clauses
        5. Broad Confidentiality Clauses
        6. Automatic Renewal Clauses
        7. Ambiguous or Vague Terms
        8. Governing Law and Jurisdiction Clauses Unfavorable to Your Company
        9. Force Majeure Clauses That Are Too Restrictive
        10. Overly Broad Assignment Clauses
        11. Clauses Requiring Waiver of Rights
        12. Intellectual Property Clauses That Grant Excessive Rights
        13. Non-Solicitation Clauses That Are Too Broad
        14. Payment Terms That Are Too Strict or Unfavorable
        15. Clauses Limiting Liability in an Unfair Manner
        16. Clauses Requiring Use of Specific Vendors or Suppliers
        17. Clauses with Hidden Costs or Obligations
    """
}

ANALYSIS_FIELDS_GPT: List[Tuple[str, str]] = [
    ("Document-Type", "Classify the provided document in one of the categories, without any additional explanation."),
    ("Confidentiality-Level", "Provide your assessment of the confidentiality level for the provided document as a single word answer, without any additional explanation."),
    ("Summary", "Given the document provided, generate a brief summary that highlights the key points and the main purpose that would be important for a quick understanding of the document's content and implications. Keep the summary to the point."),
    ("Next-Action-Items", "Summarize all the items where an action needs to be taken, including deadlines, renewals, and other time-sensitive actions."),
    ("Obligations-Summary", "Provide a concise summary of all the obligations of the parties involved, highlighting any critical responsibilities or commitments."),
    ("Term", "Specify the validity period, including the start and end dates of the contract."),
    ("Termination-Rights", "Summarize the termination rights, clearly stating which party holds the termination rights and under what conditions these rights can be exercised."),
    ("Consequences-of-Termination", "Provide a summary of the consequences of termination, detailing the obligations, penalties, or any significant impacts."),
    ("Personal-Information-Captured", "Determine whether the document captures any personal information. Provide a 'Yes' or 'No' answer."),
    ("Meta-Classification", "Classify the document as either a Constitutional Document, Operational Document, or Financial Document."),
    ("Listed-Company-Relevance", "State whether the document pertains to a listed company with a simple 'Yes' or 'No' answer."),
    ("Price-Sensitive-Information", "Determine whether the document contains any price-sensitive information. Provide a 'Yes' or 'No' answer."),
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

def truncate_text(text, max_chars=100):
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def display_field(field_name, field_value):
    st.markdown(f"***{field_name}***")
    if len(field_value) <= 100:
        st.markdown(field_value)
    else:
        with st.expander("View full content"):
            st.write(field_value)

def display_document_card(doc, fields: List[Tuple[str, str]]):
    st.markdown(f"### {doc['File Name']}")
    
    # Create a table-like structure
    num_fields = len(fields)
    col_count = min(num_fields, 4)
    
    # Create a table-like structure
    cols = st.columns(col_count)
    
    for i in range(0, len(fields), col_count):
        # Create columns with gaps between them
        cols = st.columns([1, 0.1] * (col_count - 1) + [1])  # Adding 0.1 as a gap between columns

        for j in range(col_count):
            if i + j < len(fields):
                field_name, _ = fields[i + j]
                with cols[j * 2]:
                    display_field(field_name, doc[field_name])
        
        # Add a gap between rows
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")

def main():
    st.set_page_config(page_title="Burden of Proof", layout="wide")
    
    st.title("Order! Order!")
    st.markdown("Reasons I am Mike Ross - I have photogenic memory, I am fast, I don't have a law degree.")
    st.markdown("Upload your legal documents for quick analysis and summary. Supports pdf, docx, doc, txt. Max Page Limit on Documents - Scanned: 25; Digital: None")

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
            st.markdown("<br>", unsafe_allow_html=True)
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
            st.markdown("<br>", unsafe_allow_html=True)
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
