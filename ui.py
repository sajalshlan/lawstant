import streamlit as st
import os
import io
import anthropic
from typing import List
from dotenv import load_dotenv
import pdfplumber
import docx

load_dotenv()

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

def extract_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.pdf':
        try:
            pdf_content = io.BytesIO(file.read())
            file.seek(0)  # Reset file pointer
            
            with pdfplumber.open(pdf_content) as pdf:
                return " ".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
    elif file_extension in ['.doc', '.docx']:
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        except BadZipFile:
            st.error(f"Error: The file {file.name} is not a valid Word document.")
            return ""
        except Exception as e:
            st.error(f"Error processing Word document: {e}")
            return ""
    elif file_extension == '.txt':
        return file.read().decode('utf-8')
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return ""
def display_document_content(file):
    try:
        text = extract_text_from_file(file)
        st.text_area("Document Content", text, height=300)
    except Exception as e:
        st.error(f"Error displaying {file.name}: {str(e)}")

def claudeCall(text, prompt):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature=0,
            system="You are a general counsel of a fortune 500 company.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        st.error(f"An error occurred while calling Claude API: {e}")
        return ""

    text_content = ''.join(block.text for block in message.content if block.type == 'text')
    return text_content

def main():
    st.set_page_config(page_title="Order! Order!", layout="wide")
    
    st.title("Order! Order!")
    st.markdown("Upload your legal documents for quick analysis and summary. Supports PDF, DOCX, DOC, TXT.")

    st.success("All systems online!")

    # Initialize session state variables
    if "extracted_texts" not in st.session_state:
        st.session_state.extracted_texts = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "summary_expanded" not in st.session_state:
        st.session_state.summary_expanded = False
    if "risky_analysis_expanded" not in st.session_state:
        st.session_state.risky_analysis_expanded = False
    if "ask_assistant_expanded" not in st.session_state:
        st.session_state.ask_assistant_expanded = False

    # Top options for single/multiple contracts
    analysis_type = st.radio("Select analysis type:", ["Single Contract", "Multiple Contracts"], horizontal=True)

    # File uploader
    uploaded_files = st.file_uploader("Upload document(s)", accept_multiple_files=(analysis_type == "Multiple Contracts"), type=['pdf', 'docx', 'doc', 'txt'])

    # Convert to list if single file
    if analysis_type == "Single Contract" and uploaded_files:
        uploaded_files = [uploaded_files]

    if uploaded_files:
        # Display uploaded documents and extract text
        st.subheader("Uploaded Document(s)")
        for file in uploaded_files:
            with st.expander(f"View {file.name}"):
                text = extract_text_from_file(file)
                if text:
                    st.text_area("Document Content", text, height=300)
                else:
                    st.warning(f"Could not extract text from {file.name}. Please check if the file is valid and in a supported format.")
            # Store extracted text
            st.session_state.extracted_texts[file.name] = text

    # Sidebar
    st.sidebar.title("Analysis Options")

    # Summary Section
    if st.sidebar.button("Summary"):
        st.session_state.summary_expanded = not st.session_state.summary_expanded
    
    if st.session_state.summary_expanded:
        with st.sidebar.expander("Summary", expanded=True):
            if uploaded_files:
                st.write("Generating summary...")
                for file in uploaded_files:
                    text = st.session_state.extracted_texts.get(file.name, "")
                    if text:
                        summary = claudeCall(text, "Provide a brief summary of this document.")
                        st.subheader(f"Summary of {file.name}")
                        st.write(summary)
                    else:
                        st.warning(f"Could not extract text from {file.name}. Please check if the file is corrupted or in an unsupported format.")
            else:
                st.warning("Please upload a document first.")

    # Risky Analysis Section
    if st.sidebar.button("Risky Analysis"):
        st.session_state.risky_analysis_expanded = not st.session_state.risky_analysis_expanded
    
    if st.session_state.risky_analysis_expanded:
        with st.sidebar.expander("Risky Analysis", expanded=True):
            if uploaded_files:
                st.write("Performing risky analysis...")
                for file in uploaded_files:
                    text = st.session_state.extracted_texts.get(file.name, "")
                    if text:
                        analysis = claudeCall(text, "Identify and explain any potentially risky clauses or terms in this document.")
                        st.subheader(f"Risky Analysis of {file.name}")
                        st.write(analysis)
                    else:
                        st.warning(f"Could not extract text from {file.name}. Please check if the file is corrupted or in an unsupported format.")
            else:
                st.warning("Please upload a document first.")

    # Ask Assistant Section
    if st.sidebar.button("Ask Assistant"):
        st.session_state.ask_assistant_expanded = not st.session_state.ask_assistant_expanded
    
    if st.session_state.ask_assistant_expanded:
        with st.sidebar.expander("Ask Assistant", expanded=True):
            st.write("Chat with the AI Assistant about the uploaded document(s).")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            prompt = st.text_input("What would you like to know about the document(s)?", key="chat_input", value=st.session_state.current_prompt)
            send_button = st.button("Send")

            # Check if Enter was pressed (prompt changed) or Send button was clicked
            if prompt != st.session_state.current_prompt or send_button:
                if prompt.strip():  # Ensure the prompt is not empty
                    st.session_state.current_prompt = prompt  # Update the current prompt
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        if uploaded_files:
                            full_text = "\n\n".join(st.session_state.extracted_texts.values())
                            
                            if full_text.strip():
                                response = claudeCall(full_text, prompt)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.warning("Could not extract text from the uploaded document(s). Please ensure the files are in a supported format and contain readable text.")
                                st.session_state.messages.append({"role": "assistant", "content": "Could not extract text from the uploaded document(s). Please ensure the files are in a supported format and contain readable text."})
                        else:
                            st.warning("Please upload a document first.")
                            st.session_state.messages.append({"role": "assistant", "content": "Please upload a document first."})
                    
                    # Clear the input after sending
                    st.session_state.current_prompt = ""

if __name__ == "__main__":
    main()