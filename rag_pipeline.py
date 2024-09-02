import os
import pdfplumber # type: ignore
import docx # type: ignore
from typing import Dict, Any
from langchain_community.llms import Predibase# type: ignore
from langchain_community.vectorstores import FAISS# type: ignore
from langchain.chains import RetrievalQA# type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter# type: ignore
import torch# type: ignore
import time
from langchain.embeddings.base import Embeddings# type: ignore
from transformers import AutoTokenizer, AutoModel# type: ignore
from dotenv import load_dotenv

load_dotenv()

class InLegalBERTEmbeddings(Embeddings):
    def __init__(self, model_name="law-ai/InLegalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

class RAGPipeline:
    def __init__(self, embedding_model_name: str, llm_model_name: str, api_token: str):
        print("Initializing RAGPipeline...")
        
        print(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = self._create_embeddings(embedding_model_name)
        print("Embedding model loaded successfully.")
        
        print(f"Initializing Predibase model: {llm_model_name}")
        self.llm_model = self._create_predibase_model(llm_model_name, api_token)
        print("Predibase model initialized successfully.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        print("Text splitter initialized.")
        
        print("RAGPipeline initialization complete.")

    @staticmethod
    def _create_embeddings(model_name: str) -> InLegalBERTEmbeddings:
        return InLegalBERTEmbeddings(model_name)

    @staticmethod
    def _create_predibase_model(model_name: str, api_token: str) -> Predibase:
        return Predibase(
            model=model_name,
            predibase_api_key=api_token,
            predibase_sdk_version=None,
            adapter_version=1,
            max_new_tokens=4000,
        )

    def extract_text_from_file(self, file_path: str) -> str:
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        elif file_extension.lower() in ['.doc', '.docx']:
            doc = docx.Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension.lower() in ['.txt']:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def create_vector_store(self, text: str) -> FAISS:
        documents = self.text_splitter.create_documents([text])
        return FAISS.from_documents(documents, embedding=self.embed_model)

    def create_rag_chain(self, vector_store: FAISS) -> RetrievalQA:
        retriever = vector_store.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type='stuff',
            retriever=retriever
        )

    @staticmethod
    def generate_prompt(self, system_message: str, query: str = "") -> str:
        return f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Query: ```{query}```
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    def run_rag_pipeline(self, rag_chain: RetrievalQA, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        response = rag_chain.invoke(prompt)
        end_time = time.time()
        
        return {
            "result": response['result'],
            "execution_time": end_time - start_time
        }

def process_folder(folder_path: str, rag_pipeline: RAGPipeline, system_message: str, query: str):
    prompt = rag_pipeline.generate_prompt(system_message, query)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                print(f"Processing file: {filename}")
                
                # Extract text from file
                text = rag_pipeline.extract_text_from_file(file_path)
                
                # Create vector store for this file
                vector_store = rag_pipeline.create_vector_store(text)
                
                # Create RAG chain for this file
                rag_chain = rag_pipeline.create_rag_chain(vector_store)
                
                # Process the prompt for this file
                result = rag_pipeline.run_rag_pipeline(rag_chain, prompt)
                
                print(f"Result: {result['result']}")
                print(f"Execution Time: {result['execution_time']} seconds")
                print("---\n")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                print("---\n")

def main():
    # Configuration
    PREDIBASE_API_TOKEN = os.getenv("PREDIBASE_TOKEN", "")
    LLM_MODEL_NAME = "llama-3-1-8b-instruct"
    EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"
    
    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline(EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, PREDIBASE_API_TOKEN)
    
    # Folder path containing documents
    FOLDER_PATH = "/home/ubuntu/experiments_sajal/law_assets"
    
    # Single system message and query for all documents
    SYSTEM_MESSAGE = "You are an advanced AI assistant designed to analyze and summarize various types of documents."
    QUERY = "Provide a brief summary of the document and identify its main purpose."
    
    # Process all documents in the folder
    process_folder(FOLDER_PATH, rag_pipeline, SYSTEM_MESSAGE, QUERY)

if __name__ == "__main__":
    main()
