import os
import tempfile
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_use_double_quant=True,  
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
    )
    
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def create_qa_chain(llm, vectorstore):
    prompt_template = """
    Answer the question based on your knowledge. Use the following context to help:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa

if __name__ == "__main__":
    file_path = input("Enter the path to your PDF file: ")
    
    if not os.path.exists(file_path):
        print("Error: File not found")
        exit()

    print("Processing PDF...")
    vectorstore = process_pdf(file_path)
    print("PDF processed successfully")

    print("Loading model...")
    llm = load_model()
    print("Model loaded successfully")

    qa_chain = create_qa_chain(llm, vectorstore)

    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        print("Generating answer...")
        result = qa_chain.invoke({"query": question})
        print("Answer:", result["result"])
        print("\n")

    print("Thank you for using the PDF RAG Query System")

