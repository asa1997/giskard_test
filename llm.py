from langchain_aws import BedrockLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

# Step 1: Prepare vector store with a PDF document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
documents = loader.load_and_split(text_splitter)

# Step 2: Use HuggingFace embeddings (no need for OpenAI API key)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, hf_embeddings)

# Step 3: Define the prompt template
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant.
Your task is to answer questions based on the IPCC Climate Change Report (2023).
Please be concise and informative.

Context:
{context}

Question:
{question}

Your answer:
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Step 4: Initialize the Bedrock LLM with the correct model ID
llm = BedrockLLM(
    model_id="anthropic.claude-v2",  # Update to the correct model ID
    model_kwargs={"temperature": 0},
    client=boto3.client("bedrock-runtime", region_name="us-east-1")
)

# Step 5: Set up the QA chain
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Step 6: Test with a question
question = "What are the impacts of climate change on agriculture?"
response = climate_qa_chain.run(question)
print(response)
